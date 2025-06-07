
import os
import sys

# Ensure iou3d is importable before any lidar_det imports
sys.path.append('/content/DCCLA')
sys.path.append('/content/DCCLA/lib/iou3d')

import glob
import json
import argparse
import shutil

import torch
import numpy as np
import open3d as o3d
import pandas as pd
import yaml

from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN

# Utility converters
def to_numpy(x):
    return x.cpu().numpy() if torch.is_tensor(x) else np.array(x)

def to_float(x):
    return x.cpu().item() if torch.is_tensor(x) else float(x)


# Helper: if a directory has exactly one subfolder (ignoring __MACOSX), return it.
# Otherwise return the directory itself.
def get_single_subdir(root_dir: str) -> str:
    entries = os.listdir(root_dir)
    subdirs = [
        d for d in entries
        if os.path.isdir(os.path.join(root_dir, d)) and d != "__MACOSX"
    ]
    if len(subdirs) == 1:
        return os.path.join(root_dir, subdirs[0])
    return root_dir

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="JRDB pipeline: detect (RPEA or DCCLA) + grouping + GT refinement"
    )
    parser.add_argument(
        "--model",
        choices=["rpea", "dccla"],
        required=True,
        help="Which detector to use: 'rpea' or 'dccla'"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to JRDB JSON (e.g. /content/clark-center-2019-02-28_0.json)"
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default=None,
        help="If provided, unzip this archive into data_dir (expected to contain .pcd files)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory where .pcd files live (or where zip_path should be extracted)"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to detector checkpoint (RPEA or DCCLA). Required."
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=2.0,
        help="DBSCAN epsilon distance for grouping"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/temp_per_frame",
        help="Directory to store per-frame detection files (temporary)"
    )
    parser.add_argument(
        "--det_output_file",
        type=str,
        default="/content/det.txt",
        help="Aggregated detection output file"
    )
    parser.add_argument(
        "--group_output_file",
        type=str,
        default="/content/group_detection.txt",
        help="Final group detection output file"
    )
    return parser.parse_args()


# Generate GT.txt from JRDB JSON
def generate_gt(json_path, gt_output_path):
    """
    Parse JRDB JSON to produce GT.txt with lines:
      <frame>,<orig_group_id>,<cx>,<cy>,<cz>,<l>,<w>,<h>,<local_group_label>
    """
    group_entries_per_frame = defaultdict(list)
    with open(json_path, "r") as f:
        data = json.load(f)

    for frame_file, objects in data["labels"].items():
        frame_number = int(frame_file.replace(".pcd", ""))
        for obj in objects:
            # Skip any “no_eval” objects
            if obj.get("attributes", {}).get("no_eval", True):
                continue

            box = obj.get("box", {})
            cx, cy, cz = box.get("cx"), box.get("cy"), box.get("cz")
            l, w, h = box.get("l"), box.get("w"), box.get("h")
            orig_gid = obj.get("social_group", {}).get("cluster_ID", None)
            if orig_gid is None:
                continue
            group_entries_per_frame[frame_number].append((orig_gid, cx, cy, cz, l, w, h))

    with open(gt_output_path, "w") as gt_file:
        for frame_number, dets in sorted(group_entries_per_frame.items()):
            # Count how many times each orig_gid appears
            counts = Counter([g[0] for g in dets])
            group_map = {}
            next_label = 1
            # If a group has ≥2 members, assign it a local_label ≥1; else 0
            for gid, cnt in counts.items():
                group_map[gid] = 0 if cnt < 2 else next_label
                if cnt >= 2:
                    next_label += 1

            for orig_gid, cx, cy, cz, l, w, h in dets:
                local_label = group_map[orig_gid]
                gt_file.write(f"{frame_number},{orig_gid},{cx},{cy},{cz},{l},{w},{h},{local_label}\n")

    print(f"[GT] Saved GT.txt to: {gt_output_path}")

# detection + grouping” (branches on model)
def run_realtime_grouping(
    model_name: str,
    ckpt_path: str,
    data_dir: str,
    output_dir: str,
    det_output_file: str,
    group_output_file: str,
    distance_threshold: float,
):
    os.makedirs(output_dir, exist_ok=True)
    # Remove old det/group files if they exist
    for f in [det_output_file, group_output_file]:
        if os.path.exists(f):
            os.remove(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "rpea":
        from lidar_det.detector import RPEA
        detector = RPEA(ckpt_path)
        detector._net.to(device)
        detector._net.eval()
    elif model_name == "dccla":
        from lidar_det import detector as _det_module
        from lidar_det.model import get_model
        from lidar_det.detector import dccla

        # Read the same YAML config that DCCLA expects:
        yaml_config_path = os.path.join(os.path.dirname(ckpt_path), "bin", "jrdb22.yaml")
        with open(yaml_config_path, "r") as f:
            cfg = yaml.safe_load(f)
        model_cfg = {
            "type": cfg["model"]["type"],
            "kwargs": cfg["model"]["kwargs"],
            "target_mode": cfg["model"]["target_mode"],
            "disentangled_loss": cfg["model"]["disentangled_loss"],
            "nuscenes": False
        }
        # Monkey‐patch get_model so dccla() uses our model_cfg
        _det_module.get_model = lambda _unused, inference_only=True: get_model(model_cfg, inference_only=inference_only)

        detector = dccla(ckpt_path, gpu=(device.type == "cuda"))
    else:
        raise ValueError(f"Incorrect model_name '{model_name}'. Must be 'rpea' or 'dccla'.")

    # Gather all .pcd files
    pcd_files = sorted(glob.glob(os.path.join(data_dir, "*.pcd")))
    print(f"[Detector:{model_name.upper()}] Found {len(pcd_files)} .pcd files in {data_dir}")

    for pcd_path in pcd_files:
        frame_name = os.path.splitext(os.path.basename(pcd_path))[0]
        try:
            frame_id = int(frame_name)
        except ValueError:
            continue

        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points)  # (N, 3)
        if pts.shape[0] == 0:
            print(f"[Detector:{model_name.upper()}] Frame {frame_id} empty; skipping.")
            continue

        # Prepare input: identical for both RPEA and DCCLA
        pc_input = pts.T.astype(np.float32)  # shape: (3, N)

        with torch.no_grad():
            boxes, scores = detector(pc_input)

        if boxes is None or len(boxes) == 0:
            continue

        # Save per-frame raw detections (x,y,z,l,w,h,heading,score)
        per_frame_file = os.path.join(output_dir, f"{frame_name}.txt")
        with open(per_frame_file, "w") as f_out:
            for box, score in zip(boxes, scores):
                b_np = to_numpy(box)      # [x, y, z, l, w, h, heading]
                s_val = to_float(score)
                f_out.write(" ".join(map(str, b_np.tolist() + [s_val])) + "\n")
        print(f"[Detector:{model_name.upper()}] Saved per-frame detections: {per_frame_file}")

        # Append valid detections to det_output_file
        valid_centers = []
        valid_scores = []
        filtered_boxes = []
        with open(per_frame_file, "r") as f_in, open(det_output_file, "a") as det_f:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) != 8:
                    continue
                x, y, z, l, w, h, heading, s_val = map(float, parts)
                if l <= 0 or w <= 0 or h <= 0 or s_val < 0.30:
                    continue
                det_f.write(
                    f"{frame_id},-1,"
                    f"{x:.4f},{y:.4f},{z:.4f},"
                    f"{l:.4f},{w:.4f},{h:.4f},"
                    f"{heading:.4f},{s_val:.4f}\n"
                )
                valid_centers.append((x, y, z))
                valid_scores.append(s_val)
                filtered_boxes.append((x, y, z, l, w, h, heading))

        print(f"[Detector:{model_name.upper()}] Frame {frame_id} valid for grouping: {len(valid_centers)} detections")
        if len(valid_centers) < 2:
            continue

        # DBSCAN on valid_centers
        centers_arr = np.vstack(valid_centers)
        clustering = DBSCAN(eps=distance_threshold, min_samples=2).fit(centers_arr)
        labels = clustering.labels_

        lbl2gid = {}
        next_gid = 1
        group_ids = np.zeros_like(labels, dtype=int)
        for idx, lbl in enumerate(labels):
            if lbl == -1:
                group_ids[idx] = 0
            else:
                if lbl not in lbl2gid:
                    lbl2gid[lbl] = next_gid
                    next_gid += 1
                group_ids[idx] = lbl2gid[lbl]

        # Append full 11‐column lines to group_output_file
        with open(group_output_file, "a") as grp_f:
            for (center, s_val, box_vals), gid in zip(
                zip(valid_centers, valid_scores, filtered_boxes),
                group_ids
            ):
                x, y, z = center
                l, w, h, heading = box_vals[3], box_vals[4], box_vals[5], box_vals[6]
                grp_f.write(
                    f"{frame_id},-1,"
                    f"{x:.4f},{y:.4f},{z:.4f},"
                    f"{l:.4f},{w:.4f},{h:.4f},"
                    f"{heading:.4f},{s_val:.4f},{gid}\n"
                )

        print(f"[Detector:{model_name.upper()}] Frame {frame_id} groups: {len(set(group_ids) - {0})}")

    print(f"[Detector:{model_name.upper()}] Finished raw detections → {det_output_file}")
    print(f"[Detector:{model_name.upper()}] Finished grouping → {group_output_file}")

# Ground Truth‐Based Re‐Grouping
def apply_gt_refinement(
    gt_path: str,
    det_path: str,
    group_path: str,
    distance_threshold: float,
):
    if not os.path.exists(det_path) or os.path.getsize(det_path) == 0:
        print(" No det.txt or empty → skipping refinement.")
        return

    # Load GT counts
    print(" Loading GT counts...")
    gt_counts = defaultdict(int)
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 9:
                continue
            try:
                l, w, h = float(parts[5]), float(parts[6]), float(parts[7])
                if l > 0 and w > 0 and h > 0:
                    fid = int(parts[0])
                    gt_counts[fid] += 1
            except ValueError:
                continue

    # Read det.txt into a DataFrame
    print(" Loading raw detections...")
    cols = ["frame", "id", "x", "y", "z", "l", "w", "h", "heading", "score"]
    det_df = pd.read_csv(det_path, header=None, names=cols)

    # Filter top‐N by GT counts
    print(" Filtering top-N by GT counts...")
    selected = []
    for fid, grp in det_df.groupby("frame"):
        n_gt = gt_counts.get(fid, 0)
        if n_gt <= 0:
            continue
        topk = grp.sort_values("score", ascending=False).head(n_gt)
        selected.append(topk)
    filtered_df = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=cols)

    # Run DBSCAN on those filtered predictions to assign “group” column
    filtered_df["group"] = 0
    print("[GT-Refine] Running DBSCAN on filtered predictions...")
    for fid, grp in filtered_df.groupby("frame"):
        if len(grp) < 2:
            continue
        coords = grp[["x", "y", "z"]].values
        clustering = DBSCAN(eps=distance_threshold, min_samples=2).fit(coords)
        labels = clustering.labels_
        lbl2gid = {}
        next_gid = 1
        gids = np.zeros_like(labels, dtype=int)
        for i, lbl in enumerate(labels):
            if lbl == -1:
                gids[i] = 0
            else:
                if lbl not in lbl2gid:
                    lbl2gid[lbl] = next_gid
                    next_gid += 1
                gids[i] = lbl2gid[lbl]
        filtered_df.loc[grp.index, "group"] = gids

    # Assign incremental IDs where group == 0
    print("[GT-Refine] Assigning IDs to lone detections (group=0)...")
    filtered_df["group"] = filtered_df["group"].astype(int)
    filtered_df["new_group"] = filtered_df["group"]
    for fid, grp in filtered_df.groupby("frame"):
        max_gid = grp[grp["group"] > 0]["group"].max()
        max_gid = int(max_gid) if not pd.isna(max_gid) else 0
        start_gid = max_gid + 1
        for idx in grp[grp["group"] == 0].index:
            filtered_df.at[idx, "new_group"] = start_gid
            start_gid += 1
    filtered_df["group"] = filtered_df["new_group"]
    filtered_df.drop(columns=["new_group"], inplace=True)

    # Overwrite group_output_file with final lines
    print(f"Writing final groups to {group_path} ...")
    filtered_df[
        ["frame", "id", "x", "y", "z", "l", "w", "h", "heading", "score", "group"]
    ].to_csv(group_path, header=False, index=False)
    print("[GT-Refine] Done.")


# MAIN: Execute All Steps in Order
def main():
    args = parse_args()

    # If a ZIP was provided, unzip before anything else
    if args.zip_path:
        print(f"[Setup] Extracting {args.zip_path} → {args.data_dir} ...")
        os.makedirs(args.data_dir, exist_ok=True)
        shutil.unpack_archive(args.zip_path, args.data_dir)
        print("[Setup] Extraction complete.")
        actual_pcd_dir = get_single_subdir(args.data_dir)
    else:
        actual_pcd_dir = args.data_dir

    # Fixed GT.txt path
    gt_txt = "/content/GT.txt"
    det_txt = args.det_output_file
    group_txt = args.group_output_file
    temp_dir = args.output_dir

    # STEP B: Generate GT.txt
    generate_gt(args.json_path, gt_txt)

    # detection + grouping
    if args.ckpt_path is None:
        raise ValueError("Please supply --ckpt_path to the checkpoint for your chosen model.")
    run_realtime_grouping(
        model_name=args.model,
        ckpt_path=args.ckpt_path,
        data_dir=actual_pcd_dir,
        output_dir=temp_dir,
        det_output_file=det_txt,
        group_output_file=group_txt,
        distance_threshold=args.distance_threshold,
    )

    # GT‐based refinement
    apply_gt_refinement(
        gt_path=gt_txt,
        det_path=det_txt,
        group_path=group_txt,
        distance_threshold=args.distance_threshold,
    )

    print("[All Done] group_detection.txt is generated.")

if __name__ == "__main__":
    main()
