import os
import sys
import glob
import shutil
import zipfile
import argparse
import yaml

import torch
import numpy as np
import open3d as o3d
import pandas as pd

from sklearn.cluster import DBSCAN


# Utility converters
def to_numpy(x):
    return x.cpu().numpy() if torch.is_tensor(x) else np.array(x)

def to_float(x):
    return x.cpu().item() if torch.is_tensor(x) else float(x)

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Full pipeline: unzip → preprocess → detect (RPEA|DCCLA) + grouping → GT‐filter"
    )
    parser.add_argument(
        "--model",
        choices=["rpea", "dccla"],
        required=True,
        help="Which detector to use: 'rpea' or 'dccla'"
    )
    parser.add_argument(
        "--pcd_zip",
        type=str,
        required=True,
        help="Path to ZIP archive of .pcd files"
    )
    parser.add_argument(
        "--labels_zip",
        type=str,
        required=True,
        help="Path to ZIP archive of GT label .txt files"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to RPEA or DCCLA checkpoint (required)"
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=1.5,
        help="DBSCAN ε for grouping and centroid matching (meters)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./temp_output",
        help="Base directory for unzipping and intermediate outputs"
    )
    parser.add_argument(
        "--det_output_file",
        type=str,
        default="./det.txt",
        help="Aggregated raw detections output file"
    )
    parser.add_argument(
        "--group_output_file",
        type=str,
        default="./group_detections.txt",
        help="Group detections output file (overwritten after filtering)"
    )
    return parser.parse_args()


# Unzip ZIP Archives
def unzip_to_dir(zip_path: str, extract_dir: str):
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"[Unzip] Extracted {zip_path} → {extract_dir}")

def get_single_subdir(root_dir: str) -> str:
    entries = os.listdir(root_dir)
    subdirs = [d for d in entries
               if os.path.isdir(os.path.join(root_dir, d)) and d != "__MACOSX"]
    if len(subdirs) == 1:
        return os.path.join(root_dir, subdirs[0])
    return root_dir

# Preprocess Labels and PCDs
def preprocess_labels_and_pcds(labels_folder: str, pcd_folder: str):
    """
    1. Remove any label .txt with no 'group' lines, and its matching .pcd.
    2. Remove any .pcd without a matching .txt.
    3. Rename matched (label, .pcd) pairs to sequential 6-digit basenames.
    """
    kept_count = 0
    deleted_count = 0

    for label_file in os.listdir(labels_folder):
        if not label_file.endswith(".txt"):
            continue
        label_path = os.path.join(labels_folder, label_file)
        base = os.path.splitext(label_file)[0]
        pcd_path = os.path.join(pcd_folder, base + ".pcd")

        group_count = 0
        with open(label_path, "r", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0].lower() == "group":
                    group_count += 1

        if group_count >= 1:
            kept_count += 1
        else:
            os.remove(label_path)
            if os.path.isfile(pcd_path):
                os.remove(pcd_path)
            print(f"[Preprocess] Deleted (no groups): {label_file} and {base}.pcd")
            deleted_count += 1

    print(f"[Preprocess] Label filtering complete. Kept: {kept_count}, Deleted: {deleted_count}")

    label_bases = {
        os.path.splitext(f)[0]
        for f in os.listdir(labels_folder)
        if f.endswith(".txt")
    }
    pcd_deleted = 0
    for pcd_file in os.listdir(pcd_folder):
        if not pcd_file.endswith(".pcd"):
            continue
        base = os.path.splitext(pcd_file)[0]
        if base not in label_bases:
            os.remove(os.path.join(pcd_folder, pcd_file))
            print(f"[Preprocess] Deleted unmatched PCD: {pcd_file}")
            pcd_deleted += 1

    print(f"[Preprocess] Unmatched PCD deletion complete. Deleted: {pcd_deleted}")

    pcd_bases = {
        os.path.splitext(f)[0]
        for f in os.listdir(pcd_folder)
        if f.endswith(".pcd")
    }
    txt_bases = {
        os.path.splitext(f)[0]
        for f in os.listdir(labels_folder)
        if f.endswith(".txt")
    }
    matched = sorted(pcd_bases & txt_bases)

    for idx, old_base in enumerate(matched, start=1):
        new_base = f"{idx:06d}"
        os.rename(os.path.join(pcd_folder, old_base + ".pcd"),
                  os.path.join(pcd_folder, new_base + ".pcd"))
        os.rename(os.path.join(labels_folder, old_base + ".txt"),
                  os.path.join(labels_folder, new_base + ".txt"))
        print(f"[Preprocess] Renamed pair: {old_base} → {new_base}")

    print(f"[Preprocess] Renamed {len(matched)} matched file pairs.")


# detection + grouping (branches on model)
def run_grouping(
    model_name: str,
    ckpt_path: str,
    pcd_folder: str,
    temp_dir: str,
    det_output_file: str,
    group_output_file: str,
    distance_threshold: float,
):
    """
    For each .pcd file in pcd_folder:
      • If model='rpea': load RPEA, inference → pc_input = pts.T.astype(np.float32)
      • If model='dccla': load DCCLA, inference → pc_input = pts.T.astype(np.float32)
      – Save per‐frame raw detections to temp_dir (8 columns: x,y,z,l,w,h,heading,score)
      – Append valid detections to det_output_file (score ≥0.30)
      – DBSCAN on valid centers → append full 11‐column lines to group_output_file
      – If fewer than 2 valid, write singletons with group=0
    """
    os.makedirs(temp_dir, exist_ok=True)
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
        # Ensure iou3d is importable before any lidar_det imports
        sys.path.append('/content/DCCLA')
        sys.path.append('/content/DCCLA/lib/iou3d')
        from lidar_det import detector as _det_module
        from lidar_det.model import get_model
        from lidar_det.detector import dccla

        # Patch get_model exactly as DCCLA expects
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
        _det_module.get_model = lambda _unused, inference_only=True: get_model(model_cfg, inference_only=inference_only)

        detector = dccla(ckpt_path, gpu=(device.type == "cuda"))
    else:
        raise ValueError(f"Incorrect model_name '{model_name}'. Must be 'rpea' or 'dccla'.")

    pcd_paths = sorted(glob.glob(os.path.join(pcd_folder, "*.pcd")))
    print(f"[Detector:{model_name.upper()}] Found {len(pcd_paths)} .pcd files in {pcd_folder}")

    for pcd_path in pcd_paths:
        frame_name = os.path.splitext(os.path.basename(pcd_path))[0]
        try:
            frame_id = int(frame_name)
        except ValueError:
            continue

        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points)  # shape: (N,3)
        if pts.shape[0] == 0:
            print(f"[Detector:{model_name.upper()}] Frame {frame_id} has no points, skipping.")
            continue

        # On L-CAS, both RPEA and DCCLA get raw pts.T → float32
        pc_input = pts.T.astype(np.float32)  # (3, N)

        with torch.no_grad():
            boxes, scores = detector(pc_input)

        if boxes is None or len(boxes) == 0:
            continue

        # Save per‐frame raw detections (x,y,z,l,w,h,heading,score)
        per_frame_file = os.path.join(temp_dir, f"{frame_name}.txt")
        with open(per_frame_file, "w") as f_out:
            for box, score in zip(boxes, scores):
                b_np = to_numpy(box)      # [x,y,z,l,w,h,heading]
                s_val = to_float(score)
                f_out.write(" ".join(map(str, b_np.tolist() + [s_val])) + "\n")
        print(f"[Detector:{model_name.upper()}] Saved per-frame detections: {per_frame_file}")

        # Append valid detections (l>0,w>0,h>0, score ≥0.30) to det_output_file
        valid_centers = []
        valid_scores = []
        filtered_boxes = []
        with open(per_frame_file, "r") as fin, open(det_output_file, "a") as det_f:
            for line in fin:
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
            if len(valid_centers) == 1:
                x, y, z, l, w, h, heading = filtered_boxes[0]
                s_val = valid_scores[0]
                with open(group_output_file, "a") as grp_f:
                    grp_f.write(
                        f"{frame_id},-1,"
                        f"{x:.4f},{y:.4f},{z:.4f},"
                        f"{l:.4f},{w:.4f},{h:.4f},"
                        f"{heading:.4f},{s_val:.4f},0\n"
                    )
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

        # Append full 11-column lines to group_output_file
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


# Filtering by Matching Predicted to GT Groups
def filter_groups_inplace(
    group_file: str,
    gt_folder: str,
    distance_threshold: float = 1.5
):
    """
    Overwrite group_file by retaining only those predicted‐group detections
    that match a GT group centroid within distance_threshold.
    """
    cols = ["frame", "id", "x", "y", "z",
            "length", "width", "height", "heading", "score", "group"]

    df = pd.read_csv(group_file, header=None, names=cols, dtype={"frame": str, "id": str, "group": str})
    for col in ["x", "y", "z", "length", "width", "height", "heading", "score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    filtered_rows = []
    frames_kept = 0
    frames_skipped = 0

    for frame_id, frame_df in df.groupby("frame"):
        frame_str = f"{int(frame_id):06d}.txt"
        gt_path = os.path.join(gt_folder, frame_str)

        if not os.path.isfile(gt_path):
            print(f"Skipped frame {frame_id} → GT file not found")
            frames_skipped += 1
            continue

        gt_group_centroids = []
        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] == "group":
                    x, y, z = map(float, parts[1:4])
                    gt_group_centroids.append(np.array([x, y, z]))

        if len(gt_group_centroids) == 0:
            print(f"Skipped frame {frame_id} → No group category in GT")
            frames_skipped += 1
            continue

        pred_groups = frame_df[frame_df["group"] != "0"]
        if pred_groups.empty:
            print(f"Skipped frame {frame_id} → No valid group predictions (group ≠ 0)")
            frames_skipped += 1
            continue

        pred_group_centroids = []
        for group_id, group_df in pred_groups.groupby("group"):
            centroid = group_df[["x", "y", "z"]].mean().values
            pred_group_centroids.append((group_id, centroid, group_df))

        used_group_ids = set()
        selected_detections = []

        for gt_centroid in gt_group_centroids:
            best_group_id = None
            best_group_df = None
            best_distance = float("inf")

            for group_id, pred_centroid, group_df in pred_group_centroids:
                if group_id in used_group_ids:
                    continue
                distance = np.linalg.norm(gt_centroid - pred_centroid)
                if distance <= distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_group_id = group_id
                    best_group_df = group_df

            if best_group_id is not None:
                used_group_ids.add(best_group_id)
                selected_detections.extend(best_group_df[cols].values.tolist())

        if selected_detections:
            print(f"Frame {frame_id} → Picked {len(used_group_ids)} matched group(s)")
            filtered_rows.extend(selected_detections)
            frames_kept += 1
        else:
            print(f"Skipped frame {frame_id} → No matched predicted group within threshold")
            frames_skipped += 1

    filtered_df = pd.DataFrame(filtered_rows, columns=cols)
    filtered_df.to_csv(group_file, index=False, header=False)

    print("\n=== Summary ===")
    print(f"Total frames processed: {frames_kept + frames_skipped}")
    print(f"Frames kept: {frames_kept}")
    print(f"Frames skipped: {frames_skipped}")
    print(f"Filtered predictions saved to: {group_file}")


# MAIN: All Steps
def main():
    args = parse_args()

    # Prepare directories
    pcd_unzip = os.path.join(args.output_dir, "pcd_unzipped")
    labels_unzip = os.path.join(args.output_dir, "labels_unzipped")
    temp_dir = os.path.join(args.output_dir, "detection_per_frame")
    os.makedirs(pcd_unzip, exist_ok=True)
    os.makedirs(labels_unzip, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Unzip
    print("===Unzip PCD & Labels===")
    unzip_to_dir(args.pcd_zip, pcd_unzip)
    unzip_to_dir(args.labels_zip, labels_unzip)

    actual_pcd_dir = get_single_subdir(pcd_unzip)
    actual_labels_dir = get_single_subdir(labels_unzip)
    print(f"[Info] Using PCD folder: {actual_pcd_dir}")
    print(f"[Info] Using Labels folder: {actual_labels_dir}")

    # Preprocess
    print("\n===Preprocess Labels & PCDs===")
    preprocess_labels_and_pcds(actual_labels_dir, actual_pcd_dir)

    # detection + grouping
    if args.ckpt_path is None:
        raise ValueError("Please supply --ckpt_path to the checkpoint for your chosen model.")
    print(f"\n=== Detector {args.model.upper()} Detection + Grouping ===")
    run_grouping(
        model_name=args.model,
        ckpt_path=args.ckpt_path,
        pcd_folder=actual_pcd_dir,
        temp_dir=temp_dir,
        det_output_file=args.det_output_file,
        group_output_file=args.group_output_file,
        distance_threshold=args.distance_threshold,
    )

    # GT-based filtering
    print("\n===GT-Based Filtering===")
    filter_groups_inplace(
        group_file=args.group_output_file,
        gt_folder=actual_labels_dir,
        distance_threshold=args.distance_threshold,
    )

    print("\n[All Done!] group_detections.txt has been updated.")

if __name__ == "__main__":
    main()

