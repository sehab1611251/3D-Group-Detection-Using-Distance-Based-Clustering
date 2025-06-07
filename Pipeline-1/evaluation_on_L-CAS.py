import os
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cdist
from scipy.optimize import linear_sum_assignment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate group detection with three methods in one run"
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Folder containing GT label .txt files",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="CSV file of predicted group detections (no header)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Distance threshold (meters) for matching",
    )
    return parser.parse_args()


# 3D Pedestrian Centroid Matching
def pred_pedestrian_groups(filepath):
    df = pd.read_csv(filepath, header=None)
    df.columns = [
        "frame", "id", "x_center", "y_center", "z_center",
        "length", "width", "height", "heading", "score", "group"
    ]
    pred = {}
    for frame, frame_df in df.groupby("frame"):
        centroids = []
        for gid, grp in frame_df.groupby("group"):
            coords = grp[["x_center", "y_center", "z_center"]].values
            if coords.size:
                centroids.append(coords.mean(axis=0))
        if centroids:
            pred[int(frame)] = centroids
    return pred


def gt_pedestrian_groups(gt_dir):
    gt = {}
    for fname in sorted(os.listdir(gt_dir)):
        if not fname.endswith(".txt"):
            continue
        frame_id = int(fname.replace(".txt", ""))
        centroids = []
        with open(os.path.join(gt_dir, fname), "r", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] == "group":
                    # Original GT format: "group cx cy cz xmin ymin zmin xmax ymax zmax visibility"
                    # parts[1:4] are (cx, cy, cz)
                    x, y, z = map(float, parts[1:4])
                    centroids.append(np.array([x, y, z]))
        if centroids:
            gt[frame_id] = centroids
    return gt


def evaluate_pedestrian_centroid(pred, gt, thresh):
    TP = FP = FN = 0
    for frame, gt_cents in gt.items():
        pred_cents = pred.get(frame, [])
        matched = set()
        frame_tp = 0
        for g in gt_cents:
            best_dist = float("inf")
            best_idx = -1
            for i, p in enumerate(pred_cents):
                if i in matched:
                    continue
                d = euclidean(p, g)
                if d <= thresh and d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx != -1:
                matched.add(best_idx)
                frame_tp += 1
        TP += frame_tp
        FN += len(gt_cents) - frame_tp
        FP += len(pred_cents) - len(matched)
    prec = TP / (TP + FP) if TP + FP > 0 else 0.0
    rec = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"TP": TP, "FP": FP, "FN": FN,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4)}


# 3D Bounding‐Box Centroid Matching
def pred_centroids_bbox(filepath):
    df = pd.read_csv(filepath, header=None)
    df.columns = [
        "frame", "id", "x_center", "y_center", "z_center",
        "length", "width", "height", "heading", "score", "group"
    ]
    pred = {}
    for frame, frame_df in df.groupby("frame"):
        cents = []
        for gid, grp in frame_df.groupby("group"):
            mins = []
            maxs = []
            for _, row in grp.iterrows():
                x, y, z = row["x_center"], row["y_center"], row["z_center"]
                l, w, h = row["length"], row["width"], row["height"]
                mins.append(np.array([x - l/2, y - w/2, z - h/2]))
                maxs.append(np.array([x + l/2, y + w/2, z + h/2]))
            if mins and maxs:
                mn = np.min(np.stack(mins, axis=0), axis=0)
                mx = np.max(np.stack(maxs, axis=0), axis=0)
                cents.append((mn + mx) / 2)
        if cents:
            pred[int(frame)] = cents
    return pred


def gt_centroids_bbox(gt_dir):
    gt = {}
    for fname in sorted(os.listdir(gt_dir)):
        if not fname.endswith(".txt"):
            continue
        frame_id = int(fname.replace(".txt", ""))
        cents = []
        with open(os.path.join(gt_dir, fname), "r", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] == "group":
                    # parts[4:7] are (xmin, ymin, zmin), parts[7:10] are (xmax, ymax, zmax)
                    xmin, ymin, zmin = map(float, parts[4:7])
                    xmax, ymax, zmax = map(float, parts[7:10])
                    center = (np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax])) / 2
                    cents.append(center)
        if cents:
            gt[frame_id] = cents
    return gt


def evaluate_greedy_bbox(pred, gt, thresh):
    return evaluate_pedestrian_centroid(pred, gt, thresh)


# Hungarian Matching on 3D BBox Centroids
def evaluate_hungarian(pred, gt, thresh):
    TP = FP = FN = 0
    for frame, gt_cents in gt.items():
        pred_cents = pred.get(frame, [])
        if not gt_cents and not pred_cents:
            continue
        if not pred_cents:
            FN += len(gt_cents)
            continue

        cost = cdist(np.stack(gt_cents), np.stack(pred_cents))
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_pred = set()
        matched_gt = set()

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] <= thresh:
                TP += 1
                matched_gt.add(i)
                matched_pred.add(j)
            else:
                FP += 1
                matched_gt.add(i)
                matched_pred.add(j)

        FN += len(gt_cents) - len(matched_gt)
        FP += len(pred_cents) - len(matched_pred)

    prec = TP / (TP + FP) if TP + FP > 0 else 0.0
    rec = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"TP": TP, "FP": FP, "FN": FN,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4)}


# Main: run all three methods
def main():
    args = parse_args()

    print("1) 3D Pedestrian centroid-based evaluation")
    pred_pedestrian = pred_pedestrian_groups(args.pred_file)
    gt_pedestrian = gt_pedestrian_groups(args.gt_dir)
    res_pedestrian = evaluate_pedestrian_centroid(pred_pedestrian, gt_pedestrian, args.threshold)
    print(f"Results: {res_pedestrian}\n")

    print("2) 3D bounding‐box centroid-based evaluation")
    pred_bbox = pred_centroids_bbox(args.pred_file)
    gt_bbox = gt_centroids_bbox(args.gt_dir)
    res_bbox = evaluate_greedy_bbox(pred_bbox, gt_bbox, args.threshold)
    print(f"Results: {res_bbox}\n")

    print("3) Hungarian matching on 3D bounding‐box centroids")
    res_hungarian = evaluate_hungarian(pred_bbox, gt_bbox, args.threshold)
    print(f"Results: {res_hungarian}\n")


if __name__ == "__main__":
    main()
