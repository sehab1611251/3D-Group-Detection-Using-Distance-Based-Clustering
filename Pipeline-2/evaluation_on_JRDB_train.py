import math
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


# Utility: Euclidean distance between two 3D points
def euclidean_distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


# 3D Pedestrian Centroid Matching
def pedestrian_groups(file_path, is_gt=True):
    groups_by_frame = defaultdict(lambda: defaultdict(list))

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if is_gt:
                if len(parts) != 9:
                    continue
                frame = int(parts[0])
                group_id = int(parts[1])
                x, y, z = map(float, parts[2:5])
            else:
                if len(parts) != 11:
                    continue
                frame = int(float(parts[0]))
                group_id = int(float(parts[10]))
                x, y, z = map(float, parts[2:5])

            groups_by_frame[frame][group_id].append((x, y, z))

    return groups_by_frame


def compute_centroids(groups_by_frame):
    centroids_by_frame = defaultdict(dict)
    for frame, groups in groups_by_frame.items():
        for group_id, members in groups.items():
            x_avg = sum(p[0] for p in members) / len(members)
            y_avg = sum(p[1] for p in members) / len(members)
            z_avg = sum(p[2] for p in members) / len(members)
            centroids_by_frame[frame][group_id] = (x_avg, y_avg, z_avg)
    return centroids_by_frame


def evaluate_simple(gt_centroids, pred_centroids, threshold):
    TP = FP = FN = 0

    for frame, gt_groups in gt_centroids.items():
        if frame not in pred_centroids:
            FN += len(gt_groups)
            continue

        pred_groups = pred_centroids[frame]
        matched_pred = set()

        for gt_id, gt_center in gt_groups.items():
            best_match = None
            best_dist = float("inf")
            for pred_id, pred_center in pred_groups.items():
                if pred_id in matched_pred:
                    continue
                dist = euclidean_distance(gt_center, pred_center)
                if dist < best_dist:
                    best_dist = dist
                    best_match = pred_id

            if best_match is not None and best_dist <= threshold:
                TP += 1
                matched_pred.add(best_match)
            else:
                FN += 1

        FP += len(set(pred_groups.keys()) - matched_pred)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"\n=== 3D Pedestrian centroid-based evaluation (Threshold = {threshold} m) ===")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


# 3D Bounding‐Box Centroid Matching
def bounding_box_groups(file_path, is_gt=True):
    groups_by_frame = defaultdict(lambda: defaultdict(list))
    if is_gt:
        # GT: store each detection’s box-center directly
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 9:
                    continue
                frame   = int(parts[0])
                group_id= int(parts[1])
                x, y, z = map(float, parts[2:5])
                l, w, h = map(float, parts[5:8])

                min_corner = np.array([x - l/2, y - w/2, z - h/2])
                max_corner = np.array([x + l/2, y + w/2, z + h/2])
                center     = (min_corner + max_corner) / 2

                # Store the detection‐center
                groups_by_frame[frame][group_id].append(center)

    else:
        # Prediction: collect (min_corner, max_corner) for each detection, per group
        df = pd.read_csv(
            file_path, header=None,
            names=['frame', 'id', 'x', 'y', 'z', 'l', 'w', 'h', 'heading', 'score', 'group']
        )
        for frame, frame_df in df.groupby('frame'):
            for group_id, group_df in frame_df.groupby('group'):
                mins = []
                maxes = []
                for _, row in group_df.iterrows():
                    x, y, z = row['x'], row['y'], row['z']
                    l, w, h = row['l'], row['w'], row['h']
                    min_corner = np.array([x - l/2, y - w/2, z - h/2])
                    max_corner = np.array([x + l/2, y + w/2, z + h/2])
                    mins.append(min_corner)
                    maxes.append(max_corner)
                if mins and maxes:
                    # Collapse all min‐corners into one overall min, and same for max
                    overall_min = np.min(np.stack(mins, axis=0), axis=0)
                    overall_max = np.max(np.stack(maxes, axis=0), axis=0)
                    # Store that single pair
                    groups_by_frame[int(frame)][int(group_id)].append((overall_min, overall_max))

    return groups_by_frame


def compute_centroids_box(groups_by_frame):
    centroids_by_frame = defaultdict(dict)
    for frame, groups in groups_by_frame.items():
        for group_id, corner_pairs in groups.items():
            # corner_pairs is a list of (min_corner, max_corner) tuples
            all_mins = np.stack([pair[0] for pair in corner_pairs], axis=0)
            all_maxs = np.stack([pair[1] for pair in corner_pairs], axis=0)
            merged_min = np.min(all_mins, axis=0)
            merged_max = np.max(all_maxs, axis=0)
            center = ((merged_min + merged_max) / 2.0).tolist()
            centroids_by_frame[frame][group_id] = tuple(center)
    return centroids_by_frame


def evaluate_box(gt_centroids, pred_centroids, threshold):
    TP = FP = FN = 0

    for frame, gt_groups in gt_centroids.items():
        if frame not in pred_centroids:
            FN += len(gt_groups)
            continue

        pred_groups = pred_centroids[frame]
        matched_pred = set()

        for gt_id, gt_center in gt_groups.items():
            best_match = None
            best_dist = float("inf")
            for pred_id, pred_center in pred_groups.items():
                if pred_id in matched_pred:
                    continue
                dist = euclidean_distance(gt_center, pred_center)
                if dist < best_dist:
                    best_dist = dist
                    best_match = pred_id

            if best_match is not None and best_dist <= threshold:
                TP += 1
                matched_pred.add(best_match)
            else:
                FN += 1

        FP += len(set(pred_groups.keys()) - matched_pred)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"\n=== 3D bounding‐box centroid-based evaluation (Threshold = {threshold} m) ===")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


# Hungarian Matching on 3D BBox Centroids
def evaluate_hungarian(gt_centroids, pred_centroids, threshold):
    TP = FP = FN = 0

    for frame, gt_groups in gt_centroids.items():
        if frame not in pred_centroids:
            FN += len(gt_groups)
            continue

        pred_groups = pred_centroids[frame]
        gt_ids = list(gt_groups.keys())
        pred_ids = list(pred_groups.keys())

        gt_points = np.array([gt_groups[i] for i in gt_ids])
        pred_points = np.array([pred_groups[i] for i in pred_ids])

        if gt_points.size == 0:
            FP += len(pred_ids)
            continue
        if pred_points.size == 0:
            FN += len(gt_ids)
            continue

        # Compute cost matrix: Euclidean distances
        cost_matrix = cdist(gt_points, pred_points)

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_gt = set()
        matched_pred = set()

        for r, c in zip(row_ind, col_ind):
            dist = cost_matrix[r, c]
            if dist <= threshold:
                TP += 1
            else:
                FP += 1
            matched_gt.add(gt_ids[r])
            matched_pred.add(pred_ids[c])

        FN += len(gt_ids)   - len(matched_gt)
        FP += len(pred_ids) - len(matched_pred)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"\n=== Hungarian matching on 3D bounding‐box centroids (Threshold = {threshold} m) ===")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate group detection on JRDB train using three matching strategies"
    )
    parser.add_argument(
        "--gt_file", required=True,
        help="Path to ground‐truth file (GT.txt)"
    )
    parser.add_argument(
        "--pred_file", required=True,
        help="Path to predicted group file (group_detection.txt)"
    )
    parser.add_argument(
        "--threshold", type=float, default=2.0,
        help="Distance threshold in meters"
    )
    args = parser.parse_args()

    gt_fp = args.gt_file
    pred_fp = args.pred_file
    thresh = args.threshold

    # 3D pedestrian centroid matching
    gt_pedestrian   = pedestrian_groups(gt_fp, is_gt=True)
    pred_pedestrian = pedestrian_groups(pred_fp, is_gt=False)
    gt_ped_centroids = compute_centroids(gt_pedestrian)
    pred_ped_centroids = compute_centroids(pred_pedestrian)
    evaluate_simple(gt_ped_centroids, pred_ped_centroids, thresh)

    # 3D bounding-box centroid matching
    gt_box   = bounding_box_groups(gt_fp, is_gt=True)
    pred_box = bounding_box_groups(pred_fp, is_gt=False)
    gt_box_centroids   = compute_centroids(gt_box)
    pred_box_centroids = compute_centroids_box(pred_box)
    evaluate_box(gt_box_centroids, pred_box_centroids, thresh)

    # Hungarian matching on 3D bounding-box centroids
    evaluate_hungarian(gt_box_centroids, pred_box_centroids, thresh)


if __name__ == "__main__":
    main()


