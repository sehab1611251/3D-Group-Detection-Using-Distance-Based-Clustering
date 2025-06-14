
import os
import glob
import argparse
import multiprocessing as mp

import numpy as np
import open3d as o3d
import pyvista as pv
from pyvista import Plotter
from pyvirtualdisplay import Display

import cv2 

# Helper: OpenCV VideoWriter Utility
def open_video_writer(video_path, fps, width, height):
    """
    Create and return a cv2.VideoWriter for MP4 output.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {video_path}")
    return writer

# Ground Truth Individual Pedestrian Visualization (yellow boxes) + video

def load_gt_detections(gt_file):
    """
    Load ground truth individual‐pedestrian detections.
    Each line: <frame>,<group_id>,<x>,<y>,<z>,<length>,<width>,<height>,<group_label>
    """
    detections = {}
    try:
        with open(gt_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 9:
                    continue
                frame = int(parts[0])
                cx, cy, cz = map(float, parts[2:5])
                length, width, height = map(float, parts[5:8])
                group_label = int(parts[8])  # Unused for color
                det = {
                    "center": (cx, cy, cz),
                    "size": (length, width, height),
                    "heading": 0.0,
                    "group_label": group_label
                }
                detections.setdefault(frame, []).append(det)
    except Exception as e:
        print(f"[GT] Error loading {gt_file}: {e}")
        return {}
    return detections

def create_bbox_lines_gt(det):
    cx, cy, cz = det["center"]
    l, w, h = det["size"]
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
    corners = np.array([
        [ dx,  dy,  dz], [ dx, -dy,  dz], [-dx, -dy,  dz], [-dx,  dy,  dz],
        [ dx,  dy, -dz], [ dx, -dy, -dz], [-dx, -dy, -dz], [-dx,  dy, -dz]
    ])
    R = np.eye(3)
    rotated = (R @ corners.T).T + np.array([cx, cy, cz])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    return np.array([[rotated[i], rotated[j]] for i, j in edges])

def transform_lidar(points_lidar):
    """
    Applies a +90° rotation around Z to transform LiDAR coordinates
    into the camera‐aligned frame (JRDB convention).
    """
    R = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    return (R @ points_lidar.T).T

def render_gt(pcd_dir, gt_file, output_dir, fps):
    """
    Render ground truth individual bounding boxes (in yellow) for each frame,
    save one PNG per frame in output_dir, and append that PNG to gt_video.mp4
    """
    detections_by_frame = load_gt_detections(gt_file)
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))

    os.makedirs(output_dir, exist_ok=True)
    display = Display(visible=0, size=(800, 600))
    display.start()

    video_path = "/content/gt_video.mp4"
    writer = open_video_writer(video_path, fps, 640, 360)

    for pcd_file in pcd_files:
        base = os.path.splitext(os.path.basename(pcd_file))[0]
        try:
            frame = int(base)
        except ValueError:
            continue

        if frame not in detections_by_frame:
            continue

        out_img = os.path.join(output_dir, f"{frame:06d}.png")
        print(f"[GT] Rendering frame {frame} -> {out_img}")

        try:
            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points)
            points = transform_lidar(points)
        except Exception as e:
            print(f"[GT] Error loading {pcd_file}: {e}")
            continue

        cloud = pv.PolyData(points)
        cloud["z_val"] = points[:, 2]

        plotter = Plotter(off_screen=True, window_size=(640, 360))
        plotter.background_color = "#eeeeee"

        plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                         i_size=30, j_size=30)
        plotter.add_mesh(plane, color="gray", opacity=0.15)

        plotter.add_mesh(cloud, scalars="z_val", cmap="viridis",
                         point_size=4, render_points_as_spheres=True,
                         show_scalar_bar=False)

        yellow_hex = "#FFFF00"
        rgb = tuple(int(yellow_hex.lstrip("#")[i:i+2], 16) / 255.0
                    for i in (0, 2, 4))

        for det in detections_by_frame[frame]:
            for line in create_bbox_lines_gt(det):
                p1 = transform_lidar(np.array([line[0]]))[0]
                p2 = transform_lidar(np.array([line[1]]))[0]
                mesh = pv.Line(p1, p2)
                plotter.add_mesh(mesh, color=rgb, line_width=3)

        plotter.camera_position = [(20, 10, 12), (0, 0, 0), (0, 0, 1)]
        plotter.show(screenshot=out_img, auto_close=True)

        frame_bgr = cv2.imread(out_img)
        if frame_bgr is not None:
            writer.write(frame_bgr)

    writer.release()
    display.stop()
    print("[GT] Video saved at:", video_path)

# Ground Truth Group Visualization + video
def load_gt_group_detections(gt_file):
    detections = {}
    try:
        with open(gt_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 9:
                    continue
                frame = int(parts[0])
                cx, cy, cz = map(float, parts[2:5])
                length, width, height = map(float, parts[5:8])
                group_label = int(parts[8])
                det = {
                    "center": (cx, cy, cz),
                    "size": (length, width, height),
                    "heading": 0.0,
                    "group_label": group_label
                }
                detections.setdefault(frame, []).append(det)
    except Exception as e:
        print(f"[GT Group] Error loading {gt_file}: {e}")
        return {}
    return detections

def create_bbox_lines_gtgroup(det):
    cx, cy, cz = det["center"]
    l, w, h = det["size"]
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
    corners = np.array([
        [ dx,  dy,  dz], [ dx, -dy,  dz], [-dx, -dy,  dz], [-dx,  dy,  dz],
        [ dx,  dy, -dz], [ dx, -dy, -dz], [-dx, -dy, -dz], [-dx,  dy, -dz]
    ])
    R = np.eye(3)
    rotated = (R @ corners.T).T + np.array([cx, cy, cz])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    return np.array([[rotated[e[0]], rotated[e[1]]] for e in edges])

def render_gt_group(pcd_dir, gt_file, output_dir, fps):
    detections_by_frame = load_gt_group_detections(gt_file)
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))

    os.makedirs(output_dir, exist_ok=True)
    display = Display(visible=0, size=(800, 600))
    display.start()

    video_path = "/content/gt_group_video.mp4"
    writer = open_video_writer(video_path, fps, 640, 360)

    for pcd_file in pcd_files:
        base = os.path.splitext(os.path.basename(pcd_file))[0]
        try:
            frame = int(base)
        except ValueError:
            continue

        if frame not in detections_by_frame:
            continue

        out_img = os.path.join(output_dir, f"{frame:06d}.png")
        print(f"[GT Group] Rendering frame {frame} -> {out_img}")

        try:
            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points)
            points = transform_lidar(points)
        except Exception as e:
            print(f"[GT Group] Error loading {pcd_file}: {e}")
            continue

        cloud = pv.PolyData(points)
        cloud["z_val"] = points[:, 2]
        plotter = Plotter(off_screen=True, window_size=(640, 360))
        plotter.background_color = "#eeeeee"

        plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=30, j_size=30)
        plotter.add_mesh(plane, color="gray", opacity=0.15)
        plotter.add_mesh(cloud, scalars="z_val", cmap="viridis",
                         point_size=4, render_points_as_spheres=True,
                         show_scalar_bar=False)

        palette = [
            "#FF0000", "#0000FF", "#008000", "#FFA500", "#800080",
            "#00FFFF", "#FFC0CB", "#A52A2A", "#808000", "#DC143C",
            "#FF4500", "#32CD32", "#B22222", "#20B2AA", "#FF69B4",
            "#4B0082", "#9ACD32", "#D2691E", "#ADFF2F", "#4682B4",
            "#DAA520", "#008B8B", "#800000", "#808080", "#FF1493",
            "#00FA9A", "#FF8C00", "#00BFFF"
        ]
        predefined = {i + 1: palette[i] for i in range(5)}
        used = set(predefined.values())
        dynamic_idx = 5

        for det in detections_by_frame[frame]:
            lbl = det["group_label"]
            if lbl in predefined:
                color_hex = predefined[lbl]
            else:
                while dynamic_idx < len(palette) and palette[dynamic_idx] in used:
                    dynamic_idx += 1
                if dynamic_idx < len(palette):
                    color_hex = palette[dynamic_idx]
                    used.add(color_hex)
                else:
                    color_hex = "#000000"
                dynamic_idx += 1

            rgb = tuple(int(color_hex.lstrip("#")[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            for line in create_bbox_lines_gtgroup(det):
                p1 = transform_lidar(np.array([line[0]]))[0]
                p2 = transform_lidar(np.array([line[1]]))[0]
                mesh = pv.Line(p1, p2)
                plotter.add_mesh(mesh, color=rgb, line_width=3)

        plotter.camera_position = [(20, 10, 12), (0, 0, 0), (0, 0, 1)]
        plotter.show(screenshot=out_img, auto_close=True)

        frame_bgr = cv2.imread(out_img)
        if frame_bgr is not None:
            writer.write(frame_bgr)

    writer.release()
    display.stop()
    print("[GT Group] Video saved at:", video_path)


# DCCLA / RPEA Model Output Visualization + video
def load_dccla_detections(detection_file):
    """
    Load DCCLA / RPEA model detections.
    Each line: <frame>,<id>,<x>,<y>,<z>,<length>,<width>,<height>,<heading>,<score>
    """
    detections = {}
    try:
        with open(detection_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 10:
                    continue
                frame = int(parts[0])
                det_id = int(parts[1])
                cx, cy, cz = map(float, parts[2:5])
                length, width, height = map(float, parts[5:8])
                heading = float(parts[8])
                score = float(parts[9])
                det = {
                    "id": det_id,
                    "center": (cx, cy, cz),
                    "size": (length, width, height),
                    "heading": heading,
                    "score": score
                }
                detections.setdefault(frame, []).append(det)
    except Exception as e:
        print(f"[DCCLA / RPEA] Error loading {detection_file}: {e}")
        return {}
    return detections

def create_bbox_lines_dccla(det):
    cx, cy, cz = det["center"]
    l, w, h = det["size"]
    yaw = det["heading"]
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
    corners = np.array([
        [ dx,  dy,  dz], [ dx, -dy,  dz], [-dx, -dy,  dz], [-dx,  dy,  dz],
        [ dx,  dy, -dz], [ dx, -dy, -dz], [-dx, -dy, -dz], [-dx,  dy, -dz]
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    rotated = (R @ corners.T).T + np.array([cx, cy, cz])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    lines = np.array([[rotated[i], rotated[j]] for i, j in edges])
    return lines, rotated

def render_dccla(pcd_dir, detection_file, output_dir, fps):
    """
    Render DCCLA / RPEA prediction bounding boxes in yellow with front‐face crosses,
    """
    detections_by_frame = load_dccla_detections(detection_file)
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))

    os.makedirs(output_dir, exist_ok=True)
    display = Display(visible=0, size=(800, 600))
    display.start()

    video_path = "/content/dccla_video.mp4"
    writer = open_video_writer(video_path, fps, 640, 360)

    for pcd_file in pcd_files:
        base = os.path.splitext(os.path.basename(pcd_file))[0]
        try:
            frame = int(base)
        except ValueError:
            continue

        if frame not in detections_by_frame:
            continue

        out_img = os.path.join(output_dir, f"{frame:06d}.png")
        print(f"[DCCLA / RPEA] Rendering frame {frame} -> {out_img}")

        try:
            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points)
            points = transform_lidar(points)
        except Exception as e:
            print(f"[DCCLA / RPEA] Error loading {pcd_file}: {e}")
            continue

        cloud = pv.PolyData(points)
        cloud["z_val"] = points[:, 2]
        plotter = Plotter(off_screen=True, window_size=(640, 360))
        plotter.background_color = "#eeeeee"

        plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=30, j_size=30)
        plotter.add_mesh(plane, color="gray", opacity=0.15)
        plotter.add_mesh(cloud, scalars="z_val", cmap="viridis",
                         point_size=4, render_points_as_spheres=True,
                         show_scalar_bar=False)

        rgb = (1.0, 1.0, 0.0)

        for det in detections_by_frame[frame]:
            lines, rotated = create_bbox_lines_dccla(det)

            for line in lines:
                p1 = transform_lidar(np.array([line[0]]))[0]
                p2 = transform_lidar(np.array([line[1]]))[0]
                mesh = pv.Line(p1, p2)
                plotter.add_mesh(mesh, color=rgb, line_width=3)

            front = rotated[[0, 1, 5, 4]]
            front = transform_lidar(front)
            cross1 = pv.Line(front[0], front[2])
            cross2 = pv.Line(front[1], front[3])
            plotter.add_mesh(cross1, color=rgb, line_width=3)
            plotter.add_mesh(cross2, color=rgb, line_width=3)

        plotter.camera_position = [(20, 10, 12), (0, 0, 0), (0, 0, 1)]
        plotter.show(screenshot=out_img, auto_close=True)

        frame_bgr = cv2.imread(out_img)
        if frame_bgr is not None:
            writer.write(frame_bgr)

    writer.release()
    display.stop()
    print("[DCCLA / RPEA] Video saved at:", video_path)


# Predicted Group Visualization + video
def load_pred_group_detections(pred_file):
    """
    Load predicted group bounding boxes.
    Each line: <frame>,<id>,<x>,<y>,<z>,<length>,<width>,<height>,<heading>,<score>,<group>
    """
    detections = {}
    try:
        with open(pred_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 11:
                    continue
                frame = int(parts[0])
                det_id = int(parts[1])
                cx, cy, cz = map(float, parts[2:5])
                length, width, height = map(float, parts[5:8])
                heading = float(parts[8])
                score = float(parts[9])
                group_id = int(parts[10])
                det = {
                    "id": det_id,
                    "center": (cx, cy, cz),
                    "size": (length, width, height),
                    "heading": heading,
                    "score": score,
                    "group_id": group_id
                }
                detections.setdefault(frame, []).append(det)
    except Exception as e:
        print(f"[Pred Group] Error loading {pred_file}: {e}")
        return {}
    return detections

def create_bbox_lines_pred(det):
    cx, cy, cz = det["center"]
    l, w, h = det["size"]
    yaw = det["heading"]
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
    corners = np.array([
        [ dx,  dy,  dz], [ dx, -dy,  dz], [-dx, -dy,  dz], [-dx,  dy,  dz],
        [ dx,  dy, -dz], [ dx, -dy, -dz], [-dx, -dy, -dz], [-dx,  dy, -dz]
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    rotated = (R @ corners.T).T + np.array([cx, cy, cz])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    lines = np.array([[rotated[i], rotated[j]] for i, j in edges])
    return lines, rotated

def render_pred_group(pcd_dir, pred_file, output_dir, fps):
    detections_by_frame = load_pred_group_detections(pred_file)
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))

    os.makedirs(output_dir, exist_ok=True)
    display = Display(visible=0, size=(800, 600))
    display.start()

    video_path = "/content/pred_group_video.mp4"
    writer = open_video_writer(video_path, fps, 640, 360)

    for pcd_file in pcd_files:
        base = os.path.splitext(os.path.basename(pcd_file))[0]
        try:
            frame = int(base)
        except ValueError:
            continue

        if frame not in detections_by_frame:
            continue

        out_img = os.path.join(output_dir, f"{frame:06d}.png")
        print(f"[Pred Group] Rendering frame {frame} -> {out_img}")

        try:
            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points)
            points = transform_lidar(points)
        except Exception as e:
            print(f"[Pred Group] Error loading {pcd_file}: {e}")
            continue

        cloud = pv.PolyData(points)
        cloud["z_val"] = points[:, 2]
        plotter = Plotter(off_screen=True, window_size=(640, 360))
        plotter.background_color = "#eeeeee"

        plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=30, j_size=30)
        plotter.add_mesh(plane, color="gray", opacity=0.15)
        plotter.add_mesh(cloud, scalars="z_val", cmap="viridis",
                         point_size=4, render_points_as_spheres=True,
                         show_scalar_bar=False)

        palette = [
            "#FF0000", "#0000FF", "#008000", "#FFA500", "#800080",
            "#00FFFF", "#FFC0CB", "#A52A2A", "#808000", "#DC143C",
            "#FF4500", "#32CD32", "#B22222", "#20B2AA", "#FF69B4",
            "#4B0082", "#9ACD32", "#D2691E", "#ADFF2F", "#4682B4",
            "#DAA520", "#008B8B", "#800000", "#808080", "#FF1493",
            "#00FA9A", "#FF8C00", "#00BFFF"
        ]
        predefined = {i + 1: palette[i] for i in range(5)}
        used = set(predefined.values())
        dynamic_idx = 5

        for det in detections_by_frame[frame]:
            grp = det["group_id"]
            if grp in predefined:
                color_hex = predefined[grp]
            else:
                while dynamic_idx < len(palette) and palette[dynamic_idx] in used:
                    dynamic_idx += 1
                if dynamic_idx < len(palette):
                    color_hex = palette[dynamic_idx]
                    used.add(color_hex)
                else:
                    color_hex = "#000000"
                dynamic_idx += 1

            rgb = tuple(int(color_hex.lstrip("#")[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            lines, rotated = create_bbox_lines_pred(det)

            for line in lines:
                p1 = transform_lidar(np.array([line[0]]))[0]
                p2 = transform_lidar(np.array([line[1]]))[0]
                mesh = pv.Line(p1, p2)
                plotter.add_mesh(mesh, color=rgb, line_width=3)

            front = rotated[[0, 1, 5, 4]]
            front = transform_lidar(front)
            cross1 = pv.Line(front[0], front[2])
            cross2 = pv.Line(front[1], front[3])
            plotter.add_mesh(cross1, color=rgb, line_width=3)
            plotter.add_mesh(cross2, color=rgb, line_width=3)

        plotter.camera_position = [(20, 10, 12), (0, 0, 0), (0, 0, 1)]
        plotter.show(screenshot=out_img, auto_close=True)

        frame_bgr = cv2.imread(out_img)
        if frame_bgr is not None:
            writer.write(frame_bgr)

    writer.release()
    display.stop()
    print("[Pred Group] Video saved at:", video_path)


# Main: Run all four pipelines in parallel
def parse_args():
    parser = argparse.ArgumentParser(
        description="Real‐time visualization of 4 pipelines + simultaneous video creation"
    )
    parser.add_argument("--gt_file", required=True,
                        help="Path to GT.txt (used for both individual and group‐GT)")
    parser.add_argument("--detection_file", required=True,
                        help="Path to DCCLA/RPEA detection file (det.txt)")
    parser.add_argument("--pred_file", required=True,
                        help="Path to predicted group file (group_detection.txt)")
    parser.add_argument("--pcd_dir", required=True,
                        help="Directory containing .pcd files")
    parser.add_argument("--out_gt", required=True,
                        help="Output dir for GT individual frames")
    parser.add_argument("--out_gt_group", required=True,
                        help="Output dir for GT group frames")
    parser.add_argument("--out_detector", required=True,
                        help="Output dir for DCCLA frames")
    parser.add_argument("--out_pred", required=True,
                        help="Output dir for predicted group frames")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frames per second for the videos")
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.out_gt, exist_ok=True)
    os.makedirs(args.out_gt_group, exist_ok=True)
    os.makedirs(args.out_detector, exist_ok=True)
    os.makedirs(args.out_pred, exist_ok=True)

    procs = [
        mp.Process(target=render_gt, args=(
            args.pcd_dir, args.gt_file, args.out_gt, args.fps)),
        mp.Process(target=render_gt_group, args=(
            args.pcd_dir, args.gt_file, args.out_gt_group, args.fps)),
        mp.Process(target=render_dccla, args=(
            args.pcd_dir, args.detection_file, args.out_detector, args.fps)),
        mp.Process(target=render_pred_group, args=(
            args.pcd_dir, args.pred_file, args.out_pred, args.fps)),
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()

if __name__ == "__main__":
    main()
