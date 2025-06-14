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
def open_video_writer(video_path: str, fps: float, width: int, height: int):
    """
    Create and return a cv2.VideoWriter for MP4 output.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {video_path}")
    return writer


# Aggregate Ground Truth .txt Files to Single GT.txt file
def aggregate_gt(gt_folder: str, output_file: str):
    """
    Combine all per‐frame GT .txt files (named as integers) into one GT.txt.
    Each input line: <category> <cx> <cy> <cz> <xmin> <ymin> <zmin> <xmax> <ymax> <zmax> <visibility>
    We write: frame,category,cx,cy,cz,xmin,ymin,zmin,xmax,ymax,zmax,visibility
    """
    def is_int_filename(name):
        base, ext = os.path.splitext(name)
        return ext == ".txt" and base.isdigit()

    files = sorted(
        (f for f in os.listdir(gt_folder) if is_int_filename(f)),
        key=lambda fn: int(os.path.splitext(fn)[0])
    )
    if not files:
        print("No valid Ground Truth .txt files found in", gt_folder)
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as fout:
        for fn in files:
            frame = int(os.path.splitext(fn)[0])
            path = os.path.join(gt_folder, fn)
            with open(path, "r") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 11:
                        print(f"  Skipping malformed line [{fn}]:", line)
                        continue
                    category = parts[0]
                    try:
                        values = list(map(float, parts[1:10]))
                        visibility = int(parts[10])
                    except ValueError:
                        print(f"  Skipping non‐numeric line [{fn}]:", line)
                        continue
                    fout.write(
                        f"{frame},{category},"
                        + ",".join(f"{v:.6f}" for v in values)
                        + f",{visibility}\n"
                    )
    print(f"\n Aggregated GT written to: {output_file}")


# Pipeline A: Ground Truth Individual‐Pedestrian Visualization + Video
def load_gt_individuals(gt_file: str):
    detections = {}
    try:
        with open(gt_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 12:
                    continue
                frame = int(parts[0])
                cx, cy, cz = map(float, parts[2:5])
                xmin, ymin, zmin = map(float, parts[5:8])
                xmax, ymax, zmax = map(float, parts[8:11])
                length = xmax - xmin
                width  = ymax - ymin
                height = zmax - zmin
                det = {
                    "center": (cx, cy, cz),
                    "size":   (length, width, height),
                    "heading": 0.0
                }
                detections.setdefault(frame, []).append(det)
    except Exception as e:
        print(f" Error loading {gt_file}: {e}")
        return {}
    return detections


def create_bbox_lines_ind(det: dict):
    """
    Create 12 edges of an axis‐aligned cuboid for GT pedestrian.
    Returns an array of shape (12, 2, 3) for line segments.
    """
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


def transform_lidar_to_camera(points_lidar: np.ndarray):
    """
    +90° rotation around Z‐axis to transform LiDAR coordinates to camera frame.
    """
    R = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    return (R @ points_lidar.T).T


def visualize_gt_individual(pcd_dir: str, gt_file: str, output_dir: str):
    detections_by_frame = load_gt_individuals(gt_file)
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))

    os.makedirs(output_dir, exist_ok=True)
    display = Display(visible=0, size=(800, 600))
    display.start()

    video_path = "/content/gt_individuals.mp4"
    writer = open_video_writer(video_path, fps=30.0, width=640, height=360)

    for pcd_file in pcd_files:
        base = os.path.splitext(os.path.basename(pcd_file))[0]
        try:
            frame = int(base)
        except ValueError:
            continue
        if frame not in detections_by_frame:
            continue

        out_img = os.path.join(output_dir, f"{frame:06d}.png")
        print(f"[GT-Ind] Rendering frame {frame} → {out_img}")

        try:
            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points)
            points = transform_lidar_to_camera(points)
        except Exception as e:
            print(f"[GT-Ind] Error loading {pcd_file}: {e}")
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

        yellow_rgb = (1.0, 1.0, 0.0)
        for det in detections_by_frame[frame]:
            for line in create_bbox_lines_ind(det):
                p1 = transform_lidar_to_camera(np.array([line[0]]))[0]
                p2 = transform_lidar_to_camera(np.array([line[1]]))[0]
                mesh = pv.Line(p1, p2)
                plotter.add_mesh(mesh, color=yellow_rgb, line_width=3)

        plotter.camera_position = [(-20, -10, 12), (0, 0, 0), (0, 0, 1)]
        plotter.show(screenshot=out_img, auto_close=True)

        frame_bgr = cv2.imread(out_img)
        if frame_bgr is not None:
            writer.write(frame_bgr)

    writer.release()
    display.stop()
    print("[GT-Ind] Video saved at:", video_path)



# Pipeline B: Ground Truth Group Visualization + Video
def load_gt_groups(gt_file: str):
    detections = {}
    try:
        with open(gt_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 12:
                    continue

                frame = int(parts[0])
                category = parts[1]
                cx, cy, cz = map(float, parts[2:5])
                xmin, ymin, zmin = map(float, parts[5:8])
                xmax, ymax, zmax = map(float, parts[8:11])

                length = xmax - xmin
                width  = ymax - ymin
                height = zmax - zmin

                detection = {
                    "category": category,
                    "center":   (cx, cy, cz),
                    "size":     (length, width, height),
                    "heading":  0.0
                }
                detections.setdefault(frame, []).append(detection)
    except Exception as e:
        print(f"[GT-Group] Error loading {gt_file}: {e}")
        return {}
    return detections


def create_bbox_lines_gtgroup(det: dict):
    cx, cy, cz = det["center"]
    l, w, h    = det["size"]
    yaw        = det["heading"]
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0

    # Define eight corners in local coordinates
    corners = np.array([
        [ dx,  dy,  dz],
        [ dx, -dy,  dz],
        [-dx, -dy,  dz],
        [-dx,  dy,  dz],
        [ dx,  dy, -dz],
        [ dx, -dy, -dz],
        [-dx, -dy, -dz],
        [-dx,  dy, -dz]
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    rotated = (R @ corners.T).T + np.array([cx, cy, cz])

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],   # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],   # Bottom face
        [0, 4], [1, 5], [2, 6], [3, 7]    # Vertical edges
    ]
    lines = np.array([[rotated[i], rotated[j]] for i, j in edges])
    return lines, rotated


def transform_lidar_to_camera(points_lidar: np.ndarray):
    """
    +90° rotation around Z to transform LiDAR coordinates into camera frame.
    """
    R = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    return (R @ points_lidar.T).T


def visualize_gt_groups(pcd_dir: str, gt_file: str, output_dir: str):
    detections_by_frame = load_gt_groups(gt_file)
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))

    os.makedirs(output_dir, exist_ok=True)
    display = Display(visible=0, size=(800, 600))
    display.start()

    video_path = "/content/gt_groups.mp4"
    writer = open_video_writer(video_path, fps=30.0, width=640, height=360)

    # Palette for up to 5 GT groups
    palette_hex = ["#FF0000", "#0000FF", "#008000", "#FFA500", "#800080"]
    group_colors = [
        tuple(int(h.lstrip("#")[i:i+2], 16)/255.0 for i in (0, 2, 4))
        for h in palette_hex
    ]

    for pcd_file in pcd_files:
        base = os.path.splitext(os.path.basename(pcd_file))[0]
        try:
            frame = int(base)
        except ValueError:
            continue
        if frame not in detections_by_frame:
            continue

        out_img = os.path.join(output_dir, f"{frame:06d}.png")
        print(f"[GT-Group] Rendering frame {frame} → {out_img}")

        try:
            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points)
            points = transform_lidar_to_camera(points)
        except Exception as e:
            print(f"[GT-Group] Error loading {pcd_file}: {e}")
            continue

        cloud = pv.PolyData(points)
        cloud["z_val"] = points[:, 2]

        plotter = Plotter(off_screen=True, window_size=(640, 360))
        plotter.background_color = "#eeeeee"

        # Add a reference plane
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                         i_size=30, j_size=30)
        plotter.add_mesh(plane, color="gray", opacity=0.15)
        plotter.add_mesh(cloud, scalars="z_val", cmap="viridis",
                         point_size=4, render_points_as_spheres=True,
                         show_scalar_bar=False)

        # Render each detection: yellow for pedestrians, colored for groups
        group_index = 0
        for det in detections_by_frame[frame]:
            lines, rotated = create_bbox_lines_gtgroup(det)
            if det["category"] == "pedestrian":
                color = (1.0, 1.0, 0.0)  # yellow
            elif det["category"] == "group":
                color = group_colors[group_index % len(group_colors)]
                group_index += 1
            else:
                continue

            for line in lines:
                p1 = transform_lidar_to_camera(np.array([line[0]]))[0]
                p2 = transform_lidar_to_camera(np.array([line[1]]))[0]
                mesh = pv.Line(p1, p2)
                plotter.add_mesh(mesh, color=color, line_width=3)

        plotter.camera_position = [(-20, -10, 12), (0, 0, 0), (0, 0, 1)]
        plotter.show(screenshot=out_img, auto_close=True)

        frame_bgr = cv2.imread(out_img)
        if frame_bgr is not None:
            writer.write(frame_bgr)

    writer.release()
    display.stop()
    print("[GT-Group] Video saved at:", video_path)



# Pipeline C: RPEA / DCCLA Model Output Visualization + Video
def load_rpea_detections(rpea_file: str):
    detections = {}
    try:
        with open(rpea_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 10:
                    continue
                frame = int(parts[0])
                det_id = int(parts[1])
                cx, cy, cz = map(float, parts[2:5])
                length, width, height = map(float, parts[5:8])
                heading = float(parts[8])
                det = {
                    "id": det_id,
                    "center": (cx, cy, cz),
                    "size":   (length, width, height),
                    "heading": heading
                }
                detections.setdefault(frame, []).append(det)
    except Exception as e:
        print(f"[RPEA / DCCLA] Error loading {rpea_file}: {e}")
        return {}
    return detections


def create_bbox_lines_rpea(det: dict):
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


def visualize_rpea(pcd_dir: str, rpea_file: str, output_dir: str):
    detections_by_frame = load_rpea_detections(rpea_file)
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))

    os.makedirs(output_dir, exist_ok=True)
    display = Display(visible=0, size=(800, 600))
    display.start()

    video_path = "/content/rpea.mp4"
    writer = open_video_writer(video_path, fps=30.0, width=640, height=360)

    for pcd_file in pcd_files:
        base = os.path.splitext(os.path.basename(pcd_file))[0]
        try:
            frame = int(base)
        except ValueError:
            continue
        if frame not in detections_by_frame:
            continue

        out_img = os.path.join(output_dir, f"{frame:06d}.png")
        print(f"[RPEA / DCCLA] Rendering frame {frame} → {out_img}")

        try:
            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points)
            points = transform_lidar_to_camera(points)
        except Exception as e:
            print(f"[RPEA / DCCLA] Error loading {pcd_file}: {e}")
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

        yellow_rgb = (1.0, 1.0, 0.0)
        for det in detections_by_frame[frame]:
            lines, rotated = create_bbox_lines_rpea(det)
            for line in lines:
                p1 = transform_lidar_to_camera(np.array([line[0]]))[0]
                p2 = transform_lidar_to_camera(np.array([line[1]]))[0]
                mesh = pv.Line(p1, p2)
                plotter.add_mesh(mesh, color=yellow_rgb, line_width=3)
            front = rotated[[0, 1, 5, 4]]
            front = transform_lidar_to_camera(front)
            cross1 = pv.Line(front[0], front[2])
            cross2 = pv.Line(front[1], front[3])
            plotter.add_mesh(cross1, color=yellow_rgb, line_width=3)
            plotter.add_mesh(cross2, color=yellow_rgb, line_width=3)

        plotter.camera_position = [(-20, -10, 12), (0, 0, 0), (0, 0, 1)]
        plotter.show(screenshot=out_img, auto_close=True)

        frame_bgr = cv2.imread(out_img)
        if frame_bgr is not None:
            writer.write(frame_bgr)

    writer.release()
    display.stop()
    print("[RPEA / DCCLA] Video saved at:", video_path)



# Pipeline D: Predicted Group Visualization + Video
def load_pred_groups(pred_file: str):
    """
    Load predicted group detections from group_detections.txt.
    Each line: frame,id,x,y,z,length,width,height,heading,score,group
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
                group_label = int(parts[10])
                det = {
                    "id": det_id,
                    "center": (cx, cy, cz),
                    "size":   (length, width, height),
                    "heading": heading,
                    "group_label": group_label
                }
                detections.setdefault(frame, []).append(det)
    except Exception as e:
        print(f"[Pred-Group] Error loading {pred_file}: {e}")
        return {}
    return detections


def create_bbox_lines_pred(det: dict):
    """
    Create rotated bounding box edges for a predicted detection.
    Returns (lines, rotated_corners).
    """
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


def create_bbox_lines_from_corners(min_corner: np.ndarray, max_corner: np.ndarray):
    """
    Create edges of an axis‐aligned box from its min and max corners.
    """
    x_min, y_min, z_min = min_corner
    x_max, y_max, z_max = max_corner
    corners = np.array([
        [x_max, y_max, z_max], [x_max, y_min, z_max],
        [x_min, y_min, z_max], [x_min, y_max, z_max],
        [x_max, y_max, z_min], [x_max, y_min, z_min],
        [x_min, y_min, z_min], [x_min, y_max, z_min]
    ])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    return np.array([[corners[i], corners[j]] for i, j in edges])


def visualize_pred_groups(pcd_dir: str, pred_file: str, output_dir: str):
    detections_by_frame = load_pred_groups(pred_file)
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))

    os.makedirs(output_dir, exist_ok=True)
    display = Display(visible=0, size=(800, 600))
    display.start()

    video_path = "/content/pred_groups.mp4"
    writer = open_video_writer(video_path, fps=30.0, width=640, height=360)

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

    for pcd_file in pcd_files:
        base = os.path.splitext(os.path.basename(pcd_file))[0]
        try:
            frame = int(base)
        except ValueError:
            continue
        if frame not in detections_by_frame:
            continue

        out_img = os.path.join(output_dir, f"{frame:06d}.png")
        print(f"[Pred‐Group] Rendering frame {frame} → {out_img}")

        try:
            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points)
            points = transform_lidar_to_camera(points)
        except Exception as e:
            print(f"[Pred‐Group] Error loading {pcd_file}: {e}")
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

        # Group detections by group_label
        grouped = {}
        for det in detections_by_frame[frame]:
            grp = det["group_label"]
            grouped.setdefault(grp, []).append(det)

        dynamic_idx = 5
        for grp_label, group_dets in grouped.items():
            # Assign group color
            if grp_label in predefined:
                color_hex = predefined[grp_label]
            else:
                while dynamic_idx < len(palette) and palette[dynamic_idx] in used:
                    dynamic_idx += 1
                color_hex = palette[dynamic_idx] if dynamic_idx < len(palette) else "#000000"
                used.add(color_hex)
                dynamic_idx += 1
            rgb = tuple(int(color_hex.lstrip("#")[i:i+2], 16) / 255.0 for i in (0, 2, 4))

            # Draw each box in this group
            for det in group_dets:
                lines, rotated = create_bbox_lines_pred(det)
                for line in lines:
                    p1 = transform_lidar_to_camera(np.array([line[0]]))[0]
                    p2 = transform_lidar_to_camera(np.array([line[1]]))[0]
                    mesh = pv.Line(p1, p2)
                    plotter.add_mesh(mesh, color=rgb, line_width=3)

                front = rotated[[0, 1, 5, 4]]
                front = transform_lidar_to_camera(front)
                cross1 = pv.Line(front[0], front[2])
                cross2 = pv.Line(front[1], front[3])
                plotter.add_mesh(cross1, color=rgb, line_width=3)
                plotter.add_mesh(cross2, color=rgb, line_width=3)

            # Draw merged bounding box for this group (white thick)
            all_min = []
            all_max = []
            for det in group_dets:
                cx, cy, cz = det["center"]
                l, w, h = det["size"]
                x_min, y_min, z_min = cx - l/2, cy - w/2, cz - h/2
                x_max, y_max, z_max = cx + l/2, cy + w/2, cz + h/2
                all_min.append([x_min, y_min, z_min])
                all_max.append([x_max, y_max, z_max])

            if all_min and all_max:
                merged_min = np.min(np.array(all_min), axis=0)
                merged_max = np.max(np.array(all_max), axis=0)
                merged_lines = create_bbox_lines_from_corners(merged_min, merged_max)
                for line in merged_lines:
                    p1 = transform_lidar_to_camera(np.array([line[0]]))[0]
                    p2 = transform_lidar_to_camera(np.array([line[1]]))[0]
                    mesh = pv.Line(p1, p2)
                    plotter.add_mesh(mesh, color="white", line_width=4)

        plotter.camera_position = [(-20, -10, 12), (0, 0, 0), (0, 0, 1)]
        plotter.show(screenshot=out_img, auto_close=True)

        frame_bgr = cv2.imread(out_img)
        if frame_bgr is not None:
            writer.write(frame_bgr)

    writer.release()
    display.stop()
    print("[Pred‐Group] Video saved at:", video_path)


# MAIN: Run All Four Visualization Pipelines
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 4 parallel 3D visualizations (GT Individual, GT Group, RPEA/DCCLA output, Predicted Groups)"
    )
    parser.add_argument(
        "--gt_folder",    required=True,
        help="Folder with per-frame GT .txt files"
    )
    parser.add_argument(
        "--gt_txt",       required=True,
        help="Output combined GT.txt"
    )
    parser.add_argument(
        "--pcd_dir",      required=True,
        help="Directory containing .pcd files"
    )
    parser.add_argument(
        "--det_file",     required=True,
        help="RPEA detection file (det.txt)"
    )
    parser.add_argument(
        "--pred_file",    required=True,
        help="Predicted group file (filtered_group_detections.txt)"
    )
    parser.add_argument(
        "--out_gt_ind",   required=True,
        help="Output dir for GT individual frames"
    )
    parser.add_argument(
        "--out_gt_group", required=True,
        help="Output dir for GT group frames"
    )
    parser.add_argument(
        "--out_detector",     required=True,
        help="Output dir for RPEA frames"
    )
    parser.add_argument(
        "--out_pred",     required=True,
        help="Output dir for predicted group frames"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Aggregate multiple GT .txt to single GT.txt
    os.makedirs(os.path.dirname(args.gt_txt), exist_ok=True)
    aggregate_gt(args.gt_folder, args.gt_txt)

    # Create output directories for PNGs
    os.makedirs(args.out_gt_ind, exist_ok=True)
    os.makedirs(args.out_gt_group, exist_ok=True)
    os.makedirs(args.out_detector, exist_ok=True)
    os.makedirs(args.out_pred, exist_ok=True)

    # Launch four parallel processes
    procs = [
        mp.Process(target=visualize_gt_individual,
                   args=(args.pcd_dir, args.gt_txt, args.out_gt_ind)),
        mp.Process(target=visualize_gt_groups,
                   args=(args.pcd_dir, args.gt_txt, args.out_gt_group)),
        mp.Process(target=visualize_rpea,
                   args=(args.pcd_dir, args.det_file, args.out_detector)),
        mp.Process(target=visualize_pred_groups,
                   args=(args.pcd_dir, args.pred_file, args.out_pred)),
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()

    print("\n[Main] All visualizations completed.")


if __name__ == "__main__":
    main()
