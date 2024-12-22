import cv2
import json
import numpy as np

"""
0: bus
1: car
2: cyclist
3: scooter_rider
4: motorcyclist
5: pedestrian
6: truck
"""

CLASS_MAPPING = {
    0: "bus",
    1: "car",
    2: "cyclist",
    3: "scooter_rider",
    4: "motorcyclist",
    5: "pedestrian",
    6: "truck",
}

CLASS_COLORS = {
    0: (255, 165, 0),  # Bus
    1: (0, 255, 0),  # Car
    2: (255, 0, 0),  # Bicycle
    3: (255, 0, 255),  # Scooter
    4: (0, 255, 255),  # Motorbike
    5: (255, 255, 0),  # Pedestrian
    6: (128, 0, 128),  # Truck
}


def load_data(intersection_file, tracking_file):
    with open(intersection_file, "r") as f:
        intersection_data = json.load(f)

    with open(tracking_file, "r") as f:
        tracking_data = json.load(f)

    frame_indexed_tracking = {}
    for result in tracking_data["tracking_results"]:
        frame_indexed_tracking[result["i"]] = result["res"]

    frame_indexed_intersections = {}
    for intersection in intersection_data:
        frame1 = intersection["frame_at_intersection_1"]
        frame2 = intersection["frame_at_intersection_2"]
        if frame1 not in frame_indexed_intersections:
            frame_indexed_intersections[frame1] = []
        if frame2 not in frame_indexed_intersections:
            frame_indexed_intersections[frame2] = []
        frame_indexed_intersections[frame1].append(intersection)
        frame_indexed_intersections[frame2].append(intersection)

    return frame_indexed_intersections, frame_indexed_tracking, tracking_data["video"]


class VisualizationConfig:
    def __init__(self):
        self.show_bbox = True
        self.show_track_id = True
        self.show_class_name = True
        self.show_tail = True
        self.show_gps = True
        self.show_intersection = True
        self.tail_length = 30


class VisualizationStyle:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.9
    FONT_THICKNESS = 2
    LINE_THICKNESS = 4
    INTERSECTION_RADIUS = 5
    TEXT_PADDING = 4


def draw_bbox(frame, bbox, track_id, class_id, config):
    if not config.show_bbox:
        return

    x1, y1, x2, y2 = bbox.astype(int)
    color = CLASS_COLORS[class_id]

    # Draw bounding box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        VisualizationStyle.LINE_THICKNESS,
    )

    text_y = y1 - VisualizationStyle.TEXT_PADDING

    # Draw track ID
    if config.show_track_id:
        id_text = f"ID: {track_id}"

        (id_width, id_height), _ = cv2.getTextSize(
            id_text,
            VisualizationStyle.FONT,
            VisualizationStyle.FONT_SCALE,
            VisualizationStyle.FONT_THICKNESS,
        )

        # Draw ID background
        cv2.rectangle(
            frame,
            (x1, text_y - id_height - VisualizationStyle.TEXT_PADDING),
            (x1 + id_width + VisualizationStyle.TEXT_PADDING, text_y),
            (0, 0, 0),
            -1,
        )

        # Draw ID text
        cv2.putText(
            frame,
            id_text,
            (x1, text_y - VisualizationStyle.TEXT_PADDING),
            VisualizationStyle.FONT,
            VisualizationStyle.FONT_SCALE,
            (255, 255, 255),
            VisualizationStyle.FONT_THICKNESS,
        )

        text_y -= id_height + 2 * VisualizationStyle.TEXT_PADDING

    # Draw class name
    if config.show_class_name:
        class_text = CLASS_MAPPING[class_id]

        (class_width, class_height), _ = cv2.getTextSize(
            class_text,
            VisualizationStyle.FONT,
            VisualizationStyle.FONT_SCALE,
            VisualizationStyle.FONT_THICKNESS,
        )

        # Draw class background
        cv2.rectangle(
            frame,
            (x1, text_y - class_height - VisualizationStyle.TEXT_PADDING),
            (x1 + class_width + VisualizationStyle.TEXT_PADDING, text_y),
            (0, 0, 0),
            -1,
        )

        # Draw class text
        cv2.putText(
            frame,
            class_text,
            (x1, text_y - VisualizationStyle.TEXT_PADDING),
            VisualizationStyle.FONT,
            VisualizationStyle.FONT_SCALE,
            (255, 255, 255),
            VisualizationStyle.FONT_THICKNESS,
        )


def draw_tail(frame, trajectory, config):
    if not config.show_tail or len(trajectory) < 2:
        return

    points = np.array(trajectory[-config.tail_length :])
    points = points.astype(np.int32)
    points = points.reshape((-1, 1, 2))

    if len(points) >= 2:
        cv2.polylines(
            frame,
            [points],
            False,
            (255, 255, 255),  # White for better contrast
            VisualizationStyle.LINE_THICKNESS,
        )


def draw_gps(frame, latitude, longitude, position, config):
    if not config.show_gps:
        return

    text = f"Lat: {latitude:.6f}, Lon: {longitude:.6f}"
    x, y = position

    (text_width, text_height), _ = cv2.getTextSize(
        text,
        VisualizationStyle.FONT,
        VisualizationStyle.FONT_SCALE,
        VisualizationStyle.FONT_THICKNESS,
    )

    cv2.rectangle(
        frame,
        (x, y - text_height - VisualizationStyle.TEXT_PADDING),
        (x + text_width, y + VisualizationStyle.TEXT_PADDING),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        VisualizationStyle.FONT,
        VisualizationStyle.FONT_SCALE,
        (255, 255, 255),
        VisualizationStyle.FONT_THICKNESS,
    )


def draw_intersection(frame, intersection, config):
    if not config.show_intersection:
        return

    x, y = map(int, (intersection["intersection_x"], intersection["intersection_y"]))

    cv2.circle(
        frame,
        (x, y),
        VisualizationStyle.INTERSECTION_RADIUS,
        (0, 0, 255),  # Red for intersections
        -1,
    )


def draw_velocity(frame, velocity, position, config):
    if not config.show_gps:  # velocity는 GPS 표시 옵션과 함께 제어
        return

    km_per_hour = velocity * 3.6
    text = f"{km_per_hour:.2f} km/h"
    x, y = position

    (text_width, text_height), _ = cv2.getTextSize(
        text,
        VisualizationStyle.FONT,
        VisualizationStyle.FONT_SCALE,
        VisualizationStyle.FONT_THICKNESS,
    )

    cv2.rectangle(
        frame,
        (x, y - text_height - VisualizationStyle.TEXT_PADDING),
        (x + text_width, y + VisualizationStyle.TEXT_PADDING),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        VisualizationStyle.FONT,
        VisualizationStyle.FONT_SCALE,
        (255, 255, 255),
        VisualizationStyle.FONT_THICKNESS,
    )


def visualize_video(
    intersection_data, tracking_data, video_file, output_file, config=None, show=False
):
    if config is None:
        config = VisualizationConfig()

    cap = cv2.VideoCapture(video_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    tail_storage = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in tracking_data:
            for obj in tracking_data[frame_idx]:
                tid = obj["tid"]
                cid = obj["cid"]
                bbox = np.array(obj["bbox"])

                draw_bbox(frame, bbox, tid, cid, config)

                # GPS 정보 위치 수정
                if "latitude" in obj and "longitude" in obj and config.show_gps:
                    gps_x = int(bbox[0])  # bbox의 왼쪽 x좌표
                    gps_y = int(bbox[3]) + 30  # bbox의 아래쪽 y좌표 + 여유공간
                    draw_gps(
                        frame, obj["latitude"], obj["longitude"], (gps_x, gps_y), config
                    )

                    # velocity 정보 추가 (GPS 아래에 표시)
                    if "velocity(m/s)" in obj:
                        velocity_y = gps_y + 30  # GPS 텍스트 아래에 30픽셀 간격
                        draw_velocity(
                            frame, obj["velocity(m/s)"], (gps_x, velocity_y), config
                        )

                if config.show_tail:
                    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    tail_storage.setdefault(
                        tid, {"points": [], "color": CLASS_COLORS[cid]}
                    ).setdefault("points", []).append(center)
                    points = tail_storage[tid]["points"]
                    if len(points) >= 2:
                        points_arr = np.array(points[-config.tail_length :])
                        points_arr = points_arr.astype(np.int32)
                        points_arr = points_arr.reshape((-1, 1, 2))
                        cv2.polylines(
                            frame,
                            [points_arr],
                            False,
                            tail_storage[tid]["color"],
                            VisualizationStyle.LINE_THICKNESS,
                        )

        if frame_idx in intersection_data:
            for intersection in intersection_data[frame_idx]:
                draw_intersection(frame, intersection, config)

        out.write(frame)

        if show:
            cv2.imshow("Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    intersection_file = "outputs/DJI_0268/intersections_0.json"
    tracking_file = "outputs/DJI_0268/tracking_0.json"
    video_file = "inputs/DJI_0268/DJI_0268.mp4"
    output_file = "outputs/DJI_0268/DJI_0268_bbox_tracking_gps.mp4"

    config = VisualizationConfig()
    config.show_bbox = True
    config.show_class_name = True
    config.show_track_id = True
    config.show_tail = True
    config.tail_length = 30
    config.show_gps = True
    config.show_intersection = False

    intersection_data, tracking_data, video_info = load_data(
        intersection_file, tracking_file
    )
    visualize_video(
        intersection_data,
        tracking_data,
        video_file,
        output_file,
        config,
        show=True,
    )
