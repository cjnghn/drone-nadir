import cv2
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from visualization_intersection import (
    CLASS_MAPPING,
    CLASS_COLORS,
    put_text_with_background,
)


def load_data(tracking_file, intersection_file):
    # Load tracking data
    with open(tracking_file, "r") as f:
        tracking_data = json.load(f)

    # Load intersections
    with open(intersection_file, "r") as f:
        intersections = json.load(f)

    return tracking_data, intersections


def draw_focused_trajectories(
    frame, tracking_data, track_ids, current_frame, start_frame
):
    # Dictionary to store trajectory points for each track
    trajectories = defaultdict(list)

    # Collect all trajectory points from start_frame to current_frame
    for frame_data in tracking_data["tracking_results"]:
        if start_frame <= frame_data["i"] <= current_frame:
            for obj in frame_data["res"]:
                if obj["tid"] in track_ids:
                    bbox = obj["bbox"]
                    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    trajectories[obj["tid"]].append(
                        {
                            "frame": frame_data["i"],
                            "center": center,
                            "bbox": bbox,
                            "class_id": obj["cid"],
                            "speed": obj.get("speed(m/s)"),
                        }
                    )

    # Draw trajectories and current positions
    for track_id in track_ids:
        if track_id in trajectories:
            points = trajectories[track_id]
            # Draw trajectory line
            if len(points) >= 2:
                pts = np.array([p["center"] for p in points], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 0), 2)

            # Draw current position (last point)
            if points:
                last_point = points[-1]
                bbox = last_point["bbox"]
                class_id = last_point["class_id"]
                class_name = CLASS_MAPPING.get(class_id, "unknown")
                color = CLASS_COLORS.get(class_name, (255, 255, 255))

                # Draw bbox
                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color,
                    2,
                )

                # Show ID and class
                id_text = f"ID: {track_id} ({class_name})"
                put_text_with_background(
                    frame,
                    id_text,
                    (int(bbox[0]), int(bbox[1] - 35)),  # Changed from -25 to -35
                )

                # Show speed if available
                if last_point["speed"] is not None:
                    speed_text = f"{last_point['speed']:.1f} m/s"
                    put_text_with_background(
                        frame, speed_text, (int(bbox[0]), int(bbox[1] - 10))
                    )


def get_object_bbox_at_frame(tracking_data, track_id, frame_i):
    for frame_data in tracking_data["tracking_results"]:
        if frame_data["i"] == frame_i:
            for obj in frame_data["res"]:
                if obj["tid"] == track_id:
                    return obj["bbox"]
    return None


def preload_frames(cap, start_frame, end_frame):
    """Helper function to preload frames for an intersection"""
    frames = []
    current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    return frames

def get_intersection_frame_range(intersection, total_frames, padding=30):
    """Helper function to get frame range for an intersection"""
    start_frame = min(
        intersection["frame_at_intersection_1"],
        intersection["frame_at_intersection_2"],
    ) - padding
    end_frame = max(
        intersection["frame_at_intersection_1"],
        intersection["frame_at_intersection_2"],
    ) + padding
    return max(0, start_frame), min(total_frames - 1, end_frame)

def visualize_intersections_focus(
    tracking_file, intersection_file, video_file, save_mode=False, output_dir=None
):
    tracking_data, intersections = load_data(tracking_file, intersection_file)
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    if save_mode and output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

    current_idx = 0
    frame_buffer = {}

    # Initial preload of first intersection
    start_frame, end_frame = get_intersection_frame_range(intersections[0], total_frames)
    frame_buffer[0] = preload_frames(cap, start_frame, end_frame)

    while (save_mode and current_idx < len(intersections)) or (not save_mode):
        if not save_mode:
            if current_idx < 0:
                current_idx = len(intersections) - 1
            elif current_idx >= len(intersections):
                current_idx = 0

        if save_mode and current_idx >= len(intersections):
            break

        intersection = intersections[current_idx]
        start_frame, end_frame = get_intersection_frame_range(intersection, total_frames)

        # Preload next intersection if not in buffer
        next_idx = (current_idx + 1) % len(intersections)
        if not save_mode and next_idx not in frame_buffer:
            next_start, next_end = get_intersection_frame_range(intersections[next_idx], total_frames)
            frame_buffer[next_idx] = preload_frames(cap, next_start, next_end)

        # Ensure current intersection is loaded
        if current_idx not in frame_buffer:
            frame_buffer[current_idx] = preload_frames(cap, start_frame, end_frame)

        frames = frame_buffer[current_idx]
        frame_idx = 0
        playing = True

        if save_mode:
            # Save mode일 경우 현재 intersection에 해당하는 구간을 영상으로 저장
            output_path = os.path.join(
                output_dir, f"intersection_{current_idx:03d}.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                cap.get(cv2.CAP_PROP_FPS),
                (frames[0].shape[1], frames[0].shape[0]),
            )

            # 여기가 핵심 수정사항: enumerate를 사용하여 f_idx로 현재 프레임 인덱스 관리
            for f_idx, frame in enumerate(
                tqdm(
                    frames,
                    desc=f"Saving intersection {current_idx + 1}/{len(intersections)}",
                )
            ):
                frame_copy = frame.copy()
                current_frame = (
                    start_frame + f_idx
                )  # f_idx를 이용한 current_frame 업데이트

                # Draw trajectories and current objects with speed
                track_ids = [intersection["track_id_1"], intersection["track_id_2"]]
                draw_focused_trajectories(
                    frame_copy, tracking_data, track_ids, current_frame, start_frame
                )

                # Draw intersection point
                x = int(intersection["intersection_x"])
                y = int(intersection["intersection_y"])
                cv2.circle(frame_copy, (x, y), 10, (0, 0, 255), -1)

                # Display information
                info_text = [
                    f"Intersection {current_idx + 1}/{len(intersections)}",
                    f"Frames: {start_frame}-{end_frame}",
                    f"Current frame: {current_frame}",
                    f"Track {intersection['track_id_1']}({CLASS_MAPPING[intersection['class_id_1']]}) vs",
                    f"Track {intersection['track_id_2']}({CLASS_MAPPING[intersection['class_id_2']]})",
                ]

                # Add angle information if available
                if "angle_at_intersection" in intersection:
                    info_text.append(
                        f"Angle: {intersection['angle_at_intersection']:.1f}°"
                    )

                # Display each line of text
                for i, text in enumerate(info_text):
                    put_text_with_background(frame_copy, text, (30, 30 + i * 30))

                out.write(frame_copy)

            out.release()
            frame_buffer.pop(current_idx, None)
            current_idx += 1
            continue

        # Interactive mode
        while True:
            if playing and frame_idx < len(frames):
                frame = frames[frame_idx].copy()
                current_frame = start_frame + frame_idx

                # Draw trajectories and current objects with speed
                track_ids = [intersection["track_id_1"], intersection["track_id_2"]]
                draw_focused_trajectories(
                    frame, tracking_data, track_ids, current_frame, start_frame
                )

                # Draw intersection point
                x = int(intersection["intersection_x"])
                y = int(intersection["intersection_y"])
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

                # Display information
                info_text = [
                    f"Intersection {current_idx + 1}/{len(intersections)}",
                    f"Frames: {start_frame}-{end_frame}",
                    f"Current frame: {current_frame}",
                    f"Track {intersection['track_id_1']}({CLASS_MAPPING[intersection['class_id_1']]}) vs",
                    f"Track {intersection['track_id_2']}({CLASS_MAPPING[intersection['class_id_2']]})",
                ]

                for i, text in enumerate(info_text):
                    put_text_with_background(frame, text, (30, 30 + i * 30))

                put_text_with_background(
                    frame,
                    "Press 1: Previous, 2: Next, Space: Pause/Play, Q: Quit",
                    (30, frame.shape[0] - 30),
                )

                cv2.imshow("Intersection Focus Mode", frame)
                frame_idx += 1
                
                # 프레임 재생이 끝나면 다음으로 넘어감
                if frame_idx >= len(frames):
                    # Clear old buffer
                    frame_buffer.pop(current_idx, None)
                    current_idx += 1
                    break

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key in [ord("1"), ord("2")]:
                # Clear old buffers
                prev_idx = (current_idx - 1) % len(intersections)
                if prev_idx in frame_buffer and key == ord("2"):
                    frame_buffer.pop(prev_idx)
                
                current_idx = current_idx - 1 if key == ord("1") else current_idx + 1
                current_idx = current_idx % len(intersections)
                break
            elif key == ord(" "):
                playing = not playing

            if not playing:
                cv2.waitKey(50)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tracking_file = "outputs/DJI_0268/tracking_0.json"
    intersection_file = "outputs/DJI_0268/intersections_0.json"
    video_file = "inputs/DJI_0268/DJI_0268.MP4"

    # Interactive mode:
    visualize_intersections_focus(tracking_file, intersection_file, video_file)

    # Save mode:
    # output_dir = "outputs/DJI_0268/intersection_frames"
    # visualize_intersections_focus(tracking_file, intersection_file, video_file, save_mode=True, output_dir=output_dir)
