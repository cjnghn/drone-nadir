import cv2
import json
import numpy as np
from collections import defaultdict

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
    "bus": (255, 165, 0),  # Bus
    "car": (0, 255, 0),  # Car
    "cyclist": (255, 0, 0),  # Bicycle
    "scooter_rider": (255, 0, 255),  # Scooter
    "motorcyclist": (0, 255, 255),  # Motorbike
    "pedestrian": (255, 255, 0),  # Pedestrian
    "truck": (128, 0, 128),  # Truck
}


def load_tracking_data(tracking_file):
    with open(tracking_file, "r") as f:
        tracking_data = json.load(f)
    frame_indexed_tracking = {}
    for result in tracking_data["tracking_results"]:
        frame_indexed_tracking[result["i"]] = result["res"]
    return frame_indexed_tracking, tracking_data["video"]


def load_intersection_data(intersection_file):
    with open(intersection_file, "r") as f:
        intersections = json.load(f)

    # 프레임별로 교차점 정리
    frame_to_intersections = defaultdict(list)
    for intersection in intersections:
        frame = intersection["frame_at_intersection_1"]  # 첫 번째 객체 기준
        frame_to_intersections[frame].append(intersection)

    return frame_to_intersections


def put_text_with_background(img, text, position, scale=0.6, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Add padding
    padding = 5
    cv2.rectangle(
        img,
        (position[0] - padding, position[1] - text_h - padding),
        (position[0] + text_w + padding, position[1] + padding),
        (0, 0, 0),
        -1,
    )

    cv2.putText(img, text, position, font, scale, (255, 255, 255), thickness)


def draw_all_trajectories(tracking_data, width, height):
    trajectory_frame = np.zeros((height, width, 3), dtype=np.uint8)
    object_trajectories = defaultdict(list)

    for frame_idx in sorted(tracking_data.keys()):
        for obj in tracking_data[frame_idx]:
            tid = obj["tid"]
            bbox = np.array(obj["bbox"])
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            object_trajectories[tid].append((center, obj["cid"]))

    for tid, points in object_trajectories.items():
        centers = np.array([p[0] for p in points], dtype=np.int32).reshape((-1, 1, 2))
        class_id = points[0][1]
        class_name = CLASS_MAPPING.get(class_id, "unknown")
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        cv2.polylines(trajectory_frame, [centers], False, color, 2)

    return trajectory_frame


def visualize_intersections(tracking_file, intersection_file, video_file, output_file):
    # 데이터 로드
    tracking_data, video_info = load_tracking_data(tracking_file)
    intersections = load_intersection_data(intersection_file)

    # 비디오 설정
    cap = cv2.VideoCapture(video_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 출력 비디오 설정 - XVID 코덱 사용
    out = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height), isColor=True
    )

    # 프레임 버퍼 설정
    buffer_size = 32
    frame_buffer = []

    # 전체 궤적 미리 그리기
    base_frame = draw_all_trajectories(tracking_data, width, height)

    # intersection이 있는 프레임만 처리
    intersection_frames = sorted(intersections.keys())

    for frame_idx in intersection_frames:

        current_frame = base_frame.copy()

        # 프레임 번호 표시
        frame_text = f"Frame: {frame_idx}"
        put_text_with_background(current_frame, frame_text, (width - 150, 30))

        # 현재 프레임의 교차점 표시
        for intersection in intersections[frame_idx]:
            x = int(intersection["intersection_x"])
            y = int(intersection["intersection_y"])

            # 더 큰 빨간색 원으로 교차점 표시
            cv2.circle(current_frame, (x, y), 10, (0, 0, 255), -1)

            # 교차하는 객체들의 정보 표시
            class1 = CLASS_MAPPING.get(intersection["class_id_1"], "unknown")
            class2 = CLASS_MAPPING.get(intersection["class_id_2"], "unknown")
            info_text = [
                f"T{intersection['track_id_1']}({class1}) vs T{intersection['track_id_2']}({class2})"
            ]

            # 각 객체의 교차 프레임과 시간 정보 표시
            frame1 = intersection["frame_at_intersection_1"]
            frame2 = intersection["frame_at_intersection_2"]
            time1 = frame1 / fps
            time2 = frame2 / fps

            info_text.append(f"Obj1 Frame/Time: {frame1}/{time1:.2f}s")
            info_text.append(f"Obj2 Frame/Time: {frame2}/{time2:.2f}s")

            # 속도 정보가 있다면 표시
            if (
                "speed_at_intersection_1" in intersection
                and "speed_at_intersection_2" in intersection
            ):
                info_text.append(
                    f"Speed: {intersection['speed_at_intersection_1']:.1f} vs {intersection['speed_at_intersection_2']:.1f} m/s"
                )

            # 교차 각도 정보가 있다면 표시
            if "angle_at_intersection" in intersection:
                angle = intersection["angle_at_intersection"]
                info_text.append(f"Angle: {angle:.1f}°")

            # 위도/경도 정보가 있다면 표시
            if "latitude" in intersection and "longitude" in intersection:
                info_text.append(f"Lat: {intersection['latitude']:.6f}")
                info_text.append(f"Lon: {intersection['longitude']:.6f}")

            # 텍스트 정보 표시
            for i, text in enumerate(info_text):
                put_text_with_background(current_frame, text, (x + 15, y + 20 + i * 25))

        # 버퍼에 프레임 추가
        frame_buffer.append(current_frame)

        # 버퍼가 가득 차면 한번에 쓰기
        if len(frame_buffer) >= buffer_size:
            for buffered_frame in frame_buffer:
                out.write(buffered_frame)
            frame_buffer = []

        # 화면 표시 (옵션)
        cv2.imshow("Intersection Visualization", current_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 남은 버퍼 쓰기
    for buffered_frame in frame_buffer:
        out.write(buffered_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tracking_file = "outputs/DJI_0268/tracking_0.json"
    intersection_file = "outputs/DJI_0268/intersections_0.json"
    video_file = "inputs/DJI_0268/DJI_0268.mp4"
    output_file = "outputs/DJI_0268/DJI_0268_intersections.mp4"

    visualize_intersections(tracking_file, intersection_file, video_file, output_file)
