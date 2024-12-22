import cv2
import json
import numpy as np

CLASS_COLORS = {
    0: (255, 165, 0),  # Bus
    1: (0, 255, 0),  # Car
    2: (255, 0, 0),  # Bicycle
    3: (255, 0, 255),  # Scooter
    4: (0, 255, 255),  # Motorbike
    5: (255, 255, 0),  # Pedestrian
    6: (128, 0, 128),  # Truck
}


def load_tracking_data(tracking_file):
    with open(tracking_file, "r") as f:
        tracking_data = json.load(f)
    frame_indexed_tracking = {}
    for result in tracking_data["tracking_results"]:
        frame_indexed_tracking[result["i"]] = result["res"]
    return frame_indexed_tracking, tracking_data["video"]


def visualize_trajectory(tracking_data, video_file, output_file, tail_length=30):
    cap = cv2.VideoCapture(video_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    trajectory_frame = np.zeros((height, width, 3), dtype=np.uint8)

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

                center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                tail_storage.setdefault(tid, []).append(center)
                tail_storage[tid] = tail_storage[tid][-tail_length:]

                points = np.array(tail_storage[tid], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    trajectory_frame,
                    [points],
                    False,
                    CLASS_COLORS.get(cid, (255, 255, 255)),
                    2,
                )

        # Display the trajectory frame
        cv2.imshow("Trajectory Visualization", trajectory_frame)
        out.write(trajectory_frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tracking_file = "outputs/DJI_0268/tracking_0.json"
    video_file = "inputs/DJI_0268/DJI_0268.mp4"
    output_file = "outputs/DJI_0268/DJI_0268_trajectory.mp4"

    tracking_data, _ = load_tracking_data(tracking_file)
    visualize_trajectory(tracking_data, video_file, output_file, tail_length=30)
