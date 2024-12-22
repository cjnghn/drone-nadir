import json
import pandas as pd


def load_tracking_json(file_path: str) -> dict:
    """
    트래킹 JSON 파일을 로드합니다.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def add_flight_log_to_tracking(
    tracking_data: dict,
    interpolated_flight_log: pd.DataFrame,
) -> dict:
    """
    보간된 비행 로그 데이터로 트래킹 데이터를 보강합니다.
    """
    if len(tracking_data["tracking_results"]) != interpolated_flight_log.shape[0]:
        raise ValueError(
            f"The number of tracking results must match the number of interpolated frames."
            f"Expected {len(tracking_data['tracking_results'])}, but got {interpolated_flight_log.shape[0]}."
        )

    for i, tracking_result in enumerate(tracking_data["tracking_results"]):
        for col in interpolated_flight_log.columns:
            tracking_result[col] = interpolated_flight_log[col].iloc[i]

    return tracking_data


if __name__ == "__main__":
    from flight_log import (
        load_flight_log,
        interpolate_flight_log,
        extract_video_segment,
    )

    flight_log = load_flight_log("flight_logs/flight_log.csv")
    video_segments = extract_video_segment(flight_log)
    tracking_results = [
        load_tracking_json("tracking_results/tracking1.json"),
        load_tracking_json("tracking_results/tracking2.json"),
    ]

    if len(video_segments) != len(tracking_results):
        print(f"video_segments: {len(video_segments)}")
        print(f"tracking_results: {len(tracking_results)}")
        raise ValueError(
            f"The number of video segments must match the number of tracking results."
        )

    for i, (segment, tracking_result) in enumerate(
        zip(video_segments, tracking_results)
    ):
        fps = tracking_result["video"]["fps"]
        total_frames = tracking_result["video"]["total_frames"]

        interpolated_log = interpolate_flight_log(segment, fps, total_frames)
        tracking_data = add_flight_log_to_tracking(tracking_result, interpolated_log)
        print(f"{i + 1} has {len(tracking_data['tracking_results'])} frames.")
