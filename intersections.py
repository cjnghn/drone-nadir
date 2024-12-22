import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from tqdm import tqdm
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count


def create_track_data(tracking_data: dict) -> pd.DataFrame:
    records = []
    for frame in tracking_data["tracking_results"]:
        frame_records = [
            {
                "track_id": obj["tid"],
                "class_id": obj["cid"],
                "center_x": (obj["bbox"][0] + obj["bbox"][2]) / 2,
                "center_y": (obj["bbox"][1] + obj["bbox"][3]) / 2,
                "frame_index": frame["i"],
                "time": frame["time(millisecond)"] / 1000.0,
                "speed(m/s)": obj.get("speed(m/s)", 0.0),  # 속도 정보 추가
            }
            for obj in frame["res"]
        ]
        records.extend(frame_records)
    return pd.DataFrame(records)


def calculate_segment_angle(track_data: pd.DataFrame, frame_idx: int) -> float:
    nearby_points = track_data[
        (track_data["frame_index"] >= frame_idx - 1)
        & (track_data["frame_index"] <= frame_idx + 1)
    ].sort_values("frame_index")

    if len(nearby_points) >= 2:
        dx = nearby_points["center_x"].iloc[-1] - nearby_points["center_x"].iloc[0]
        dy = nearby_points["center_y"].iloc[-1] - nearby_points["center_y"].iloc[0]
        return np.degrees(np.arctan2(dy, dx))
    return 0.0


def process_intersection(args: Tuple) -> List[dict]:
    line1, line2, track_id1, track_id2, track1_data, track2_data = args
    if not line1.intersects(line2):
        return []

    intersection = line1.intersection(line2)
    points = [intersection] if isinstance(intersection, Point) else intersection.geoms
    results = []

    for pt in points:
        track1_points = track1_data[["center_x", "center_y"]].values
        track2_points = track2_data[["center_x", "center_y"]].values

        distances1 = np.sqrt(np.sum((track1_points - [pt.x, pt.y]) ** 2, axis=1))
        distances2 = np.sqrt(np.sum((track2_points - [pt.x, pt.y]) ** 2, axis=1))

        idx1 = np.argmin(distances1)
        idx2 = np.argmin(distances2)

        frame_idx1 = track1_data.iloc[idx1]["frame_index"]
        frame_idx2 = track2_data.iloc[idx2]["frame_index"]
        time1 = track1_data.iloc[idx1]["time"]
        time2 = track2_data.iloc[idx2]["time"]

        angle1 = calculate_segment_angle(track1_data, int(frame_idx1))
        angle2 = calculate_segment_angle(track2_data, int(frame_idx2))
        angle_diff = abs(angle1 - angle2)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # TODO: 여기서 처리하는거 맘에 들진 않지만 일단 이대로 진행
        speed1_row = track1_data[(track1_data["frame_index"] == frame_idx1)]
        speed2_row = track2_data[(track2_data["frame_index"] == frame_idx2)]
        speed1 = (
            speed1_row["speed(m/s)"].iloc[0]
            if "speed(m/s)" in speed1_row.columns and not speed1_row.empty
            else 0.0
        )
        speed2 = (
            speed2_row["speed(m/s)"].iloc[0]
            if "speed(m/s)" in speed2_row.columns and not speed2_row.empty
            else 0.0
        )

        results.append(
            {
                "track_id_1": track_id1,
                "track_id_2": track_id2,
                "class_id_1": int(track1_data["class_id"].iloc[0]),
                "class_id_2": int(track2_data["class_id"].iloc[0]),
                "intersection_x": pt.x,
                "intersection_y": pt.y,
                "frame_at_intersection_1": int(frame_idx1),
                "frame_at_intersection_2": int(frame_idx2),
                "angle_at_intersection": angle_diff,
                "time_difference": abs(time1 - time2),
                "speed_at_intersection_1": speed1,
                "speed_at_intersection_2": speed2,
            }
        )

    return results


def calculate_intersections(tracking_data: dict) -> list:
    df = create_track_data(tracking_data)

    track_lines = {
        track_id: LineString(group[["center_x", "center_y"]].values)
        for track_id, group in df.groupby("track_id")
        if len(group) > 1
    }

    geometries, track_ids = list(track_lines.values()), list(track_lines.keys())
    rtree = STRtree(geometries)

    intersection_tasks = []
    processed_pairs = set()

    for i, track_id1 in enumerate(track_ids):
        line1 = track_lines[track_id1]
        track1_data = df[df["track_id"] == track_id1]
        class_id1 = track1_data["class_id"].iloc[0]

        for j in rtree.query(line1):
            track_id2 = track_ids[j]
            if track_id1 == track_id2:
                continue

            track2_data = df[df["track_id"] == track_id2]
            class_id2 = track2_data["class_id"].iloc[0]

            if class_id1 == class_id2:
                continue

            pair = tuple(sorted([track_id1, track_id2]))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            line2 = track_lines[track_id2]
            intersection_tasks.append(
                (line1, line2, track_id1, track_id2, track1_data, track2_data)
            )

    # 병렬 처리를 위한 설정
    num_processes = cpu_count()
    chunk_size = max(
        1, len(intersection_tasks) // (num_processes * 4)
    )  # 작업 분배 최적화

    intersections = []
    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(
                    process_intersection, intersection_tasks, chunksize=chunk_size
                ),
                total=len(intersection_tasks),
                desc="Processing intersections",
            )
        )
        for result in results:
            intersections.extend(result)

    intersections.sort(key=lambda x: x["time_difference"])
    return intersections
