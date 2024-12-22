import logging
import json
from pathlib import Path
from flight_log import load_flight_log, extract_video_segment, interpolate_flight_log
from tracking import load_tracking_json, add_flight_log_to_tracking
from georeferencing import (
    georeference_tracking_data,
    georeference_intersections,
)
from intersections import calculate_intersections

# 여기서 경로를 수정하세요.

# FLIGHT_LOG_PATH = "inputs/DJI_0268/flight_log.csv"
# TRACKING_PATHS = [
#     "inputs/DJI_0268/DJI_0268.json",
# ]
# OUTPUT_BASE_DIR = "outputs/DJI_0268"

# FLIGHT_LOG_PATH = "inputs/DJI_0269,DJI_0271,DJI_0272,DJI_0273/Sep-25th-2024-06-04PM-Flight-Airdata.csv"
# TRACKING_PATHS = [
#     "inputs/DJI_0269,DJI_0271,DJI_0272,DJI_0273/DJI_0269.json",
#     "inputs/DJI_0269,DJI_0271,DJI_0272,DJI_0273/DJI_0271.json",
#     "inputs/DJI_0269,DJI_0271,DJI_0272,DJI_0273/DJI_0272.json",
#     "inputs/DJI_0269,DJI_0271,DJI_0272,DJI_0273/DJI_0273.json",
# ]
# OUTPUT_BASE_DIR = "outputs/DJI_0269,DJI_0271,DJI_0272,DJI_0273"

FLIGHT_LOG_PATH = "inputs/DJI_0279,DJI_0280/flight_log.csv"
TRACKING_PATHS = [
    "inputs/DJI_0279,DJI_0280/DJI_0279.json",
    "inputs/DJI_0279,DJI_0280/DJI_0280.json",
]
OUTPUT_BASE_DIR = "outputs/DJI_0279,DJI_0280"

# FLIGHT_LOG_PATH = "inputs/DJI_0287/Dec-16th-2024-10-41AM-Flight-Airdata.csv"
# TRACKING_PATHS = [
#     "inputs/DJI_0287/DJI_0287.json",
# ]
# OUTPUT_BASE_DIR = "outputs/DJI_0287"

# FLIGHT_LOG_PATH = "inputs/DJI_0289/Dec-16th-2024-10-59AM-Flight-Airdata.csv"
# TRACKING_PATHS = [
#     "inputs/DJI_0289/DJI_0289.json",
# ]
# OUTPUT_BASE_DIR = "outputs/DJI_0289"
DJI_MINI_2_FOV = 71.8


def main(do_georeference: bool = True):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("drone_processing.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    output_base_dir = Path(OUTPUT_BASE_DIR)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    logger.info("드론 데이터 처리를 시작합니다")
    flight_log = load_flight_log(FLIGHT_LOG_PATH)
    logger.info("비행 로그 데이터를 불러왔습니다")

    video_segments = extract_video_segment(flight_log)
    # 1분 이상의 비디오 세그먼트만 추출 (하드코딩)
    video_segments = [
        segment
        for segment in video_segments
        if segment.iloc[-1]["time(millisecond)"] - segment.iloc[0]["time(millisecond)"]
        > 60000
    ]
    logger.info(f"비디오 세그먼트 {len(video_segments)}개를 추출했습니다")

    tracking_results = [
        load_tracking_json(tracking_path) for tracking_path in TRACKING_PATHS
    ]
    logger.info(f"트래킹 결과 {len(tracking_results)}개를 불러왔습니다")

    if len(tracking_results) != len(video_segments):
        logger.error(
            f"비디오 세그먼트 {len(video_segments)}개와 트래킹 결과 {len(tracking_results)}개의 개수가 일치하지 않습니다"
        )
        raise ValueError("비디오 세그먼트와 트래킹 결과의 개수가 일치하지 않습니다")

    # 각 tracking result를 해당하는 세그먼트와 매칭
    for segment_id, (tracking_result, segment) in enumerate(
        zip(tracking_results, video_segments)
    ):
        fps = tracking_result["video"]["fps"]
        total_frames = tracking_result["video"]["total_frames"]
        logger.info(f"FPS {fps}, 총 {total_frames}프레임의 비디오를 처리 중입니다")

        logger.info(f"세그먼트 {segment_id}를 처리 중입니다")
        interpolated = interpolate_flight_log(segment, fps, total_frames)
        tracking_result = add_flight_log_to_tracking(tracking_result, interpolated)

        if do_georeference:
            logger.info("트래킹 데이터를 지리참조 중입니다")
            tracking_result = georeference_tracking_data(
                tracking_data=tracking_result, fov=DJI_MINI_2_FOV
            )

        logger.info("교차점을 계산 중입니다")
        intersections = calculate_intersections(tracking_result)

        if do_georeference:
            intersections = georeference_intersections(
                intersections=intersections,
                tracking_data=tracking_result,
                fov=DJI_MINI_2_FOV,
            )
            logger.info(f"총 {len(intersections)}개의 교차점을 찾았습니다")
            logger.debug("처음 5개의 교차점:")
            for intersection in intersections[:5]:
                logger.debug(intersection)

        # 최종 결과 저장
        logger.info("최종 결과를 저장 중입니다")
        tracking_path = output_base_dir / f"tracking_{segment_id}.json"
        intersections_path = output_base_dir / f"intersections_{segment_id}.json"

        tracking_path.write_text(json.dumps(tracking_result, indent=2))
        intersections_path.write_text(json.dumps(intersections, indent=2))

    logger.info("모든 처리가 성공적으로 완료되었습니다")


if __name__ == "__main__":
    main(do_georeference=True)
