import pandas as pd
import numpy as np


def load_flight_log(file_path: str) -> pd.DataFrame:
    """
    비행 로그 CSV 파일을 로드하여 DataFrame으로 반환합니다.

    Args:
        file_path (str): CSV 파일의 경로

    Returns:
        pd.DataFrame: 비행 로그 데이터가 포함된 DataFrame
    """
    columns = [
        "time(millisecond)",
        "datetime(utc)",
        "latitude",
        "longitude",
        "ascent(feet)",
        "compass_heading(degrees)",
        "isVideo",
    ]

    # CSV 파일을 먼저 읽고 컬럼명의 공백을 제거
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # 필요한 컬럼만 선택
    return df[columns]


def extract_video_segment(log: pd.DataFrame) -> list:
    """
    비행 로그에서 비디오 녹화 구간을 추출합니다.

    Args:
        log (pd.DataFrame): 비행 로그 데이터

    Returns:
        list: 비디오 녹화 구간별로 분리된 DataFrame 리스트
    """
    log["group"] = (log["isVideo"] != log["isVideo"].shift()).cumsum()
    return [group for _, group in log[log["isVideo"] == 1].groupby("group")]


def interpolate_flight_log(
    log: pd.DataFrame, fps: float, total_frames: int = None
) -> pd.DataFrame:
    """
    비행 로그 데이터를 비디오 프레임에 맞게 보간합니다.

    Args:
        log (pd.DataFrame): 비행 로그 데이터
        fps (float): 비디오의 초당 프레임 수
        total_frames (int, optional): 총 프레임 수. 기본값은 None

    Returns:
        pd.DataFrame: 프레임별로 보간된 비행 데이터
    """
    start_time = log["time(millisecond)"].iloc[0]
    end_time = log["time(millisecond)"].iloc[-1]

    if total_frames is None:
        total_frames = int((end_time - start_time) / (1000 / fps)) + 1

    time = np.linspace(start_time, end_time, total_frames)
    lat = np.interp(time, log["time(millisecond)"], log["latitude"])
    lon = np.interp(time, log["time(millisecond)"], log["longitude"])
    alt = np.interp(time, log["time(millisecond)"], log["ascent(feet)"])
    heading = np.interp(time, log["time(millisecond)"], log["compass_heading(degrees)"])

    return pd.DataFrame(
        {
            "time(millisecond)": time,
            "latitude": lat,
            "longitude": lon,
            "ascent(feet)": alt,
            "compass_heading(degrees)": heading,
        }
    )


if __name__ == "__main__":
    log = load_flight_log("inputs/_logs/Dec-16th-2024-10-50AM-Flight-Airdata.csv")
    video_segments = extract_video_segment(log)
    print(f"Total {len(video_segments)} video segments are found.")

    for i, segment in enumerate(video_segments):
        # 영상 길이 보기
        row_interval = (
            segment["time(millisecond)"].iloc[-1] - segment["time(millisecond)"].iloc[0]
        )
        print(f"Video segment {i + 1}: {row_interval / 1000:.2f} sec")
