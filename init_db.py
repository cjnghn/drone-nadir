import sqlite3


def init_db(db_name="video_data.db"):
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()

        # 1. Videos 테이블 생성
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS Videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,    -- 비디오 파일명
            filepath TEXT UNIQUE,    -- 비디오 파일 경로
            fps INTEGER,             -- 초당 프레임 수
            resolution TEXT          -- 해상도 (예: 1920x1080)
        )
        """
        )

        # 2. TrackingData 테이블 생성
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS TrackingData (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,        -- 해당 비디오 ID
            track_id INTEGER,        -- 트래킹 ID
            class_id INTEGER,        -- 클래스 ID
            bbox TEXT,               -- 바운딩 박스 정보 (JSON)
            latitude REAL,           -- 위도
            longitude REAL,          -- 경도
            speed REAL,              -- 속도 (m/s)
            FOREIGN KEY (video_id) REFERENCES Videos (id)
        )
        """
        )

        # 3. IntersectionData 테이블 생성
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS IntersectionData (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,        -- 해당 비디오 ID
            track_id_1 INTEGER,      -- 첫 번째 객체의 트랙 ID
            class_id_1 INTEGER,      -- 첫 번째 객체의 클래스 ID
            track_id_2 INTEGER,      -- 두 번째 객체의 트랙 ID
            class_id_2 INTEGER,      -- 두 번째 객체의 클래스 ID
            intersection_x INTEGER,  -- 교차점 X 좌표 (픽셀)
            intersection_y INTEGER,  -- 교차점 Y 좌표 (픽셀)
            frame_1 INTEGER,         -- 첫 번째 객체의 교차 프레임
            frame_2 INTEGER,         -- 두 번째 객체의 교차 프레임
            time_difference REAL,    -- 두 객체의 교차 시간 차이 (초)
            latitude REAL,           -- 교차점 위도
            longitude REAL,          -- 교차점 경도
            speed_1 REAL,            -- 첫 번째 객체의 교차 시 속도
            speed_2 REAL,            -- 두 번째 객체의 교차 시 속도
            angle REAL,              -- 교차 각도
            FOREIGN KEY (video_id) REFERENCES Videos (id)
        )
        """
        )

        # 인덱스 추가 (옵션, 성능 향상)
        cursor.execute(
            """
        CREATE INDEX IF NOT EXISTS idx_tracking_lat_lng 
        ON TrackingData (latitude, longitude)
        """
        )

        cursor.execute(
            """
        CREATE INDEX IF NOT EXISTS idx_intersection_lat_lng 
        ON IntersectionData (latitude, longitude)
        """
        )

        conn.commit()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully with updated schema!")
