### 입력 데이터

#### 1. 비행 로그 (CSV)
- `time(millisecond)`: 밀리초 단위 타임스탬프
- `datetime(utc)`: UTC 기준 날짜/시간
- `latitude`: 위도
- `longitude`: 경도
- `ascent(feet)`: 고도 (피트)
- `compass_heading(degrees)`: 드론 방향 (도, 북쪽 0도 기준 시계방향)
- `isVideo`: 비디오 녹화 여부 (1: 녹화중, 0: 미녹화)

#### 2. 객체 추적 결과 (JSON)
```json
{
    "video": {
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "total_frames": 3600
    },
    "tracking_results": [
        {
            "i": 0,                    // 프레임 인덱스
            "time(millisecond)": 0,    // 타임스탬프
            "latitude": 37.123456,     // 드론 위도
            "longitude": 127.123456,   // 드론 경도
            "ascent(feet)": 100.0,     // 드론 고도
            "compass_heading(degrees)": 90.0,  // 드론 방향
            "res": [
                {
                    "tid": 1,          // 객체 트래킹 ID
                    "cid": 0,          // 객체 클래스 ID
                    "bbox": [x1, y1, x2, y2]  // 바운딩 박스 좌표
                }
            ]
        }
    ]
}
```

### 출력 데이터

#### 1. 지리참조된 추적 결과 (JSON)
```json
{
    "video": {
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "total_frames": 3600
    },
    "tracking_results": [
        {
            // ...기존 필드 유지...
            "res": [
                {
                    "tid": 1,          // 객체 트래킹 ID
                    "cid": 0,          // 객체 클래스 ID
                    "bbox": [x1, y1, x2, y2],

                    // georeference를 할때만 추가됨 (옵셔널)
                    "latitude": 37.123456,     // 객체 위도
                    "longitude": 127.123456,   // 객체 경도
                    "speed(m/s)": 5.0       // 객체 속도
                }
            ]
        }
    ]
}
```

#### 2. 교차점 데이터 (JSON)
```json
[
    {
        "track_id_1": 1,              // 첫 번째 객체의 트랙 ID
        "class_id_1": 3,              // 첫 번째 객체의 클래스 ID
        "track_id_2": 2,              // 두 번째 객체의 트랙 ID
        "class_id_1": 5,              // 두 번째 객체의 클래스 ID
        "intersection_x": 960,         // 교차점 X 좌표 (픽셀)
        "intersection_y": 540,         // 교차점 Y 좌표 (픽셀)
        "frame_at_intersection_1": 30, // 첫 번째 객체의 교차 프레임
        "frame_at_intersection_2": 32, // 두 번째 객체의 교차 프레임

        // georeference를 할때만 추가됨 (옵셔널)
        "latitude": 37.123456,        // 교차점 위도
        "longitude": 127.123456,       // 교차점 경도

        // TODO
        "speed_at_intersection_1": 123, // 첫 번째 객체가 교차할때 속도
        "speed_at_intersection_2": 123, // 두 번째 객체가 교차할때 속도

        "angle_at_intersection": 60     // 교차 세그먼트의 각도 차이
    }
]
```