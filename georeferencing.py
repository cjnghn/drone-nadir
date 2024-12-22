import math


def pixel_to_latlon(
    px,
    py,
    drone_lat,
    drone_lon,
    drone_alt,
    drone_heading,
    image_width,
    image_height,
    fov,  # 대각선 화각 (degrees)
):
    """
    단일 대각선 fov를 활용한 이미지 픽셀 -> 위도/경도 변환 함수

    매개변수:
        px (float): 이미지 상의 X 픽셀 좌표 (왼→오른쪽 증가)
        py (float): 이미지 상의 Y 픽셀 좌표 (위→아래쪽 증가)
        drone_lat (float): 드론의 위도 (WGS84)
        drone_lon (float): 드론의 경도 (WGS84)
        drone_alt (float): 드론의 고도 (m)
        drone_heading (float): 드론의 진행 방향 (북쪽 기준, 시계방향 각도)
        image_width (int): 이미지 너비 (픽셀)
        image_height (int): 이미지 높이 (픽셀)
        fov (float): 대각선 화각 (degrees)

    반환값:
        (lat, lon): 변환된 위도/경도 좌표 (WGS84)
    """

    # 이미지 중심 픽셀
    cx = image_width / 2.0
    cy = image_height / 2.0

    # 대각선 fov를 사용해 지면 상 대각선 거리 계산
    diagonal_fov_rad = math.radians(fov)
    diagonal_ground = 2 * drone_alt * math.tan(diagonal_fov_rad / 2.0)

    # 이미지 비율을 바탕으로 ground_width, ground_height 계산
    # aspect = w/h
    aspect = image_width / image_height
    # 대각선^2 = width^2 + height^2
    # width = aspect * height
    # diagonal_ground^2 = (aspect^2 + 1)*height^2
    # height = diagonal_ground / sqrt(aspect^2 + 1)
    ground_height = diagonal_ground / math.sqrt(aspect**2 + 1)
    ground_width = aspect * ground_height

    # 한 픽셀당 실제 지상 거리
    meter_per_pixel_x = ground_width / image_width
    meter_per_pixel_y = ground_height / image_height

    # 이미지 중심 기준으로 픽셀 좌표 → 지상 거리 변환
    dx = (px - cx) * meter_per_pixel_x  # 동/서 방향
    dy = (py - cy) * meter_per_pixel_y  # 남/북 방향 (남양수)

    # 북쪽 좌표계로 변환 (dy는 남양수이므로 북양수로 만들기 위해 음수화)
    dy_north = -dy

    # 드론 헤딩 적용
    adjusted_heading = (-drone_heading) % 360
    theta = math.radians(adjusted_heading)

    # 회전 변환 적용
    dx_rot = dx * math.cos(theta) - dy_north * math.sin(theta)
    dy_rot = dx * math.sin(theta) + dy_north * math.cos(theta)

    # 위도/경도 변환
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(drone_lat))

    dlat = dy_rot / meters_per_deg_lat
    dlon = dx_rot / meters_per_deg_lon

    lat = drone_lat + dlat
    lon = drone_lon + dlon

    return lat, lon


def haversine(lat1, lon1, lat2, lon2):
    """
    지구 표면상의 두 지점 간 대원 거리를 계산합니다.
    """
    R = 6371000  # Radius of the Earth in meters

    # 입력값 검증
    if not all(isinstance(x, (int, float)) for x in [lat1, lon1, lat2, lon2]):
        raise ValueError("Coordinates must be numeric values")

    # Convert latitude and longitude from degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula with improved numerical stability
    sin_phi = math.sin(delta_phi / 2.0)
    sin_lambda = math.sin(delta_lambda / 2.0)
    a = sin_phi * sin_phi + math.cos(phi1) * math.cos(phi2) * sin_lambda * sin_lambda
    c = 2 * math.atan2(math.sqrt(min(1.0, a)), math.sqrt(max(0.0, 1.0 - a)))

    return R * c


def calculate_moving_average_speed(
    frames, current_frame_index, track_id, window_size=5
):
    """
    여러 프레임에 걸친 이동 평균 속도를 계산합니다.
    """
    try:
        if current_frame_index < 1 or not frames:
            return 0.0

        positions = []
        times = []

        for i in range(
            current_frame_index, max(-1, current_frame_index - window_size), -1
        ):
            if i < 0 or i >= len(frames):
                continue

            frame = frames[i]
            obj = next((o for o in frame["res"] if o["tid"] == track_id), None)

            if obj and "latitude" in obj and "longitude" in obj:
                if isinstance(obj["latitude"], (int, float)) and isinstance(
                    obj["longitude"], (int, float)
                ):
                    positions.append((obj["latitude"], obj["longitude"]))
                    times.append(frame["time(millisecond)"])

        if len(positions) < 2:
            return 0.0

        total_distance = sum(
            haversine(
                positions[i][0],
                positions[i][1],
                positions[i + 1][0],
                positions[i + 1][1],
            )
            for i in range(len(positions) - 1)
        )

        time_diff = (times[0] - times[-1]) / 1000.0
        return total_distance / time_diff if time_diff > 0 else 0.0

    except Exception as e:
        print(f"Error calculating speed: {str(e)}")
        return 0.0


def georeference_tracking_data(tracking_data: dict, fov: float) -> dict:
    """
    객체 추적 데이터의 바운딩 박스 중심점을 지리참조하고 실제 속도를 계산합니다.

    매개변수:
        tracking_data (dict): 객체 추적 결과와 비디오 메타데이터를 포함하는 딕셔너리
            {
                'tracking_results': list[dict], # 프레임별 추적 결과
                'video': dict                   # 비디오 정보
            }
        fov (float): 카메라 화각 정보

    반환값:
        dict: 지리참조된 추적 데이터 및 계산된 속도 정보가 포함된 딕셔너리
    """
    if (
        not tracking_data
        or "tracking_results" not in tracking_data
        or "video" not in tracking_data
    ):
        raise ValueError("Invalid tracking data format")

    frames = tracking_data["tracking_results"]
    video = tracking_data["video"]
    image_width, image_height = video["width"], video["height"]

    for frame_index, frame in enumerate(frames):
        try:
            drone_lat, drone_lon = frame["latitude"], frame["longitude"]
            drone_alt = frame["ascent(feet)"] * 0.3048
            drone_heading = frame["compass_heading(degrees)"]

            for obj in frame["res"]:
                minX, minY, maxX, maxY = obj["bbox"]
                centerX, centerY = (minX + maxX) / 2, (minY + maxY) / 2

                obj_lat, obj_lon = pixel_to_latlon(
                    centerX,
                    centerY,
                    drone_lat,
                    drone_lon,
                    drone_alt,
                    drone_heading,
                    image_width,
                    image_height,
                    fov,
                )
                obj["latitude"], obj["longitude"] = obj_lat, obj_lon

                # 이동 평균 속도 계산으로 통합
                speed = calculate_moving_average_speed(frames, frame_index, obj["tid"])
                obj["speed(m/s)"] = round(speed, 2)  # 소수점 둘째자리까지 반올림

        except Exception as e:
            print(f"Error processing frame {frame_index}: {str(e)}")
            continue

    return tracking_data


def georeference_intersections(
    intersections: list,
    tracking_data: dict,
    fov: float,
) -> list:
    """
    교차점의 픽셀 좌표를 위도/경도 좌표로 변환합니다.

    매개변수:
        intersections (list): 픽셀 좌표로 표현된 교차점 목록
        tracking_data (dict): 비디오 메타데이터와 드론 정보를 포함하는 추적 데이터
            {
                'tracking_results': list[dict], # 프레임별 추적 결과
                'video': dict                   # 비디오 정보
            }
        fov (float): 카메라 화각 정보

    반환값:
        list: 지리참조된 좌표가 추가된 교차점 목록
    """
    video = tracking_data["video"]
    image_width, image_height = video["width"], video["height"]

    # Iterate over intersections and georeference their pixel coordinates
    georeferenced_intersections = []
    for intersection in intersections:
        frame_index_1 = intersection["frame_at_intersection_1"]
        frame_index_2 = intersection["frame_at_intersection_2"]

        # Use the first track's frame as reference for drone data
        reference_frame = tracking_data["tracking_results"][frame_index_1]
        drone_lat = reference_frame["latitude"]
        drone_lon = reference_frame["longitude"]
        drone_alt = reference_frame["ascent(feet)"] * 0.3048  # Convert feet to meters
        drone_heading = reference_frame["compass_heading(degrees)"]

        # Convert pixel coordinates to georeferenced coordinates
        obj_lat, obj_lon = pixel_to_latlon(
            px=intersection["intersection_x"],
            py=intersection["intersection_y"],
            drone_lat=drone_lat,
            drone_lon=drone_lon,
            drone_alt=drone_alt,
            drone_heading=drone_heading,
            image_width=image_width,
            image_height=image_height,
            fov=fov,
        )

        # Append georeferenced intersection data
        georeferenced_intersections.append(
            {
                **intersection,
                "latitude": obj_lat,
                "longitude": obj_lon,
            }
        )

    return georeferenced_intersections
