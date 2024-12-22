import sqlite3
import json
import os
from tqdm import tqdm

DB_NAME = 'video_data.db'


def insert_video_from_metadata(tracking_json, video_filepath, db_name=DB_NAME):
    """JSON에서 비디오 메타데이터 추출 후 Videos 테이블에 삽입"""
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        with open(tracking_json, 'r') as f:
            data = json.load(f)

            # 비디오 메타데이터 추출
            fps = data['video']['fps']
            resolution = f"{data['video']['width']}x{data['video']['height']}"
            total_frames = data['video']['total_frames']

            # 비디오 파일명 추출
            filename = os.path.basename(video_filepath)

            # 데이터베이스에 삽입
            cursor.execute('''
            INSERT OR IGNORE INTO Videos (filename, filepath, fps, resolution)
            VALUES (?, ?, ?, ?)
            ''', (filename, video_filepath, fps, resolution))
            conn.commit()

            # 삽입된 비디오 ID 반환
            cursor.execute('SELECT id FROM Videos WHERE filename = ?', (filename,))
            video_id = cursor.fetchone()[0]
            return video_id


def insert_tracking_data(tracking_json, video_id, db_name=DB_NAME):
    """TrackingData 테이블에 데이터 삽입"""
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        with open(tracking_json, 'r') as f:
            data = json.load(f)
            total_items = sum(len(result['res']) for result in data['tracking_results'])
            
            with tqdm(total=total_items, desc="Inserting tracking data") as pbar:
                for result in data['tracking_results']:
                    for res in result['res']:
                        try:
                            cursor.execute('''
                            INSERT INTO TrackingData (video_id, track_id, class_id, bbox, latitude, longitude, speed)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                video_id, res['tid'], res['cid'],
                                json.dumps(res['bbox']),
                                res.get('latitude'), res.get('longitude'), res.get('speed(m/s)')
                            ))
                            pbar.update(1)
                        except Exception as e:
                            print(f"Error inserting tracking data: {e}")
        conn.commit()


def insert_intersection_data(intersection_json, video_id, db_name=DB_NAME):
    """IntersectionData 테이블에 데이터 삽입"""
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        with open(intersection_json, 'r') as f:
            data = json.load(f)
            for inter in tqdm(data, desc="Inserting intersection data"):
                try:
                    cursor.execute('''
                    INSERT INTO IntersectionData (
                        video_id, track_id_1, class_id_1, track_id_2, class_id_2, 
                        intersection_x, intersection_y, frame_1, frame_2, 
                        latitude, longitude, speed_1, speed_2, angle
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        video_id, inter['track_id_1'], inter['class_id_1'],
                        inter['track_id_2'], inter['class_id_2'],
                        inter['intersection_x'], inter['intersection_y'],
                        inter['frame_at_intersection_1'], inter['frame_at_intersection_2'],
                        inter.get('latitude'), inter.get('longitude'),
                        inter.get('speed_at_intersection_1'), inter.get('speed_at_intersection_2'),
                        inter.get('angle_at_intersection')
                    ))
                except Exception as e:
                    print(f"Error inserting intersection data: {e}")
        conn.commit()


def main():
    tracking_files = [
        'outputs/DJI_0279,DJI_0280/tracking_0.json',
        'outputs/DJI_0279,DJI_0280/tracking_1.json'
    ]
    intersection_files = [
        'outputs/DJI_0279,DJI_0280/intersections_0.json',
        'outputs/DJI_0279,DJI_0280/intersections_1.json'
    ]
    video_files = [
        'inputs/DJI_0279,DJI_0280/DJI_0279.MP4',
        'inputs/DJI_0279,DJI_0280/DJI_0280.MP4'
    ]  

    print("\nProcessing tracking and intersection data...")

    for i, tracking_file in enumerate(tqdm(tracking_files, desc="Processing videos")):
        print(f"Processing {tracking_file}...")

        # 비디오 메타데이터 삽입 (비디오 경로 전달)
        video_id = insert_video_from_metadata(tracking_file, video_files[i])

        # 트래킹 데이터 삽입
        insert_tracking_data(tracking_file, video_id)

        # 교차점 데이터 삽입
        insert_intersection_data(intersection_files[i], video_id)

    print("Data insertion complete!")


if __name__ == '__main__':
    main()
