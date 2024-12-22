from flask import Flask, jsonify, request, render_template
import sqlite3

app = Flask(__name__)
DB_NAME = 'video_data.db'


def query_db(query, args=(), one=False):
    """Helper function to query the database."""
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, args)
        rv = cursor.fetchall()
        return (rv[0] if rv else None) if one else rv


@app.route('/api/videos', methods=['GET'])
def get_videos():
    """Fetch all video metadata."""
    query = "SELECT * FROM Videos"
    videos = query_db(query)
    return jsonify([dict(video) for video in videos])


@app.route('/api/tracking/<int:video_id>', methods=['GET'])
def get_tracking_data(video_id):
    """Fetch tracking data for a specific video."""
    query = '''
    SELECT * FROM TrackingData
    WHERE video_id = ?
    '''
    tracking_data = query_db(query, [video_id])
    return jsonify([dict(track) for track in tracking_data])


@app.route('/api/intersections/<int:video_id>', methods=['GET'])
def get_intersections(video_id):
    """Fetch intersection data for a specific video."""
    query = '''
    SELECT * FROM IntersectionData
    WHERE video_id = ?
    '''
    intersections = query_db(query, [video_id])
    return jsonify([dict(intersection) for intersection in intersections])


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
