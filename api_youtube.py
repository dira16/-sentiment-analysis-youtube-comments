from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# Load dataset (CSV/SQL etc.)
df = pd.read_csv("yt_processed.csv")

@app.route('/comments', methods=['GET'])
def get_comments():
    return jsonify(df.to_dict(orient="records"))

@app.route('/comments/<video_id>', methods=['GET'])
def get_comments_by_video(video_id):
    filtered = df[df["video_id"] == video_id]
    return jsonify(filtered.to_dict(orient="records"))

if __name__ == "__main__":
    # Runs on localhost:5000
    app.run(debug=True)
