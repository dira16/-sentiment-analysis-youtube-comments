import os, datetime as dt, pandas as pd
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm   # progress bar

# Setup
YT_API_KEY = "AIzaSyApXlF4bTuzvroCJFt0eXrVw882WyKEzfg"
VIDEO_ID = "IHNzOHi8sJs"
CSV_FILE = "yt_sentiment.csv"
MAX_PAGES = 10   # limit to 100 pages (≈ 5000 comments)

# Download VADER if not already
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
vader = SentimentIntensityAnalyzer()

def classify(text):
    scores = vader.polarity_scores(text)
    comp = scores["compound"]
    if comp >= 0.05:
        return "positive", comp
    elif comp <= -0.05:
        return "negative", comp
    return "neutral", comp

def fetch_comments(video_id, page_token=None):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 50,
        "order": "time",
        "key": YT_API_KEY
    }
    if page_token:
        params["pageToken"] = page_token
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

def pipeline(video_id, max_pages=10):
    rows = []
    page = None
    page_count = 0
    with tqdm(total=max_pages, desc="Fetching comments", unit="page") as pbar:
        while page_count < max_pages:
            data = fetch_comments(video_id, page)
            for item in data.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                cid = item["id"]
                text = top.get("textDisplay", "")
                sentiment, score = classify(text)
                rows.append({
                    "video_id": video_id,
                    "comment_id": cid,
                    "author": top.get("authorDisplayName"),
                    "text": text,
                    "sentiment": sentiment,
                    "score": score,
                    "publishedAt": top.get("publishedAt")
                })
            page = data.get("nextPageToken")
            page_count += 1
            pbar.update(1)
            if not page:
                break
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = pipeline(VIDEO_ID, MAX_PAGES)
    if os.path.exists(CSV_FILE):
        old = pd.read_csv(CSV_FILE)
        df = pd.concat([old, df]).drop_duplicates("comment_id")
    df.to_csv(CSV_FILE, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows to {CSV_FILE}")
