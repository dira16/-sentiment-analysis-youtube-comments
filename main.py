import os
import sqlite3
import pandas as pd
import requests
import nltk
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================
# --- YouTube API ---
YT_API_KEY = "AIzaSyApXlF4bTuzvroCJFt0eXrVw882WyKEzfg"  # IMPORTANT: Keep your API key private
VIDEO_ID = "IHNzOHi8sJs"
MAX_PAGES = 10  # Max pages to fetch (50 comments per page)

# --- Database ---
DB_FILE = "youtube_data.db"
TABLE_NAME = "comments"

# ==============================================================================
# 2. LOAD ALL MODELS (Done once for efficiency)
# ==============================================================================
print("Loading Hugging Face models... This may take a moment.")

# --- Sentiment Model (CardiffNLP) ---
# We load the tokenizer and model separately to bypass a known bug
sentiment_model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_classifier = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_model = pipeline("sentiment-analysis", model=sentiment_classifier, tokenizer=sentiment_tokenizer)

# --- Other Models ---
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1)
sarcasm_model = pipeline("text-classification", model="helinivan/english-sarcasm-detector")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

print("✅ All models loaded successfully.")

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def fetch_youtube_comments_page(video_id, page_token=None):
    """Fetches a single page of comments from the YouTube API."""
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 50,
        "order": "time",
        "textFormat": "plainText", # Request plain text to avoid HTML tags
        "key": YT_API_KEY
    }
    if page_token:
        params["pageToken"] = page_token
    
    resp = requests.get(url, params=params)
    resp.raise_for_status() # Will raise an error for bad responses (4xx or 5xx)
    return resp.json()

def process_comment(text):
    """
    Runs a single comment through the full analysis pipeline:
    translate -> sentiment -> emotion -> sarcasm.
    """
    # Truncate to 512 tokens for model compatibility
    text_short = text[:512]

    # Default values in case of errors
    results = {
        "processed_text": text,
        "sentiment": "error",
        "emotion": "error",
        "sarcasm": "error"
    }

    # 1. Translation
    try:
        translated = translator(text_short)[0]["translation_text"]
        results["processed_text"] = translated
    except Exception:
        # If translation fails, we analyze the original text but still log it
        pass # processed_text already defaults to the original text

    processed_text = results["processed_text"]

    # 2. Sentiment Analysis
    try:
        results["sentiment"] = sentiment_model(processed_text)[0]["label"]
    except Exception: pass

    # 3. Emotion Detection
    try:
        results["emotion"] = emotion_model(processed_text)[0][0]["label"]
    except Exception: pass

    # 4. Sarcasm Detection
    try:
        results["sarcasm"] = sarcasm_model(processed_text)[0]["label"]
    except Exception: pass
    
    return results

# ==============================================================================
# 4. MAIN PIPELINE
# ==============================================================================
def run_full_pipeline(video_id, max_pages):
    """
    Main function to fetch, process, and save comments to the database.
    """
    processed_comments = []
    page_token = None
    
    with tqdm(total=max_pages, desc="Fetching & Processing", unit="page") as pbar:
        for page_num in range(max_pages):
            try:
                data = fetch_youtube_comments_page(video_id, page_token)
            except requests.exceptions.HTTPError as e:
                print(f"\nAPI Error: {e}. Check your YT_API_KEY or VIDEO_ID.")
                break

            for item in data.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                original_text = snippet.get("textOriginal", "")
                
                if not original_text.strip():
                    continue

                # Run the full analysis on the comment text
                analysis_results = process_comment(original_text)

                processed_comments.append({
                    "comment_id": item["id"],
                    "video_id": video_id,
                    "author": snippet.get("authorDisplayName"),
                    "published_at": pd.to_datetime(snippet.get("publishedAt")),
                    "original_text": original_text,
                    "processed_text": analysis_results["processed_text"],
                    "sentiment": analysis_results["sentiment"],
                    "emotion": analysis_results["emotion"],
                    "sarcasm": analysis_results["sarcasm"],
                })

            page_token = data.get("nextPageToken")
            pbar.update(1)
            if not page_token:
                print("\nReached the last page of comments.")
                break # Exit loop if there are no more pages
    
    if not processed_comments:
        print("No new comments were fetched.")
        return

    # --- Database Operations ---
    print(f"\nConnecting to database: {DB_FILE}")
    conn = sqlite3.connect(DB_FILE)
    
    # Create a DataFrame from the newly fetched comments
    df_new = pd.DataFrame(processed_comments)
    
    # Get existing comment IDs from the database to avoid duplicates
    try:
        df_old = pd.read_sql(f"SELECT comment_id FROM {TABLE_NAME}", conn)
        existing_ids = df_old['comment_id'].tolist()
        df_to_append = df_new[~df_new['comment_id'].isin(existing_ids)]
    except pd.io.sql.DatabaseError:
        # Table doesn't exist yet, so we append all new comments
        df_to_append = df_new

    if df_to_append.empty:
        print("No new unique comments to add to the database.")
    else:
        # Append only the new, unique comments to the SQL table
        df_to_append.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
        print(f"✅ Successfully added {len(df_to_append)} new rows to the '{TABLE_NAME}' table.")

    conn.close()


if __name__ == "__main__":
    run_full_pipeline(video_id=VIDEO_ID, max_pages=MAX_PAGES)