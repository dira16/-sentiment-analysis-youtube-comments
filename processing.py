import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ==============================================================================
# MODIFIED SECTION: Load Models Manually to Avoid Tokenizer Bug
# ==============================================================================

# --- Sentiment Model ---
# We load the tokenizer and model separately to bypass an internal bug in the 
# pipeline's auto-loading mechanism for this specific model.
sentiment_model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_classifier = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

sentiment_model = pipeline(
    "sentiment-analysis",
    model=sentiment_classifier,
    tokenizer=sentiment_tokenizer
)

# --- Other Models (loaded normally as they don't have the issue) ---
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1)
sarcasm_model = pipeline("text-classification", model="helinivan/english-sarcasm-detector")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

print("✅ All models loaded successfully.")

# ==============================================================================
# Data Processing (No changes needed here)
# ==============================================================================

# Read YouTube comments CSV
try:
    df = pd.read_csv("yt_sentiment.csv")
except FileNotFoundError:
    print("Error: 'yt_sentiment.csv' not found. Please make sure the file is in the same directory.")
    exit()


# Create new columns if they don't exist
df["translated"] = ""
df["sentiment_model"] = ""
df["emotion"] = ""
df["sarcasm"] = ""

# Process each row in the DataFrame
for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing Comments"):
    text = str(row["text"])
    
    # Ensure text is not empty
    if not text.strip():
        df.at[i, "translated"] = ""
        df.at[i, "sentiment_model"] = "empty"
        df.at[i, "emotion"] = "empty"
        df.at[i, "sarcasm"] = "empty"
        continue

    # 1. Translation
    try:
        # Truncate long text before sending to translator
        translated = translator(text[:512])[0]["translation_text"]
    except Exception as e:
        # If translation fails, use the original text
        translated = text
    df.at[i, "translated"] = translated
    
    # Process the translated text (or original if translation failed)
    processed_text = translated[:512]

    # 2. Sentiment Analysis
    try:
        df.at[i, "sentiment_model"] = sentiment_model(processed_text)[0]["label"]
    except Exception as e:
        df.at[i, "sentiment_model"] = f"error: {e}"
    
    # 3. Emotion Detection
    try:
        df.at[i, "emotion"] = emotion_model(processed_text)[0][0]["label"]
    except Exception as e:
        df.at[i, "emotion"] = f"error: {e}"
    
    # 4. Sarcasm Detection
    try:
        df.at[i, "sarcasm"] = sarcasm_model(processed_text)[0]["label"]
    except Exception as e:
        df.at[i, "sarcasm"] = f"error: {e}"

# Save results to a new CSV file
df.to_csv("yt_processed.csv", index=False, encoding="utf-8")
print("\n✅ Processing complete. Saved to yt_processed.csv")
