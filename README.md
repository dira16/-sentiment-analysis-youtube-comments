# YouTube Sentiment Analysis Dashboard


## Overview

This project performs comprehensive sentiment analysis on YouTube comments using API data to provide actionable insights into audience engagement and content performance. The system integrates data ingestion, preprocessing, NLP-based sentiment classification, and real-time visualization through Power BI dashboards.

## Features

- **YouTube API Integration**: Automated data collection from YouTube comments
- **Data Preprocessing**: Comprehensive text cleaning and normalization
- **Sentiment Analysis**: Advanced NLP classification for audience sentiment detection
- **Real-time Visualization**: Interactive Power BI dashboards for sentiment trends
- **Content Insights**: Data-driven recommendations for content strategy optimization

## Architecture

```
YouTube API → Data Ingestion → Preprocessing → Sentiment Analysis → Power BI Dashboard
```

## Technologies Used

- **Data Collection**: YouTube Data API v3
- **Data Processing**: Python, Pandas, NumPy
- **NLP & Sentiment Analysis**: NLTK, TextBlob, or Transformers (specify your choice)
- **Visualization**: Microsoft Power BI
- **Data Storage**: [Specify your database/storage solution]

## Installation

### Prerequisites

- Python 3.8+
- YouTube Data API v3 key
- Power BI Desktop
- Required Python packages (see requirements.txt)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/youtube-sentiment-analysis.git
cd youtube-sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API credentials:
```bash
# Create a .env file and add your YouTube API key
YOUTUBE_API_KEY=your_api_key_here
```

4. Run the application:
```bash
python main.py
```

## Usage

1. **Data Collection**: Configure target YouTube channels or videos
2. **Processing**: Run the sentiment analysis pipeline
3. **Visualization**: Open the Power BI dashboard file (.pbix)
4. **Analysis**: Monitor real-time sentiment trends and insights

## Project Structure

```
├── data/
│   ├── raw/                 # Raw YouTube comment data
│   ├── processed/           # Cleaned and preprocessed data
│   └── results/             # Sentiment analysis results
├── src/
│   ├── data_ingestion.py    # YouTube API data collection
│   ├── preprocessing.py     # Text cleaning and normalization
│   ├── sentiment_model.py   # NLP sentiment classification
│   └── utils.py            # Helper functions
├── dashboards/
│   └── sentiment_dashboard.pbix  # Power BI dashboard
├── assets/
│   └── Screenshot 2025-08-29 210548.png
├── requirements.txt
├── README.md
└── main.py
```

## Key Metrics Tracked

- **Sentiment Distribution**: Positive, negative, and neutral comment ratios
- **Trend Analysis**: Sentiment changes over time
- **Engagement Correlation**: Relationship between sentiment and engagement metrics
- **Content Performance**: Sentiment-based content evaluation scores

## Dashboard Features

- Real-time sentiment monitoring
- Interactive filtering by date, channel, and video
- Sentiment trend visualization
- Top positive/negative comment identification
- Engagement metrics correlation analysis

![Project Screenshot](assets/Screenshot%202025-08-29%20210548.png)

## Results & Insights

The dashboard enables content creators and marketers to:
- Identify audience sentiment patterns
- Optimize content strategy based on feedback analysis
- Monitor brand perception in real-time
- Make data-driven decisions for content improvement

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Contact

- **Author**: [Divyanshi Rathore]
- **Email**: [dira160803@gmail.com]

## Acknowledgments

- YouTube Data API v3 for data access
- Open-source NLP libraries for sentiment analysis capabilities
- Power BI for visualization platform

---

⭐ Star this repository if you found it helpful!
