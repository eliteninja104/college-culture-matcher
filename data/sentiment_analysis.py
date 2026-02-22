"""
Sentiment analysis pipeline for College Culture Matcher.
Processes scraped Reddit text into per-school culture dimension scores.

Usage:
    python data/sentiment_analysis.py

Reads:  data/reddit_raw.csv, data/niche_grades.csv, data/rmp_data.csv
Writes: data/school_profiles.csv, data/representative_quotes.csv
"""

import os
import re
import csv
import json
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Topic categories and their keyword lists
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "social": [
        "greek", "frat", "sorority", "party", "tailgate", "bar", "nightlife",
        "social", "gameday", "football saturday", "drinking", "club", "rush",
    ],
    "academics": [
        "professor", "class", "homework", "exam", "study", "gpa", "library",
        "lecture", "difficult", "easy", "workload", "course", "major", "grad",
        "research", "ta", "teaching",
    ],
    "campus": [
        "campus", "dorm", "dining", "food", "gym", "rec center", "building",
        "beautiful", "ugly", "parking", "facility", "housing", "cafeteria",
    ],
    "diversity": [
        "diverse", "diversity", "inclusive", "international", "minority",
        "welcoming", "culture", "community", "race", "lgbtq", "inclusion",
    ],
    "career": [
        "career", "internship", "job", "recruit", "salary", "alumni",
        "networking", "placement", "hire", "employer", "interview", "offer",
    ],
    "location": [
        "town", "city", "weather", "downtown", "off-campus", "transport",
        "safe", "walkable", "boring", "fun", "area", "neighborhood", "location",
    ],
    "religion": [
        "religious", "church", "conservative", "liberal", "spiritual", "faith",
        "values", "christian", "jewish", "muslim", "atheist", "bible",
    ],
    "spirit": [
        "football", "basketball", "gameday", "rivalry", "stadium", "cheer",
        "spirit", "sec", "tailgate", "march madness", "bowl game", "playoff",
    ],
}

# Letter grade to numeric mapping
GRADE_MAP: dict[str, float] = {
    "A+": 4.3, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "D-": 0.7,
    "F": 0.0,
}


def text_matches_topic(text: str, keywords: list[str]) -> bool:
    """Check if text contains at least one keyword from the list."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def run_sentiment_analysis(reddit_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Compute per-school, per-topic sentiment scores and volumes.
    Also collect representative quotes for each school/topic.

    Returns:
        - DataFrame with columns: school, <topic>_sentiment, <topic>_volume for each topic
        - List of dicts with representative quotes
    """
    analyzer = SentimentIntensityAnalyzer()
    schools = reddit_df["school"].unique()

    results = []
    all_quotes = []

    for school in schools:
        school_df = reddit_df[reddit_df["school"] == school]
        row = {"school": school}

        for topic, keywords in TOPIC_KEYWORDS.items():
            # Filter texts matching this topic
            mask = school_df["text"].apply(lambda t: text_matches_topic(str(t), keywords))
            matching = school_df[mask]

            if len(matching) == 0:
                row[f"{topic}_sentiment"] = 0.5  # neutral default
                row[f"{topic}_volume"] = 0
                continue

            # Run VADER on each matching snippet
            sentiments = matching["text"].apply(
                lambda t: analyzer.polarity_scores(str(t))["compound"]
            )
            row[f"{topic}_sentiment"] = sentiments.mean()
            row[f"{topic}_volume"] = len(matching)

            # Collect representative quotes (top 3 by upvotes for each sentiment direction)
            if len(matching) > 0:
                matching_with_sent = matching.copy()
                matching_with_sent["_sent"] = sentiments.values

                # Top positive quotes
                pos = matching_with_sent[matching_with_sent["_sent"] > 0.2].nlargest(
                    3, "score"
                )
                for _, q in pos.iterrows():
                    all_quotes.append({
                        "school": school,
                        "topic": topic,
                        "text": str(q["text"])[:500],  # truncate long texts
                        "sentiment": "positive",
                        "score": int(q["score"]),
                    })

                # Top negative quotes
                neg = matching_with_sent[matching_with_sent["_sent"] < -0.2].nlargest(
                    3, "score"
                )
                for _, q in neg.iterrows():
                    all_quotes.append({
                        "school": school,
                        "topic": topic,
                        "text": str(q["text"])[:500],
                        "sentiment": "negative",
                        "score": int(q["score"]),
                    })

        results.append(row)

    return pd.DataFrame(results), all_quotes


def normalize_sentiments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize sentiment scores from [-1, 1] to [0, 1] range.
    Normalize volume counts via min-max scaling to [0, 1].
    """
    for topic in TOPIC_KEYWORDS:
        sent_col = f"{topic}_sentiment"
        vol_col = f"{topic}_volume"

        # Sentiment: shift from [-1,1] to [0,1]
        if sent_col in df.columns:
            df[sent_col] = (df[sent_col] + 1) / 2

        # Volume: min-max scale to [0,1]
        if vol_col in df.columns:
            vmin = df[vol_col].min()
            vmax = df[vol_col].max()
            if vmax > vmin:
                df[vol_col] = (df[vol_col] - vmin) / (vmax - vmin)
            else:
                df[vol_col] = 0.5

    return df


def load_niche_grades(path: str) -> pd.DataFrame:
    """Load Niche grades CSV and convert letter grades to numeric."""
    niche = pd.read_csv(path)
    grade_cols = [c for c in niche.columns if c != "school"]
    for col in grade_cols:
        niche[col] = niche[col].map(GRADE_MAP)
    # Normalize to 0-1 range (max is 4.3)
    for col in grade_cols:
        niche[col] = niche[col] / 4.3
    return niche


def load_rmp_data(path: str) -> pd.DataFrame:
    """Load RMP data and normalize."""
    if not os.path.exists(path):
        return pd.DataFrame()
    rmp = pd.read_csv(path)
    # Normalize avg_rating (1-5 scale) to 0-1
    if "avg_rating" in rmp.columns:
        rmp["avg_rating"] = rmp["avg_rating"].fillna(3.0)
        rmp["avg_rating"] = (rmp["avg_rating"] - 1) / 4
    return rmp


def main():
    # Load Reddit data
    reddit_path = os.path.join(DATA_DIR, "reddit_raw.csv")
    if not os.path.exists(reddit_path):
        print(f"Reddit data not found at {reddit_path}.")
        print("Run scrape_reddit.py first, or the app will use Niche grades only.")
        reddit_df = pd.DataFrame(columns=["school", "text", "source_type", "score"])
    else:
        reddit_df = pd.read_csv(reddit_path)
        print(f"Loaded {len(reddit_df)} Reddit text snippets.")

    # Run sentiment analysis
    if len(reddit_df) > 0:
        sentiment_df, quotes = run_sentiment_analysis(reddit_df)
        sentiment_df = normalize_sentiments(sentiment_df)
    else:
        sentiment_df = pd.DataFrame(columns=["school"])
        quotes = []

    # Load Niche grades
    niche_path = os.path.join(DATA_DIR, "niche_grades.csv")
    niche_df = load_niche_grades(niche_path)
    print(f"Loaded Niche grades for {len(niche_df)} schools.")

    # Load RMP data
    rmp_path = os.path.join(DATA_DIR, "rmp_data.csv")
    rmp_df = load_rmp_data(rmp_path)
    if len(rmp_df) > 0:
        print(f"Loaded RMP data for {len(rmp_df)} schools.")

    # Merge all data on school name
    if len(sentiment_df) > 0:
        profiles = niche_df.merge(sentiment_df, on="school", how="left")
    else:
        profiles = niche_df.copy()

    if len(rmp_df) > 0:
        profiles = profiles.merge(rmp_df[["school", "avg_rating"]], on="school", how="left")

    # Fill NaN sentiment columns with neutral (0.5)
    for topic in TOPIC_KEYWORDS:
        for suffix in ["_sentiment", "_volume"]:
            col = f"{topic}{suffix}"
            if col in profiles.columns:
                profiles[col] = profiles[col].fillna(0.5)
            else:
                profiles[col] = 0.5

    # Save profiles
    output_path = os.path.join(DATA_DIR, "school_profiles.csv")
    profiles.to_csv(output_path, index=False)
    print(f"Saved school profiles to {output_path} ({len(profiles)} schools)")

    # Save representative quotes
    if quotes:
        quotes_path = os.path.join(DATA_DIR, "representative_quotes.csv")
        pd.DataFrame(quotes).to_csv(quotes_path, index=False)
        print(f"Saved {len(quotes)} representative quotes to {quotes_path}")


if __name__ == "__main__":
    main()
