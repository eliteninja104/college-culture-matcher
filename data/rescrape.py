"""
Re-scrape schools that have low data counts or wrong subreddits.
Fetches more pages and also pulls from 'top' (all time) in addition to 'hot'.

Usage:
    python data/rescrape.py
"""

import os
import time
import csv
import json
import random
import urllib.request
import urllib.error
import urllib.parse
import pandas as pd

DATA_DIR = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(DATA_DIR, "reddit_raw.csv")

HEADERS = {
    "User-Agent": "CollegeCultureMatcher/1.0 (educational hackathon project)"
}

DELAY = 3
RETRY_DELAY = 30
MAX_RETRIES = 3

# Schools to re-scrape: school name -> list of subreddits to pull from
# Auburn: replace r/auburn (city sub) with r/wde (athletics) + r/AuburnUniversity (students)
# Alabama: re-scrape r/capstone with more pages and top posts
RESCRAPE = {
    "Auburn University": {"subs": ["wde", "AuburnUniversity"], "reason": "r/auburn is city sub, not university"},
    "University of Alabama": {"subs": ["capstone"], "reason": "only got 2 snippets first time"},
}

# Also re-scrape any school with fewer than 30 snippets
MIN_THRESHOLD = 30


def fetch_json(url, retry=0):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429 and retry < MAX_RETRIES:
            wait = RETRY_DELAY * (retry + 1)
            print(f"  [429] Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            return fetch_json(url, retry + 1)
        print(f"  [ERROR] HTTP {e.code}")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def fetch_comments(permalink, max_comments=15):
    safe_path = urllib.parse.quote(permalink, safe="/:@!$&'()*+,;=-._~")
    url = f"https://www.reddit.com{safe_path}.json?limit={max_comments}&raw_json=1"
    data = fetch_json(url)
    if not data or not isinstance(data, list) or len(data) < 2:
        return []
    comments = []
    for child in data[1].get("data", {}).get("children", []):
        if child.get("kind") != "t1":
            continue
        body = child.get("data", {}).get("body", "")
        score = child.get("data", {}).get("score", 0)
        if body and body != "[deleted]" and body != "[removed]":
            comments.append({"text": body, "score": score})
    return comments


def scrape_subreddit(school, subreddit_name, sort_modes=None):
    """Scrape a subreddit using multiple sort modes for more coverage."""
    if sort_modes is None:
        sort_modes = ["hot", "top"]  # top = all time best posts

    rows = []
    for sort in sort_modes:
        if sort == "top":
            url = f"https://www.reddit.com/r/{subreddit_name}/top.json?t=all&limit=50&raw_json=1"
        else:
            url = f"https://www.reddit.com/r/{subreddit_name}/{sort}.json?limit=50&raw_json=1"

        data = fetch_json(url)
        if not data or "data" not in data:
            continue

        posts = data["data"].get("children", [])
        comment_count = 0

        for post_wrapper in posts:
            post = post_wrapper.get("data", {})
            if post.get("stickied"):
                continue

            title = post.get("title", "")
            selftext = post.get("selftext", "")
            text = f"{title}. {selftext}" if selftext else title
            score = post.get("score", 0)

            if text.strip():
                rows.append({
                    "school": school,
                    "text": text,
                    "source_type": "post",
                    "score": score,
                })

            permalink = post.get("permalink", "")
            if permalink and comment_count < 15:
                time.sleep(DELAY + random.uniform(0, 2))
                comments = fetch_comments(permalink)
                for c in comments:
                    rows.append({
                        "school": school,
                        "text": c["text"],
                        "source_type": "comment",
                        "score": c["score"],
                    })
                comment_count += 1

        time.sleep(DELAY + random.uniform(0, 2))

    return rows


def main():
    # Load existing data
    df = pd.read_csv(OUTPUT_PATH)
    counts = df.groupby("school").size()
    print(f"Loaded {len(df)} existing snippets.\n")

    # Build list of schools to re-scrape
    to_scrape = dict(RESCRAPE)  # start with manual overrides

    # Add low-count schools using their original subreddit mapping
    from scrape_reddit import SCHOOL_SUBREDDITS
    for school, count in counts.items():
        if count < MIN_THRESHOLD and school not in to_scrape:
            sub = SCHOOL_SUBREDDITS.get(school)
            if sub:
                to_scrape[school] = {"subs": [sub], "reason": f"only {count} snippets"}

    if not to_scrape:
        print("All schools have enough data. Nothing to re-scrape.")
        return

    print(f"Re-scraping {len(to_scrape)} schools:")
    for school, info in to_scrape.items():
        current = counts.get(school, 0)
        print(f"  {school}: {current} snippets -> re-scraping from r/{', r/'.join(info['subs'])} ({info['reason']})")

    print()

    # Remove old data for schools being re-scraped
    schools_to_remove = set(to_scrape.keys())
    df = df[~df["school"].isin(schools_to_remove)]
    print(f"Removed old data for {len(schools_to_remove)} schools. {len(df)} snippets remaining.\n")

    # Re-scrape
    new_rows = []
    total = len(to_scrape)
    for i, (school, info) in enumerate(to_scrape.items(), 1):
        print(f"[{i}/{total}] Scraping {school}...")
        school_rows = []
        for sub in info["subs"]:
            print(f"  Fetching r/{sub}...")
            rows = scrape_subreddit(school, sub)
            school_rows.extend(rows)
            time.sleep(5 + random.uniform(0, 3))

        new_rows.extend(school_rows)
        print(f"  Got {len(school_rows)} snippets for {school}")

    # Combine and save
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([df, new_df], ignore_index=True)

    # Save with retry
    for attempt in range(5):
        try:
            combined.to_csv(OUTPUT_PATH, index=False)
            break
        except PermissionError:
            print(f"  [WARN] File locked, retrying in 5s... ({attempt + 1}/5)")
            time.sleep(5)

    # Print final counts
    final_counts = combined.groupby("school").size().sort_values(ascending=False)
    print(f"\nDone. Final counts:")
    print("=" * 50)
    for school, count in final_counts.items():
        flag = "  <-- LOW" if count < MIN_THRESHOLD else ""
        print(f"  {school:45s} {count:5d}{flag}")
    print(f"\nTotal: {len(combined)} snippets")


if __name__ == "__main__":
    main()
