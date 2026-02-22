"""
Reddit scraper for College Culture Matcher.
Uses Reddit's public JSON endpoints — NO API key or account needed.

Usage:
    python data/scrape_reddit.py
"""

import os
import time
import csv
import json
import random
import urllib.request
import urllib.error
import urllib.parse

# ---------------------------------------------------------------------------
# School-to-subreddit mapping.  Extend this dict to add more schools.
# ---------------------------------------------------------------------------
SCHOOL_SUBREDDITS: dict[str, str] = {
    # SEC
    "University of Alabama": "capstone",
    "Auburn University": "auburn",
    "University of Florida": "ufl",
    "University of Georgia": "UGA",
    "LSU": "LSU",
    "University of Mississippi": "OleMiss",
    "Mississippi State": "msstate",
    "University of Tennessee": "UTK",
    "Vanderbilt University": "Vanderbilt",
    "University of Kentucky": "wildcats",
    "University of South Carolina": "Gamecocks",
    "University of Arkansas": "uark",
    "Texas A&M": "aggies",
    "University of Missouri": "mizzou",
    "University of Texas": "UTAustin",
    "University of Oklahoma": "sooners",
    # Top public flagships
    "UCLA": "UCLA",
    "UC Berkeley": "berkeley",
    "University of Michigan": "uofm",
    "University of Virginia": "UVA",
    "UNC Chapel Hill": "UNC",
    "University of Wisconsin": "UWMadison",
    "Ohio State University": "OSU",
    "Penn State": "PennStateUniversity",
    "University of Illinois": "UIUC",
    "Purdue University": "Purdue",
    "Indiana University": "IndianaUniversity",
    "University of Maryland": "UMD",
    "University of Washington": "udub",
    "University of Colorado Boulder": "cuboulder",
    "Arizona State University": "ASU",
    "University of Arizona": "UofArizona",
    "University of Oregon": "UofO",
    "University of Minnesota": "uofmn",
    "Rutgers University": "rutgers",
    "Georgia Tech": "gatech",
    "Virginia Tech": "VirginiaTech",
    "NC State": "NCSU",
    "University of Pittsburgh": "Pitt",
    "Florida State University": "fsu",
    "Clemson University": "Clemson",
    # Top privates
    "Stanford University": "stanford",
    "MIT": "mit",
    "Harvard University": "Harvard",
    "Yale University": "yale",
    "Princeton University": "princeton",
    "Duke University": "duke",
    "Northwestern University": "Northwestern",
    "University of Chicago": "uchicago",
    "Rice University": "riceuniversity",
    "Notre Dame": "notredame",
    "Boston University": "BostonU",
    "NYU": "nyu",
    "USC": "USC",
    "Georgetown University": "georgetown",
    "Emory University": "Emory",
    "Wake Forest University": "wfu",
    "Tulane University": "Tulane",
}

# Only fetch 1 page of posts (top 25) per school — enough for sentiment, avoids rate limits
MAX_PAGES = 1
POSTS_PER_PAGE = 25

# Only fetch comments on the top N posts per school to limit requests
MAX_POSTS_WITH_COMMENTS = 10

# Delays to avoid 429 rate limiting
DELAY_BETWEEN_REQUESTS = 3  # seconds between individual requests
DELAY_BETWEEN_SCHOOLS = 5   # seconds between schools
RETRY_DELAY = 30             # seconds to wait on a 429 before retrying
MAX_RETRIES = 3

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "reddit_raw.csv")

HEADERS = {
    "User-Agent": "CollegeCultureMatcher/1.0 (educational hackathon project)"
}


def fetch_json(url: str, retry: int = 0) -> dict | None:
    """Fetch JSON from a URL with proper headers and retry on 429."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429 and retry < MAX_RETRIES:
            wait = RETRY_DELAY * (retry + 1)
            print(f"  [429] Rate limited. Waiting {wait}s before retry {retry + 1}...")
            time.sleep(wait)
            return fetch_json(url, retry + 1)
        print(f"  [ERROR] HTTP {e.code} for {url}")
        return None
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"  [ERROR] {e}")
        return None


def scrape_school(school: str, subreddit_name: str) -> list[dict]:
    """Scrape posts and comments from a school subreddit using JSON API."""
    rows: list[dict] = []

    # Fetch one page of hot posts
    url = f"https://www.reddit.com/r/{subreddit_name}/hot.json?limit={POSTS_PER_PAGE}&raw_json=1"
    data = fetch_json(url)
    if not data or "data" not in data:
        return rows

    posts = data["data"].get("children", [])
    comment_count = 0

    for post_wrapper in posts:
        post = post_wrapper.get("data", {})

        # Skip stickied mod posts
        if post.get("stickied"):
            continue

        # Save post text
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

        # Only fetch comments for the top N posts
        permalink = post.get("permalink", "")
        if permalink and comment_count < MAX_POSTS_WITH_COMMENTS:
            time.sleep(DELAY_BETWEEN_REQUESTS + random.uniform(0, 2))
            comments = fetch_comments(permalink)
            for c in comments:
                rows.append({
                    "school": school,
                    "text": c["text"],
                    "source_type": "comment",
                    "score": c["score"],
                })
            comment_count += 1

    return rows


def fetch_comments(permalink: str, max_comments: int = 15) -> list[dict]:
    """Fetch top-level comments for a post using JSON API."""
    # URL-encode the permalink to handle non-ASCII characters (emoji, unicode titles)
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


def main():
    all_rows: list[dict] = []

    # Check if we have partial progress to resume from
    already_scraped: set[str] = set()
    if os.path.exists(OUTPUT_PATH):
        import pandas as pd
        try:
            existing = pd.read_csv(OUTPUT_PATH)
            already_scraped = set(existing["school"].unique())
            all_rows = existing.to_dict("records")
            print(f"Resuming — already have data for {len(already_scraped)} schools.")
        except Exception:
            pass

    total = len(SCHOOL_SUBREDDITS)
    for i, (school, sub) in enumerate(SCHOOL_SUBREDDITS.items(), 1):
        if school in already_scraped:
            print(f"[{i}/{total}] Skipping {school} (already scraped)")
            continue

        print(f"[{i}/{total}] Scraping r/{sub} for {school}...")
        rows = scrape_school(school, sub)
        all_rows.extend(rows)
        print(f"  Collected {len(rows)} text snippets. Total so far: {len(all_rows)}")

        # Save progress after each school (retry if file is locked by OneDrive/etc)
        for attempt in range(5):
            try:
                with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["school", "text", "source_type", "score"])
                    writer.writeheader()
                    writer.writerows(all_rows)
                break
            except PermissionError:
                print(f"  [WARN] File locked, retrying save in 5s... (attempt {attempt + 1}/5)")
                time.sleep(5)

        # Wait between schools
        time.sleep(DELAY_BETWEEN_SCHOOLS + random.uniform(0, 3))

    print(f"\nDone. Saved {len(all_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
