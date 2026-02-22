"""
Club/organization scraper for College Culture Matcher.
Uses CampusLabs Engage API (public, no key needed) to pull student org lists.

Usage:
    python data/scrape_clubs.py
"""

import os
import re
import time
import json
import csv
import urllib.request
import urllib.error


def strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)       # strip tags
    text = re.sub(r"&[a-zA-Z]+;", " ", text)   # strip HTML entities like &amp;
    text = re.sub(r"&#\d+;", " ", text)         # strip numeric entities like &#160;
    text = re.sub(r"\s+", " ", text).strip()    # collapse whitespace
    return text

DATA_DIR = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(DATA_DIR, "clubs.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

# School -> CampusLabs Engage domain
# Only includes schools with confirmed working endpoints
SCHOOL_ENGAGE_DOMAINS: dict[str, str] = {
    "University of Alabama": "ua.campuslabs.com",
    "Auburn University": "auburn.campuslabs.com",
    "University of Georgia": "uga.campuslabs.com",
    "University of Mississippi": "olemiss.campuslabs.com",
    "Mississippi State": "msstate.campuslabs.com",
    "University of Tennessee": "utk.campuslabs.com",
    "Vanderbilt University": "anchorlink.vanderbilt.edu",
    "University of Kentucky": "uky.campuslabs.com",
    "University of South Carolina": "sc.campuslabs.com",
    "University of Oklahoma": "ou.campuslabs.com",
    "University of Texas": "utexas.campuslabs.com",
    "University of Michigan": "maizepages.umich.edu",
    "University of Virginia": "virginia.campuslabs.com",
    "UNC Chapel Hill": "unc.campuslabs.com",
    "University of Wisconsin": "win.wisc.edu",
    "Penn State": "psu.campuslabs.com",
    "Purdue University": "purdue.campuslabs.com",
    "Indiana University": "beinvolved.indiana.edu",
    "University of Maryland": "terplink.umd.edu",
    "University of Washington": "huskylink.washington.edu",
    "UC Berkeley": "berkeley.campuslabs.com",
    "Georgia Tech": "gatech.campuslabs.com",
    "Virginia Tech": "gobblerconnect.vt.edu",
    "NC State": "ncsu.campuslabs.com",
    "Florida State University": "nolecentral.dsa.fsu.edu",
    "Clemson University": "clemson.campuslabs.com",
    "University of Oregon": "uoregon.campuslabs.com",
    "Rutgers University": "rutgers.campuslabs.com",
    "University of Chicago": "uchicago.campuslabs.com",
    "Rice University": "rice.campuslabs.com",
    "Notre Dame": "nd.campuslabs.com",
    "Boston University": "bu.campuslabs.com",
    "NYU": "nyu.campuslabs.com",
    "Wake Forest University": "wfu.campuslabs.com",
    "Tulane University": "tulane.campuslabs.com",
}

PAGE_SIZE = 100  # max per request


def fetch_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def scrape_school_clubs(school: str, domain: str) -> list[dict]:
    """Fetch all student organizations for a school."""
    clubs = []
    skip = 0

    while True:
        url = (
            f"https://{domain}/engage/api/discovery/search/organizations"
            f"?orderBy%5B0%5D=UpperName%20asc&top={PAGE_SIZE}&skip={skip}&filter=&query="
        )
        data = fetch_json(url)
        if not data:
            break

        items = data.get("value", [])
        if not items:
            break

        for org in items:
            name = org.get("Name", "").strip()
            if name:
                clubs.append({
                    "school": school,
                    "club_name": name,
                    "category": org.get("CategoryNames", [""])[0] if org.get("CategoryNames") else "",
                    "description": strip_html((org.get("Description") or ""))[:500],
                })

        total = data.get("@odata.count", 0)
        skip += PAGE_SIZE
        if skip >= total:
            break

        time.sleep(0.5)

    return clubs


def main():
    all_clubs: list[dict] = []
    total = len(SCHOOL_ENGAGE_DOMAINS)

    for i, (school, domain) in enumerate(SCHOOL_ENGAGE_DOMAINS.items(), 1):
        print(f"[{i}/{total}] Scraping clubs for {school}...")
        clubs = scrape_school_clubs(school, domain)
        all_clubs.extend(clubs)
        print(f"  Found {len(clubs)} clubs/orgs.")
        time.sleep(1)

    # Save
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["school", "club_name", "category", "description"])
        writer.writeheader()
        writer.writerows(all_clubs)

    print(f"\nDone. Saved {len(all_clubs)} clubs to {OUTPUT_PATH}")

    # Print summary
    from collections import Counter
    counts = Counter(c["school"] for c in all_clubs)
    for school, count in counts.most_common():
        print(f"  {school:45s} {count:5d} clubs")


if __name__ == "__main__":
    main()
