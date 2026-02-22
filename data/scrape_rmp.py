"""
Rate My Professors scraper for College Culture Matcher.
Uses RMP's GraphQL API directly — no broken packages needed.

Usage:
    python data/scrape_rmp.py
"""

import os
import csv
import time
import json
import requests

RMP_GRAPHQL_URL = "https://www.ratemyprofessors.com/graphql"
RMP_HEADERS = {
    "Authorization": "Basic dGVzdDp0ZXN0",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
}

# Schools to scrape — keys are our canonical names, values are RMP search strings.
SCHOOL_RMP_NAMES: dict[str, str] = {
    "University of Alabama": "University of Alabama",
    "Auburn University": "Auburn University",
    "University of Florida": "University of Florida",
    "University of Georgia": "University of Georgia",
    "LSU": "Louisiana State University",
    "University of Mississippi": "University of Mississippi",
    "Mississippi State": "Mississippi State University",
    "University of Tennessee": "University of Tennessee",
    "Vanderbilt University": "Vanderbilt University",
    "University of Kentucky": "University of Kentucky",
    "University of South Carolina": "University of South Carolina",
    "University of Arkansas": "University of Arkansas",
    "Texas A&M": "Texas A&M University",
    "University of Missouri": "University of Missouri",
    "University of Texas": "University of Texas at Austin",
    "University of Oklahoma": "University of Oklahoma",
    "UCLA": "University of California Los Angeles",
    "UC Berkeley": "University of California Berkeley",
    "University of Michigan": "University of Michigan",
    "University of Virginia": "University of Virginia",
    "UNC Chapel Hill": "University of North Carolina at Chapel Hill",
    "University of Wisconsin": "University of Wisconsin - Madison",
    "Ohio State University": "Ohio State University",
    "Penn State": "Pennsylvania State University",
    "University of Illinois": "University of Illinois Urbana-Champaign",
    "Purdue University": "Purdue University",
    "Indiana University": "Indiana University Bloomington",
    "University of Maryland": "University of Maryland",
    "University of Washington": "University of Washington",
    "University of Colorado Boulder": "University of Colorado Boulder",
    "Arizona State University": "Arizona State University",
    "University of Arizona": "University of Arizona",
    "University of Oregon": "University of Oregon",
    "University of Minnesota": "University of Minnesota",
    "Rutgers University": "Rutgers University",
    "Georgia Tech": "Georgia Institute of Technology",
    "Virginia Tech": "Virginia Polytechnic Institute and State University",
    "NC State": "North Carolina State University",
    "University of Pittsburgh": "University of Pittsburgh",
    "Florida State University": "Florida State University",
    "Clemson University": "Clemson University",
    "Stanford University": "Stanford University",
    "MIT": "Massachusetts Institute of Technology",
    "Harvard University": "Harvard University",
    "Yale University": "Yale University",
    "Princeton University": "Princeton University",
    "Duke University": "Duke University",
    "Northwestern University": "Northwestern University",
    "University of Chicago": "University of Chicago",
    "Rice University": "Rice University",
    "Notre Dame": "University of Notre Dame",
    "Boston University": "Boston University",
    "NYU": "New York University",
    "USC": "University of Southern California",
    "Georgetown University": "Georgetown University",
    "Emory University": "Emory University",
    "Wake Forest University": "Wake Forest University",
    "Tulane University": "Tulane University",
}

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "rmp_data.csv")

# GraphQL queries
SEARCH_SCHOOL_QUERY = """
query NewSearchSchoolsQuery($query: SchoolSearchQuery!) {
  newSearch {
    schools(query: $query) {
      edges {
        node {
          id
          legacyId
          name
          city
          state
        }
      }
    }
  }
}
"""

SCHOOL_RATINGS_QUERY = """
query SchoolRatingsQuery($id: ID!) {
  node(id: $id) {
    ... on School {
      id
      name
      city
      state
      avgRatingRounded
      numRatings
    }
  }
}
"""


def search_school(search_name: str) -> dict | None:
    """Search for a school by name and return the first match's GraphQL ID."""
    payload = {
        "query": SEARCH_SCHOOL_QUERY,
        "variables": {"query": {"text": search_name}},
    }
    try:
        resp = requests.post(RMP_GRAPHQL_URL, json=payload, headers=RMP_HEADERS, timeout=10)
        data = resp.json()
        edges = data.get("data", {}).get("newSearch", {}).get("schools", {}).get("edges", [])
        if edges:
            return edges[0]["node"]
    except Exception as e:
        print(f"  [ERROR] Search failed: {e}")
    return None


def get_school_rating(graphql_id: str) -> dict | None:
    """Get school-level rating data using its GraphQL ID."""
    payload = {
        "query": SCHOOL_RATINGS_QUERY,
        "variables": {"id": graphql_id},
    }
    try:
        resp = requests.post(RMP_GRAPHQL_URL, json=payload, headers=RMP_HEADERS, timeout=10)
        data = resp.json()
        return data.get("data", {}).get("node")
    except Exception as e:
        print(f"  [ERROR] Rating fetch failed: {e}")
    return None


def scrape_school(school_name: str, rmp_name: str) -> dict | None:
    """Search for school on RMP and fetch its ratings."""
    # Step 1: Find the school
    school_node = search_school(rmp_name)
    if not school_node:
        print(f"  [WARN] Not found on RMP: {rmp_name}")
        return None

    # Step 2: Get ratings
    ratings = get_school_rating(school_node["id"])
    if not ratings:
        print(f"  [WARN] No ratings for: {rmp_name}")
        return None

    return {
        "school": school_name,
        "avg_rating": ratings.get("avgRatingRounded"),
        "num_ratings": ratings.get("numRatings"),
        "rmp_city": ratings.get("city"),
        "rmp_state": ratings.get("state"),
    }


def main():
    rows: list[dict] = []
    total = len(SCHOOL_RMP_NAMES)

    for i, (school_name, rmp_name) in enumerate(SCHOOL_RMP_NAMES.items(), 1):
        print(f"[{i}/{total}] Fetching RMP data for {school_name}...")
        row = scrape_school(school_name, rmp_name)
        if row:
            print(f"  Rating: {row['avg_rating']} ({row['num_ratings']} ratings)")
            rows.append(row)
        time.sleep(1)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["school", "avg_rating", "num_ratings", "rmp_city", "rmp_state"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Saved {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
