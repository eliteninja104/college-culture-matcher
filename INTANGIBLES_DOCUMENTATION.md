# College Culture Matcher — Intangibles Side
## Comprehensive Technical Documentation for Bama Builds Hackathon (Data Science Track)

---

## 1. Project Overview

**College Culture Matcher** is a data-driven web application that helps prospective college students find schools that match their *cultural preferences* — not just rankings or test scores. The "Intangibles" side focuses on the qualities you can't find on a college's admissions page: how students actually feel about social life, academics, campus vibes, diversity, and more.

The app collects real student opinions from Reddit, combines them with structured data from Niche.com and Rate My Professors, and uses sentiment analysis + machine learning to build a 17-dimensional "culture profile" for each of 58 major US universities. Users take a 10-question personality-style quiz, and the app matches them to schools using Euclidean distance across all 17 dimensions.

---

## 2. Data Sources & Collection

### 2.1 Reddit (Student Opinions)
**What:** Posts and comments from each school's subreddit (e.g., r/capstone for University of Alabama, r/UTAustin for UT Austin).

**How:** We built a custom scraper using Reddit's public JSON API (`reddit.com/r/{subreddit}/hot.json`). No API key or authentication is required — the scraper fetches posts from each school's subreddit using standard HTTP requests.

**Details:**
- 58 schools mapped to their respective subreddits (e.g., University of Alabama -> r/capstone, Auburn -> r/wde + r/AuburnUniversity)
- Scrapes both "hot" and "top (all time)" posts
- Collects post titles, selftext, and top comments
- Rate limit handling: 3-second delays between requests, exponential backoff on HTTP 429 errors
- Resume capability: skips already-scraped schools if re-run
- Result: ~3,400+ text snippets across all 58 schools saved to `reddit_raw.csv`

**Why Reddit?** Reddit is one of the few platforms where college students discuss their genuine, unfiltered opinions about campus culture. Unlike official university marketing materials, Reddit posts capture the real student experience — complaints about dining hall food, excitement about gameday, frustrations with parking, praise for professors, etc.

### 2.2 Niche.com (Structured Grades)
**What:** Letter grades (A+ through F) across 9 quality dimensions for each school.

**Dimensions:**
| Column | What It Measures |
|--------|-----------------|
| academics_grade | Overall academic quality |
| value_grade | Value for tuition cost |
| diversity_grade | Campus diversity |
| campus_grade | Campus facilities and grounds |
| athletics_grade | Athletic programs and culture |
| party_scene_grade | Social and party culture |
| safety_grade | Campus safety |
| food_grade | Dining quality |
| dorms_grade | Housing quality |

**Processing:** Letter grades are converted to a 0-1 numeric scale using the mapping: A+ = 4.3, A = 4.0, A- = 3.7, ..., F = 0.0, then divided by 4.3 to normalize to [0, 1].

**Why Niche?** Niche aggregates millions of student reviews and official data into consistent, comparable grades. This gives us a reliable baseline even for schools where Reddit data may be sparse.

### 2.3 Rate My Professors (Professor Quality)
**What:** Average professor rating for each school.

**How:** We call RMP's GraphQL API directly at `ratemyprofessors.com/graphql`:
1. **Search query** (`NewSearchSchoolsQuery`): finds the school by name and returns its internal GraphQL ID
2. **Ratings query** (`SchoolRatingsQuery`): fetches `avgRatingRounded` and `numRatings`

This bypasses the broken `ratemyprofessor` Python package by going directly to the API that RMP's own website uses.

**Why RMP?** Professor quality is a major factor in the student experience, and RMP is the largest database of student-submitted professor reviews.

### 2.4 CampusLabs Engage (Student Organizations)
**What:** Complete club and organization lists for ~35 schools.

**How:** CampusLabs Engage is a platform many universities use to manage student organizations. Their API at `https://{domain}/engage/api/discovery/search/organizations` is public and returns JSON with club names, categories, and descriptions.

**Coverage:** 35 of our 58 schools have working CampusLabs Engage endpoints. For the remaining ~23 schools (including Florida, Ohio State, UCLA, Stanford, MIT, Harvard), the platform either uses a different system or doesn't expose a public API.

---

## 3. Sentiment Analysis Pipeline

### 3.1 Topic-Based Keyword Filtering
Before analyzing sentiment, each Reddit text snippet is categorized into one or more of **8 culture topics** using keyword matching:

| Topic | Example Keywords |
|-------|-----------------|
| **Social** | greek, frat, sorority, party, tailgate, bar, nightlife, rush |
| **Academics** | professor, class, exam, study, gpa, library, workload, research |
| **Campus** | campus, dorm, dining, food, gym, building, parking, housing |
| **Diversity** | diverse, inclusive, international, minority, welcoming, lgbtq |
| **Career** | career, internship, job, recruit, salary, alumni, networking |
| **Location** | town, city, weather, downtown, walkable, neighborhood |
| **Religion** | religious, church, conservative, liberal, spiritual, faith |
| **Spirit** | football, basketball, gameday, rivalry, stadium, cheer, SEC |

A single text snippet can match multiple topics (e.g., "The tailgate before the football game was insane" matches both Social and Spirit).

### 3.2 VADER Sentiment Scoring
For each school-topic combination, we run **VADER (Valence Aware Dictionary and sEntiment Reasoner)** on all matching text snippets.

**What VADER does:**
- Scores each text on a compound scale from -1 (extremely negative) to +1 (extremely positive)
- Handles slang, capitalization emphasis ("GREAT"), emoji, and punctuation ("amazing!!!")
- Pre-trained — no model training required

**Per-school, per-topic output:**
- **Sentiment score**: Average VADER compound score across all matching snippets, normalized from [-1, 1] to [0, 1]
- **Volume score**: Number of matching snippets, min-max scaled to [0, 1] across all schools

### 3.3 Representative Quotes
For each school-topic combination, we extract the top 3 most-upvoted positive quotes (VADER > 0.2) and top 3 most-upvoted negative quotes (VADER < -0.2). These are displayed in the app under each school's match card so users can read what real students are saying.

### 3.4 Profile Assembly
The final **school profile** is a merge of three data sources:

```
school_profiles.csv = Niche Grades (9 cols) + Reddit Sentiments (8 cols) + RMP Rating (1 col)
```

This gives us **17 numeric features** per school (plus volume columns used internally), all normalized to the [0, 1] range. Schools missing Reddit data default to 0.5 (neutral).

---

## 4. The 58 Schools

Our dataset covers a curated set of 58 schools chosen to represent different tiers and types:

- **All 16 SEC schools** (Alabama, Auburn, Florida, Georgia, LSU, Ole Miss, Mississippi State, Tennessee, Vanderbilt, Kentucky, South Carolina, Arkansas, Texas A&M, Missouri, Texas, Oklahoma)
- **Top public flagships** (UCLA, UC Berkeley, Michigan, UVA, UNC, Wisconsin, Ohio State, Penn State, Illinois, Purdue, Indiana, Maryland, UW, CU Boulder, ASU, Arizona, Oregon, Minnesota, Rutgers, Georgia Tech, Virginia Tech, NC State, Pitt, FSU, Clemson)
- **Top private universities** (Stanford, MIT, Harvard, Yale, Princeton, Duke, Northwestern, UChicago, Rice, Notre Dame, BU, NYU, USC, Georgetown, Emory, Wake Forest, Tulane)

---

## 5. The Culture Quiz

### 5.1 Design Philosophy
Instead of asking users to rate abstract concepts like "How much do you value diversity? (1-10)", we designed **10 scenario-based questions** that feel natural and relatable. Each answer maps to weighted scores across multiple culture dimensions.

### 5.2 The 10 Questions

| # | Question | What It Measures |
|---|----------|-----------------|
| Q1 | "It's Friday night. What sounds like the best time?" | Social life vs. academics vs. campus engagement |
| Q2 | "What matters most to you in a professor?" | Academic rigor vs. career focus vs. approachability |
| Q3 | "Pick your ideal Saturday in the fall:" | School spirit vs. city exploration vs. community |
| Q4 | "When you think about life after college, what excites you most?" | Career outcomes vs. social experience vs. academics |
| Q5 | "What kind of campus vibe do you want?" | Campus beauty vs. urban vs. college town |
| Q6 | "How important is Greek life?" | Social engagement level |
| Q7 | "How do you feel about diversity on campus?" | Diversity importance |
| Q8 | "How do you feel about faith and religion on campus?" | Religious culture preference |
| Q9 | "You got a free elective. What are you taking?" | Career vs. fun vs. intellectual vs. meaningful |
| Q10 | "What's your ideal school size?" | Big school energy vs. small school intimacy |

### 5.3 Scoring Mechanism
Each answer option has a hidden score map. For example, Q1's first option ("Huge house party or hitting the bars with a big group") scores `{"social": 10, "spirit": 6}`.

After all 10 questions, the scores for each dimension are **averaged** across all questions that contributed to it. If a dimension received no contributions, it defaults to 5 (neutral). This produces 8 dimension scores on a 1-10 scale:

- Social / Party Scene
- Academic Rigor
- Campus & Facilities
- Diversity & Inclusion
- Career Outcomes
- City vs. College Town
- Religious Community
- School Spirit & Athletics

### 5.4 Fine-Tuning
After the quiz, users see sidebar sliders pre-filled with their quiz-derived scores. They can manually adjust any dimension to explore different scenarios.

---

## 6. Matching Algorithm

### 6.1 User Vector Construction
The user's 8 dimension scores (1-10) are mapped to a **17-dimensional feature vector** that matches the school profile format:

```
SURVEY_TO_FEATURES = {
    "social":    ["social_sentiment", "party_scene_grade"],
    "academics": ["academics_sentiment", "academics_grade"],
    "campus":    ["campus_sentiment", "campus_grade", "food_grade", "dorms_grade"],
    "diversity": ["diversity_sentiment", "diversity_grade"],
    "career":    ["career_sentiment", "value_grade"],
    "location":  ["location_sentiment"],
    "religion":  ["religion_sentiment"],
    "spirit":    ["spirit_sentiment", "athletics_grade"],
}
```

**Critical detail:** Each slider value (1-10) is **interpolated within the actual data range** for that feature. If the min `academics_grade` across all schools is 0.63 and the max is 1.0, then slider value 1 maps to 0.63 and slider value 10 maps to 1.0. This ensures the user's point falls *within* the cloud of school points on the PCA map, not far outside it.

### 6.2 Euclidean Distance Scoring
We compute the **Euclidean distance** from the user's 17-dimensional vector to each school's 17-dimensional profile:

```
distance = sqrt( sum( (user[i] - school[i])^2 for i in all 17 features ) )
```

Distances are then converted to match percentages:
```
normalized = (distance - min_distance) / (max_distance - min_distance)
match_score = (1 - normalized) * 60 + 40
```

This produces scores from **40% (worst match) to 100% (best match)**, giving meaningful spread.

**Why Euclidean over Cosine Similarity?** We originally used cosine similarity, but because all school profiles have positive values in a similar range (0.5-1.0), cosine similarity gave everyone 95-99% match scores — no meaningful differentiation. Euclidean distance captures the *magnitude* of differences, not just directional alignment.

---

## 7. Visualization — PCA Culture Map

### 7.1 What PCA Does
PCA (Principal Component Analysis) compresses the 17-dimensional school profiles into 2 dimensions for visualization. Each school becomes a dot on a 2D scatter plot. Schools with similar culture profiles cluster together.

**Process:**
1. All 17 features are standardized (StandardScaler — zero mean, unit variance)
2. PCA extracts the 2 directions of maximum variance
3. Each school is projected onto these 2 axes

### 7.2 Axis Labels
The axes are labeled with the features that load most heavily onto each principal component. To avoid redundant labels (e.g., "Academic Quality" and "Academic Culture" appearing together), we deduplicate by first-word root.

### 7.3 Map Annotations
The **top 5 best-matching schools** (by 17D Euclidean distance, NOT by 2D PCA proximity) are labeled with arrows on the map. The user's ideal point is shown as a gold star.

**Important note:** Schools close on the 2D map may not be the best matches, and vice versa. The map is a 2D projection of 17D reality — like a shadow on a wall. The match rankings use all 17 dimensions, which is more accurate than the 2D view.

---

## 8. Additional Features

### 8.1 Radar Charts
Each school's match card includes a radar (spider) chart showing its scores across all 17 culture dimensions. Users also see their own preference profile as a radar chart.

### 8.2 Structural Filters
Users can filter by:
- **School size**: Small (<5k), Medium (5k-15k), Large (15k-30k), Very Large (30k+)
- **Region**: Northeast, Southeast, Midwest, Southwest, West Coast

### 8.3 Student Quotes
Each school card shows 2-4 representative Reddit quotes (top-upvoted positive and negative) tagged by topic, so users can read firsthand student opinions.

### 8.4 Club & Organization Finder
Users can search for specific clubs/activities (e.g., "ice skating", "robotics", "debate"). Results are ranked by **culture match score**, showing which schools both fit the user's preferences AND have the desired club. Each result includes links to the school's club directory.

### 8.5 Dimension Explorer
An alternative visualization mode lets users pick any two of the 17 dimensions as X and Y axes, with top-10 matches highlighted in green and the user's position shown as a gold star.

---

## 9. Technology Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit (Python web framework) |
| Sentiment Analysis | VADER (vaderSentiment) |
| Dimensionality Reduction | PCA (scikit-learn) |
| Matching | Euclidean Distance (NumPy) |
| Visualization | Plotly (interactive charts) |
| Data Collection | Custom scrapers (urllib, requests) |
| Reddit API | Public JSON endpoints (no auth) |
| RMP API | GraphQL API (direct calls) |
| Club Data | CampusLabs Engage REST API |

---

## 10. Data Pipeline Summary

```
                    Reddit Subreddits
                          |
                    [scrape_reddit.py]
                          |
                    reddit_raw.csv (3,400+ text snippets)
                          |
                    [sentiment_analysis.py]
                          |
    Niche Grades -------> Merge <------- RMP Ratings
    (9 dimensions)        |              (1 dimension)
                          |
                    school_profiles.csv (17 features x 58 schools)
                          |
                    [app.py]
                          |
              +-----------+-----------+
              |           |           |
          Quiz/Survey   PCA Map    Club Search
              |           |           |
          17D Vector   2D Projection  Ranked Results
              |           |           |
          Euclidean     Gold Star    Match Score +
          Distance      + Labels     Club Lists
              |
          Match Rankings (40-100%)
```

---

## 11. Key Design Decisions

1. **Scenario-based quiz over direct sliders**: Asking "What's your ideal Friday night?" produces more honest, natural responses than "Rate your interest in social life from 1-10."

2. **Euclidean distance over cosine similarity**: Cosine similarity failed to differentiate between schools because all profiles were positive vectors in a narrow range. Euclidean distance captures magnitude differences.

3. **Data range interpolation**: User slider values are mapped to each feature's actual data range, keeping the user within the school cloud on the PCA map.

4. **17 dimensions over 8**: By mapping each survey dimension to 2-4 profile features, we capture nuance. "Campus" preference affects campus_sentiment, campus_grade, food_grade, and dorms_grade separately.

5. **PCA labels show best matches, not nearest dots**: PCA proximity in 2D doesn't equal match quality in 17D. Labeling the actual top matches avoids confusion.

6. **Public APIs only**: All data collection uses public endpoints requiring no API keys, making the project fully reproducible.

---

## 12. File Structure

```
college-culture-matcher/
|-- app.py                          # Main Streamlit app (quiz, results, PCA map)
|-- college-major-map.html          # MajorMap interactive career map (Tangibles)
|-- requirements.txt                # Python dependencies
|-- .streamlit/config.toml          # Streamlit theme configuration
|-- data/
|   |-- scrape_reddit.py            # Reddit scraper (public JSON API)
|   |-- scrape_rmp.py               # Rate My Professors scraper (GraphQL)
|   |-- scrape_clubs.py             # CampusLabs Engage club scraper
|   |-- sentiment_analysis.py       # VADER sentiment pipeline
|   |-- niche_grades.csv            # Pre-populated Niche.com letter grades
|   |-- reddit_raw.csv              # Scraped Reddit text snippets
|   |-- rmp_data.csv                # Scraped RMP ratings
|   |-- school_profiles.csv         # Final merged profiles (17 features x 58 schools)
|   |-- representative_quotes.csv   # Top Reddit quotes per school/topic
|   |-- clubs.csv                   # Student organization data
|-- matching/
    |-- pca.py                      # Standalone PCA module (historical)
    |-- matcher.py                  # Standalone matcher module (historical)
```
