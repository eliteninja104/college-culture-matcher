# College Culture Matcher

**Find your perfect college fit based on culture, not just rankings.**

Built for **Bama Builds Hackathon** — Data Science Track

---

## The Problem

College application volumes have **exploded** — up 47% per student over the last decade. Students are "shotgunning" applications at every prestigious name, driving acceptance rates down and stress up. Rankings focus on metrics students may not care about, while the factors that actually determine whether someone *thrives* at a school — social life, campus culture, diversity, faith community, school spirit — go unmeasured.

## Our Solution

College Culture Matcher uses **sentiment analysis of real student opinions** (Reddit, Niche.com, Rate My Professors) to build 17-dimensional culture profiles for 58 major US universities, then matches students to schools based on **cultural fit** through an interactive quiz.

The app has two paths:
- **Intangibles** — A 10-question culture quiz that matches you to schools based on vibe, social life, academics, diversity, and more
- **Tangibles** — An interactive career map showing salary, job markets, and cost of living by major across 16 US metro areas

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/college-culture-matcher.git
cd college-culture-matcher

# Install dependencies (Python 3.10+ required)
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Features

- **Culture Quiz**: 10 scenario-based questions mapping to 8 cultural dimensions
- **Smart Matching**: Euclidean distance matching across 17 culture features
- **PCA Visualization**: Interactive 2D scatter map of all 58 schools' culture profiles
- **Region Weighting**: Selecting a preferred region boosts (not filters) match scores
- **Tuition Filter**: Set a budget cap with in-state/out-of-state toggle
- **Career Tangibles**: Cost-of-living adjusted salary data for 10 majors across 16 metros
- **Club Finder**: Search 55,000+ student organizations ranked by culture match
- **MajorMap**: Interactive Leaflet.js career data map (standalone HTML)

## Data Sources

| Source | What We Get | Method |
|--------|------------|--------|
| **Reddit** | Student opinions on social life, academics, campus, diversity, careers, location, religion, school spirit | Public JSON API, VADER sentiment analysis |
| **Niche.com** | Letter grades for academics, value, diversity, campus, athletics, party scene, safety, food, dorms | Pre-scraped grade data |
| **Rate My Professors** | Average professor ratings | GraphQL API |
| **CampusLabs Engage** | Student organizations and clubs | REST API |
| **MajorMap** | Salary, jobs, cost of living, top employers by major and metro | Curated dataset |

## Tech Stack

- **Frontend**: Streamlit
- **Sentiment Analysis**: VADER (vaderSentiment)
- **Dimensionality Reduction**: PCA (scikit-learn)
- **Matching**: Euclidean distance on 17D feature vectors
- **Visualization**: Plotly, Leaflet.js
- **Data**: pandas, NumPy

## Project Structure

```
college-culture-matcher/
├── app.py                      # Main Streamlit application
├── application_trends.py       # Shotgunning problem visuals (Streamlit)
├── shotgunning_visuals.html    # Static version of trend visuals (open in browser)
├── college-major-map.html      # Interactive career data map
├── requirements.txt            # Python dependencies
├── INTANGIBLES_DOCUMENTATION.md # Technical documentation
├── data/
│   ├── school_profiles.csv     # 58 schools x 17 culture features
│   ├── representative_quotes.csv # Reddit quotes for display
│   ├── clubs.csv               # 55,000+ student organizations
│   ├── niche_grades.csv        # Niche.com letter grades
│   ├── reddit_raw.csv          # Raw Reddit scrape data
│   ├── rmp_data.csv            # Rate My Professors ratings
│   ├── scrape_reddit.py        # Reddit scraper
│   ├── scrape_rmp.py           # RMP scraper
│   ├── scrape_clubs.py         # Club data scraper
│   └── sentiment_analysis.py   # VADER pipeline
└── matching/
    ├── matcher.py              # Cosine similarity matching
    └── pca.py                  # PCA dimensionality reduction
```

## Schools Covered (58)

All 16 SEC schools, top public flagships (UCLA, UC Berkeley, Michigan, UVA, UNC, Wisconsin, Ohio State, Penn State, etc.), and elite private universities (Stanford, MIT, Harvard, Yale, Princeton, Duke, Northwestern, etc.)

## Team

Built for Bama Builds Hackathon — Data Science Track
