"""
College Culture Matcher — Streamlit App
Find your best-fit college based on cultural alignment, not just rankings.

Run with:
    streamlit run app.py
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
MATCHING_DIR = os.path.join(APP_DIR, "matching")

SENTIMENT_COLS = [
    "social_sentiment", "academics_sentiment", "campus_sentiment",
    "diversity_sentiment", "career_sentiment", "location_sentiment",
    "religion_sentiment", "spirit_sentiment",
]

NICHE_COLS = [
    "academics_grade", "value_grade", "diversity_grade", "campus_grade",
    "athletics_grade", "party_scene_grade", "safety_grade", "food_grade",
    "dorms_grade",
]

ALL_FEATURE_COLS = SENTIMENT_COLS + NICHE_COLS

FEATURE_LABELS = {
    "social_sentiment": "Social/Party Life",
    "academics_sentiment": "Academic Culture",
    "campus_sentiment": "Campus Feel",
    "diversity_sentiment": "Diversity Vibe",
    "career_sentiment": "Career Focus",
    "location_sentiment": "Location Appeal",
    "religion_sentiment": "Religious Culture",
    "spirit_sentiment": "School Spirit",
    "academics_grade": "Academic Quality",
    "value_grade": "Value",
    "diversity_grade": "Diversity Score",
    "campus_grade": "Campus Quality",
    "athletics_grade": "Athletics",
    "party_scene_grade": "Party Scene",
    "safety_grade": "Safety",
    "food_grade": "Food Quality",
    "dorms_grade": "Dorm Quality",
}

# Maps survey slider dimensions to profile feature columns
SURVEY_TO_FEATURES = {
    "social": ["social_sentiment", "party_scene_grade"],
    "academics": ["academics_sentiment", "academics_grade"],
    "campus": ["campus_sentiment", "campus_grade", "food_grade", "dorms_grade"],
    "diversity": ["diversity_sentiment", "diversity_grade"],
    "career": ["career_sentiment", "value_grade"],
    "location": ["location_sentiment"],
    "religion": ["religion_sentiment"],
    "spirit": ["spirit_sentiment", "athletics_grade"],
}

# School metadata for filtering and display
# tuition_in = in-state annual tuition, tuition_out = out-of-state annual tuition (approx. 2024-25)
SCHOOL_META = {
    "University of Alabama": {"size": "Very Large", "region": "Southeast", "enrollment": 38320, "city": "Tuscaloosa, AL", "tuition_in": 11580, "tuition_out": 32300, "url": "https://www.ua.edu", "clubs_url": "https://ua.campuslabs.com/engage/organizations"},
    "Auburn University": {"size": "Very Large", "region": "Southeast", "enrollment": 31526, "city": "Auburn, AL", "tuition_in": 12440, "tuition_out": 33610, "url": "https://www.auburn.edu", "clubs_url": "https://auburn.campuslabs.com/engage/organizations"},
    "University of Florida": {"size": "Very Large", "region": "Southeast", "enrollment": 55211, "city": "Gainesville, FL", "tuition_in": 6380, "tuition_out": 28660, "url": "https://www.ufl.edu", "clubs_url": "https://www.studentinvolvement.ufl.edu/organizations/"},
    "University of Georgia": {"size": "Very Large", "region": "Southeast", "enrollment": 40607, "city": "Athens, GA", "tuition_in": 12080, "tuition_out": 31120, "url": "https://www.uga.edu", "clubs_url": "https://uga.campuslabs.com/engage/organizations"},
    "LSU": {"size": "Very Large", "region": "Southeast", "enrollment": 36136, "city": "Baton Rouge, LA", "tuition_in": 11950, "tuition_out": 28630, "url": "https://www.lsu.edu", "clubs_url": "https://www.lsu.edu/studentlife/involvement/"},
    "University of Mississippi": {"size": "Large", "region": "Southeast", "enrollment": 21703, "city": "Oxford, MS", "tuition_in": 9220, "tuition_out": 26390, "url": "https://www.olemiss.edu", "clubs_url": "https://olemiss.campuslabs.com/engage/organizations"},
    "Mississippi State": {"size": "Large", "region": "Southeast", "enrollment": 22226, "city": "Starkville, MS", "tuition_in": 9808, "tuition_out": 25250, "url": "https://www.msstate.edu", "clubs_url": "https://msstate.campuslabs.com/engage/organizations"},
    "University of Tennessee": {"size": "Very Large", "region": "Southeast", "enrollment": 34425, "city": "Knoxville, TN", "tuition_in": 13244, "tuition_out": 31504, "url": "https://www.utk.edu", "clubs_url": "https://utk.campuslabs.com/engage/organizations"},
    "Vanderbilt University": {"size": "Medium", "region": "Southeast", "enrollment": 13710, "city": "Nashville, TN", "tuition_in": 62458, "tuition_out": 62458, "url": "https://www.vanderbilt.edu", "clubs_url": "https://anchorlink.vanderbilt.edu/engage/organizations"},
    "University of Kentucky": {"size": "Very Large", "region": "Southeast", "enrollment": 30545, "city": "Lexington, KY", "tuition_in": 12860, "tuition_out": 32570, "url": "https://www.uky.edu", "clubs_url": "https://uky.campuslabs.com/engage/organizations"},
    "University of South Carolina": {"size": "Very Large", "region": "Southeast", "enrollment": 35364, "city": "Columbia, SC", "tuition_in": 12688, "tuition_out": 33928, "url": "https://www.sc.edu", "clubs_url": "https://sc.campuslabs.com/engage/organizations"},
    "University of Arkansas": {"size": "Very Large", "region": "Southeast", "enrollment": 29068, "city": "Fayetteville, AR", "tuition_in": 9658, "tuition_out": 27338, "url": "https://www.uark.edu", "clubs_url": "https://www.uark.edu/campus-life/student-organizations.php"},
    "Texas A&M": {"size": "Very Large", "region": "Southwest", "enrollment": 72982, "city": "College Station, TX", "tuition_in": 12342, "tuition_out": 39394, "url": "https://www.tamu.edu", "clubs_url": "https://stuactonline.tamu.edu/app/search/"},
    "University of Missouri": {"size": "Very Large", "region": "Midwest", "enrollment": 31401, "city": "Columbia, MO", "tuition_in": 11836, "tuition_out": 30244, "url": "https://missouri.edu", "clubs_url": "https://missouri.edu/student-life/organizations"},
    "University of Texas": {"size": "Very Large", "region": "Southwest", "enrollment": 51991, "city": "Austin, TX", "tuition_in": 11448, "tuition_out": 41070, "url": "https://www.utexas.edu", "clubs_url": "https://utexas.campuslabs.com/engage/organizations"},
    "University of Oklahoma": {"size": "Very Large", "region": "Southwest", "enrollment": 28826, "city": "Norman, OK", "tuition_in": 9062, "tuition_out": 27567, "url": "https://www.ou.edu", "clubs_url": "https://ou.campuslabs.com/engage/organizations"},
    "UCLA": {"size": "Very Large", "region": "West Coast", "enrollment": 46116, "city": "Los Angeles, CA", "tuition_in": 13804, "tuition_out": 46326, "url": "https://www.ucla.edu", "clubs_url": "https://www.studentorgs.ucla.edu/"},
    "UC Berkeley": {"size": "Very Large", "region": "West Coast", "enrollment": 45057, "city": "Berkeley, CA", "tuition_in": 14312, "tuition_out": 48176, "url": "https://www.berkeley.edu", "clubs_url": "https://berkeley.campuslabs.com/engage/organizations"},
    "University of Michigan": {"size": "Very Large", "region": "Midwest", "enrollment": 47907, "city": "Ann Arbor, MI", "tuition_in": 17228, "tuition_out": 57273, "url": "https://umich.edu", "clubs_url": "https://maizepages.umich.edu/engage/organizations"},
    "University of Virginia": {"size": "Large", "region": "Southeast", "enrollment": 25627, "city": "Charlottesville, VA", "tuition_in": 19814, "tuition_out": 55914, "url": "https://www.virginia.edu", "clubs_url": "https://virginia.campuslabs.com/engage/organizations"},
    "UNC Chapel Hill": {"size": "Very Large", "region": "Southeast", "enrollment": 30862, "city": "Chapel Hill, NC", "tuition_in": 8998, "tuition_out": 38066, "url": "https://www.unc.edu", "clubs_url": "https://unc.campuslabs.com/engage/organizations"},
    "University of Wisconsin": {"size": "Very Large", "region": "Midwest", "enrollment": 47932, "city": "Madison, WI", "tuition_in": 10796, "tuition_out": 39427, "url": "https://www.wisc.edu", "clubs_url": "https://win.wisc.edu/engage/organizations"},
    "Ohio State University": {"size": "Very Large", "region": "Midwest", "enrollment": 61369, "city": "Columbus, OH", "tuition_in": 11936, "tuition_out": 36722, "url": "https://www.osu.edu", "clubs_url": "https://activities.osu.edu/involvement/student_organizations/"},
    "Penn State": {"size": "Very Large", "region": "Northeast", "enrollment": 46810, "city": "State College, PA", "tuition_in": 19286, "tuition_out": 38651, "url": "https://www.psu.edu", "clubs_url": "https://psu.campuslabs.com/engage/organizations"},
    "University of Illinois": {"size": "Very Large", "region": "Midwest", "enrollment": 56607, "city": "Champaign, IL", "tuition_in": 15094, "tuition_out": 34316, "url": "https://illinois.edu", "clubs_url": "https://illinois.edu/campus-life/student-organizations"},
    "Purdue University": {"size": "Very Large", "region": "Midwest", "enrollment": 49639, "city": "West Lafayette, IN", "tuition_in": 9992, "tuition_out": 28794, "url": "https://www.purdue.edu", "clubs_url": "https://purdue.campuslabs.com/engage/organizations"},
    "Indiana University": {"size": "Very Large", "region": "Midwest", "enrollment": 42760, "city": "Bloomington, IN", "tuition_in": 11164, "tuition_out": 38332, "url": "https://www.indiana.edu", "clubs_url": "https://beinvolved.indiana.edu/engage/organizations"},
    "University of Maryland": {"size": "Very Large", "region": "Northeast", "enrollment": 40792, "city": "College Park, MD", "tuition_in": 11233, "tuition_out": 40306, "url": "https://www.umd.edu", "clubs_url": "https://terplink.umd.edu/engage/organizations"},
    "University of Washington": {"size": "Very Large", "region": "West Coast", "enrollment": 48149, "city": "Seattle, WA", "tuition_in": 12076, "tuition_out": 40740, "url": "https://www.washington.edu", "clubs_url": "https://huskylink.washington.edu/engage/organizations"},
    "University of Colorado Boulder": {"size": "Very Large", "region": "Southwest", "enrollment": 38661, "city": "Boulder, CO", "tuition_in": 13250, "tuition_out": 40260, "url": "https://www.colorado.edu", "clubs_url": "https://www.colorado.edu/involvement/student-organizations"},
    "Arizona State University": {"size": "Very Large", "region": "Southwest", "enrollment": 77881, "city": "Tempe, AZ", "tuition_in": 12338, "tuition_out": 32068, "url": "https://www.asu.edu", "clubs_url": "https://asu.campuslabs.com/engage/organizations"},
    "University of Arizona": {"size": "Very Large", "region": "Southwest", "enrollment": 46525, "city": "Tucson, AZ", "tuition_in": 12900, "tuition_out": 38200, "url": "https://www.arizona.edu", "clubs_url": "https://arizona.campuslabs.com/engage/organizations"},
    "University of Oregon": {"size": "Large", "region": "West Coast", "enrollment": 23202, "city": "Eugene, OR", "tuition_in": 13251, "tuition_out": 40653, "url": "https://www.uoregon.edu", "clubs_url": "https://uoregon.campuslabs.com/engage/organizations"},
    "University of Minnesota": {"size": "Very Large", "region": "Midwest", "enrollment": 54955, "city": "Minneapolis, MN", "tuition_in": 15288, "tuition_out": 36402, "url": "https://twin-cities.umn.edu", "clubs_url": "https://gopherlink.umn.edu/organizations"},
    "Rutgers University": {"size": "Very Large", "region": "Northeast", "enrollment": 50411, "city": "New Brunswick, NJ", "tuition_in": 16592, "tuition_out": 34116, "url": "https://www.rutgers.edu", "clubs_url": "https://rutgers.campuslabs.com/engage/organizations"},
    "Georgia Tech": {"size": "Very Large", "region": "Southeast", "enrollment": 44007, "city": "Atlanta, GA", "tuition_in": 12682, "tuition_out": 33964, "url": "https://www.gatech.edu", "clubs_url": "https://gatech.campuslabs.com/engage/organizations"},
    "Virginia Tech": {"size": "Very Large", "region": "Southeast", "enrollment": 37431, "city": "Blacksburg, VA", "tuition_in": 14625, "tuition_out": 35186, "url": "https://www.vt.edu", "clubs_url": "https://gobblerconnect.vt.edu/engage/organizations"},
    "NC State": {"size": "Very Large", "region": "Southeast", "enrollment": 36304, "city": "Raleigh, NC", "tuition_in": 9128, "tuition_out": 30869, "url": "https://www.ncsu.edu", "clubs_url": "https://ncsu.campuslabs.com/engage/organizations"},
    "University of Pittsburgh": {"size": "Very Large", "region": "Northeast", "enrollment": 34228, "city": "Pittsburgh, PA", "tuition_in": 20220, "tuition_out": 37332, "url": "https://www.pitt.edu", "clubs_url": "https://www.pitt.edu/campus-life/student-organizations"},
    "Florida State University": {"size": "Very Large", "region": "Southeast", "enrollment": 44161, "city": "Tallahassee, FL", "tuition_in": 6517, "tuition_out": 21683, "url": "https://www.fsu.edu", "clubs_url": "https://nolecentral.dsa.fsu.edu/engage/organizations"},
    "Clemson University": {"size": "Large", "region": "Southeast", "enrollment": 27341, "city": "Clemson, SC", "tuition_in": 15558, "tuition_out": 39498, "url": "https://www.clemson.edu", "clubs_url": "https://clemson.campuslabs.com/engage/organizations"},
    "Stanford University": {"size": "Medium", "region": "West Coast", "enrollment": 17680, "city": "Stanford, CA", "tuition_in": 62484, "tuition_out": 62484, "url": "https://www.stanford.edu", "clubs_url": "https://www.stanford.edu/student-life/student-organizations/"},
    "MIT": {"size": "Medium", "region": "Northeast", "enrollment": 11858, "city": "Cambridge, MA", "tuition_in": 61990, "tuition_out": 61990, "url": "https://www.mit.edu", "clubs_url": "https://engage.mit.edu/"},
    "Harvard University": {"size": "Large", "region": "Northeast", "enrollment": 21015, "city": "Cambridge, MA", "tuition_in": 59076, "tuition_out": 59076, "url": "https://www.harvard.edu", "clubs_url": "https://college.harvard.edu/campus-life/student-organizations"},
    "Yale University": {"size": "Medium", "region": "Northeast", "enrollment": 14776, "city": "New Haven, CT", "tuition_in": 64700, "tuition_out": 64700, "url": "https://www.yale.edu", "clubs_url": "https://yaleconnect.yale.edu/"},
    "Princeton University": {"size": "Medium", "region": "Northeast", "enrollment": 8478, "city": "Princeton, NJ", "tuition_in": 59710, "tuition_out": 59710, "url": "https://www.princeton.edu", "clubs_url": "https://www.princeton.edu/campus-life/student-organizations"},
    "Duke University": {"size": "Medium", "region": "Southeast", "enrollment": 17620, "city": "Durham, NC", "tuition_in": 63054, "tuition_out": 63054, "url": "https://www.duke.edu", "clubs_url": "https://dukegroups.com/"},
    "Northwestern University": {"size": "Large", "region": "Midwest", "enrollment": 22603, "city": "Evanston, IL", "tuition_in": 63468, "tuition_out": 63468, "url": "https://www.northwestern.edu", "clubs_url": "https://northwestern.campuslabs.com/engage/organizations"},
    "University of Chicago": {"size": "Medium", "region": "Midwest", "enrollment": 17834, "city": "Chicago, IL", "tuition_in": 64260, "tuition_out": 64260, "url": "https://www.uchicago.edu", "clubs_url": "https://uchicago.campuslabs.com/engage/organizations"},
    "Rice University": {"size": "Small", "region": "Southwest", "enrollment": 8285, "city": "Houston, TX", "tuition_in": 58128, "tuition_out": 58128, "url": "https://www.rice.edu", "clubs_url": "https://rice.campuslabs.com/engage/organizations"},
    "Notre Dame": {"size": "Medium", "region": "Midwest", "enrollment": 13139, "city": "Notre Dame, IN", "tuition_in": 62693, "tuition_out": 62693, "url": "https://www.nd.edu", "clubs_url": "https://nd.campuslabs.com/engage/organizations"},
    "Boston University": {"size": "Very Large", "region": "Northeast", "enrollment": 36714, "city": "Boston, MA", "tuition_in": 65168, "tuition_out": 65168, "url": "https://www.bu.edu", "clubs_url": "https://bu.campuslabs.com/engage/organizations"},
    "NYU": {"size": "Very Large", "region": "Northeast", "enrollment": 58226, "city": "New York, NY", "tuition_in": 60438, "tuition_out": 60438, "url": "https://www.nyu.edu", "clubs_url": "https://nyu.campuslabs.com/engage/organizations"},
    "USC": {"size": "Very Large", "region": "West Coast", "enrollment": 49500, "city": "Los Angeles, CA", "tuition_in": 66640, "tuition_out": 66640, "url": "https://www.usc.edu", "clubs_url": "https://campusactivities.usc.edu/"},
    "Georgetown University": {"size": "Large", "region": "Northeast", "enrollment": 19459, "city": "Washington, DC", "tuition_in": 63820, "tuition_out": 63820, "url": "https://www.georgetown.edu", "clubs_url": "https://hoyalink.georgetown.edu/"},
    "Emory University": {"size": "Medium", "region": "Southeast", "enrollment": 14067, "city": "Atlanta, GA", "tuition_in": 60774, "tuition_out": 60774, "url": "https://www.emory.edu", "clubs_url": "https://emory.campuslabs.com/engage/organizations"},
    "Wake Forest University": {"size": "Medium", "region": "Southeast", "enrollment": 8938, "city": "Winston-Salem, NC", "tuition_in": 62294, "tuition_out": 62294, "url": "https://www.wfu.edu", "clubs_url": "https://wfu.campuslabs.com/engage/organizations"},
    "Tulane University": {"size": "Medium", "region": "Southeast", "enrollment": 13433, "city": "New Orleans, LA", "tuition_in": 63178, "tuition_out": 63178, "url": "https://www.tulane.edu", "clubs_url": "https://tulane.campuslabs.com/engage/organizations"},
}

# ---------------------------------------------------------------------------
# Career / Tangibles Data (from MajorMap)
# ---------------------------------------------------------------------------
MAJORS = [
    {"id": "cs",         "label": "Computer Science",   "icon": "💻"},
    {"id": "nursing",    "label": "Nursing / Health",    "icon": "🏥"},
    {"id": "finance",    "label": "Finance / Business",  "icon": "📈"},
    {"id": "mecheng",    "label": "Mechanical Eng.",     "icon": "⚙️"},
    {"id": "education",  "label": "Education",           "icon": "📚"},
    {"id": "biology",    "label": "Biology / Pre-Med",   "icon": "🔬"},
    {"id": "marketing",  "label": "Marketing / Comm.",   "icon": "📣"},
    {"id": "psychology", "label": "Psychology",           "icon": "🧠"},
    {"id": "polisci",    "label": "Political Science",   "icon": "⚖️"},
    {"id": "design",     "label": "Design / UX",         "icon": "🎨"},
]

# Metro areas with career data per major
# col = cost of living index (100 = national average)
# raw = raw median salary, jobs = job postings per 100k residents, emp = top employers
METRO_DATA = {
    "San Francisco, CA":  {"col": 194, "cs": {"raw": 165000, "jobs": 310, "emp": "Google, Salesforce, Stripe"}, "nursing": {"raw": 95000, "jobs": 88, "emp": "UCSF, Sutter Health"}, "finance": {"raw": 130000, "jobs": 140, "emp": "Wells Fargo, BlackRock"}, "mecheng": {"raw": 125000, "jobs": 95, "emp": "Tesla, Apple"}, "education": {"raw": 72000, "jobs": 45, "emp": "SFUSD, Stanford"}, "biology": {"raw": 115000, "jobs": 130, "emp": "Genentech, UCSF"}, "marketing": {"raw": 98000, "jobs": 120, "emp": "Salesforce, Lyft"}, "psychology": {"raw": 75000, "jobs": 60, "emp": "UCSF, Kaiser"}, "polisci": {"raw": 85000, "jobs": 55, "emp": "City Gov, Law Firms"}, "design": {"raw": 110000, "jobs": 145, "emp": "Figma, Adobe, Airbnb"}},
    "Seattle, WA":        {"col": 162, "cs": {"raw": 155000, "jobs": 280, "emp": "Amazon, Microsoft, Zillow"}, "nursing": {"raw": 88000, "jobs": 95, "emp": "UW Medicine, Swedish"}, "finance": {"raw": 110000, "jobs": 100, "emp": "Washington Federal, Starbucks"}, "mecheng": {"raw": 115000, "jobs": 130, "emp": "Boeing, Amazon"}, "education": {"raw": 65000, "jobs": 50, "emp": "Seattle Public Schools"}, "biology": {"raw": 100000, "jobs": 100, "emp": "Fred Hutch, UW"}, "marketing": {"raw": 88000, "jobs": 105, "emp": "Amazon, Expedia"}, "psychology": {"raw": 70000, "jobs": 65, "emp": "UW, VA Hospital"}, "polisci": {"raw": 80000, "jobs": 50, "emp": "City of Seattle, NGOs"}, "design": {"raw": 105000, "jobs": 130, "emp": "Amazon, Microsoft"}},
    "New York, NY":       {"col": 187, "cs": {"raw": 150000, "jobs": 250, "emp": "Google, Goldman Sachs, Meta"}, "nursing": {"raw": 100000, "jobs": 110, "emp": "NYC Health, NYP"}, "finance": {"raw": 160000, "jobs": 290, "emp": "JPMorgan, Goldman Sachs"}, "mecheng": {"raw": 110000, "jobs": 80, "emp": "GE, Consolidated Edison"}, "education": {"raw": 75000, "jobs": 60, "emp": "NYC DOE, Teach For America"}, "biology": {"raw": 105000, "jobs": 120, "emp": "Pfizer, Columbia Med"}, "marketing": {"raw": 100000, "jobs": 180, "emp": "WPP, Publicis, L'Oreal"}, "psychology": {"raw": 80000, "jobs": 90, "emp": "NYU, Bellevue Hospital"}, "polisci": {"raw": 90000, "jobs": 100, "emp": "NYC Gov, UN, Law Firms"}, "design": {"raw": 105000, "jobs": 160, "emp": "R/GA, Huge, Pentagram"}},
    "Austin, TX":         {"col": 121, "cs": {"raw": 135000, "jobs": 240, "emp": "Dell, Tesla, Apple"}, "nursing": {"raw": 72000, "jobs": 100, "emp": "St. David's, Ascension"}, "finance": {"raw": 95000, "jobs": 120, "emp": "Indeed, Charles Schwab"}, "mecheng": {"raw": 105000, "jobs": 110, "emp": "Tesla, Samsung Austin"}, "education": {"raw": 52000, "jobs": 55, "emp": "Austin ISD, UT Austin"}, "biology": {"raw": 82000, "jobs": 90, "emp": "UT Austin, Bio-techne"}, "marketing": {"raw": 78000, "jobs": 130, "emp": "Dell, Indeed, HomeAway"}, "psychology": {"raw": 60000, "jobs": 70, "emp": "UT Health Austin"}, "polisci": {"raw": 70000, "jobs": 65, "emp": "State Capitol, Lobbying Firms"}, "design": {"raw": 88000, "jobs": 120, "emp": "Whole Foods, GSD&M"}},
    "Chicago, IL":        {"col": 117, "cs": {"raw": 125000, "jobs": 190, "emp": "Motorola, Groupon, Braintree"}, "nursing": {"raw": 75000, "jobs": 100, "emp": "Northwestern Medicine, Rush"}, "finance": {"raw": 130000, "jobs": 200, "emp": "CME Group, Citadel"}, "mecheng": {"raw": 100000, "jobs": 100, "emp": "Boeing, Caterpillar, Abbott"}, "education": {"raw": 60000, "jobs": 65, "emp": "CPS, DePaul"}, "biology": {"raw": 85000, "jobs": 100, "emp": "AbbVie, Northwestern Med"}, "marketing": {"raw": 82000, "jobs": 140, "emp": "Orbitz, Morningstar"}, "psychology": {"raw": 62000, "jobs": 75, "emp": "UChicago, Rush Hospital"}, "polisci": {"raw": 78000, "jobs": 90, "emp": "City of Chicago, Law Firms"}, "design": {"raw": 82000, "jobs": 100, "emp": "Leo Burnett, FCB"}},
    "Houston, TX":        {"col": 108, "cs": {"raw": 118000, "jobs": 160, "emp": "HP, CenterPoint Energy"}, "nursing": {"raw": 70000, "jobs": 115, "emp": "Houston Methodist, MD Anderson"}, "finance": {"raw": 105000, "jobs": 140, "emp": "JP Morgan Chase, Shell"}, "mecheng": {"raw": 115000, "jobs": 200, "emp": "ExxonMobil, Chevron, Shell"}, "education": {"raw": 50000, "jobs": 55, "emp": "HISD, Houston CC"}, "biology": {"raw": 88000, "jobs": 105, "emp": "MD Anderson, UTHealth"}, "marketing": {"raw": 75000, "jobs": 110, "emp": "Sysco, Group 1 Auto"}, "psychology": {"raw": 60000, "jobs": 65, "emp": "UTHealth, VA Medical Center"}, "polisci": {"raw": 72000, "jobs": 60, "emp": "Harris County, Law Firms"}, "design": {"raw": 75000, "jobs": 80, "emp": "C&D, Design Bridge"}},
    "Denver, CO":         {"col": 130, "cs": {"raw": 125000, "jobs": 180, "emp": "Arrow Electronics, Ibotta"}, "nursing": {"raw": 78000, "jobs": 95, "emp": "UCHealth, HealthOne"}, "finance": {"raw": 95000, "jobs": 105, "emp": "Charles Schwab, TIAA"}, "mecheng": {"raw": 100000, "jobs": 105, "emp": "Lockheed, Ball Aerospace"}, "education": {"raw": 55000, "jobs": 60, "emp": "Denver Public Schools, CU Denver"}, "biology": {"raw": 82000, "jobs": 85, "emp": "CU Anschutz, Env. firms"}, "marketing": {"raw": 80000, "jobs": 110, "emp": "REI, VF Corporation"}, "psychology": {"raw": 64000, "jobs": 75, "emp": "CU Anschutz, Centura Health"}, "polisci": {"raw": 70000, "jobs": 60, "emp": "Colorado Gov, NGOs"}, "design": {"raw": 85000, "jobs": 105, "emp": "Cactus, Made"}},
    "Boston, MA":         {"col": 162, "cs": {"raw": 145000, "jobs": 230, "emp": "HubSpot, Wayfair, MIT"}, "nursing": {"raw": 90000, "jobs": 100, "emp": "Mass General, Dana-Farber"}, "finance": {"raw": 130000, "jobs": 160, "emp": "Fidelity, State Street, Putnam"}, "mecheng": {"raw": 115000, "jobs": 120, "emp": "Raytheon, GE Aviation, MIT LL"}, "education": {"raw": 65000, "jobs": 55, "emp": "BPS, Teach For America"}, "biology": {"raw": 110000, "jobs": 160, "emp": "Broad Institute, Biogen, Novartis"}, "marketing": {"raw": 92000, "jobs": 120, "emp": "HubSpot, Wayfair, Brightcove"}, "psychology": {"raw": 72000, "jobs": 80, "emp": "Harvard Med, McLean"}, "polisci": {"raw": 85000, "jobs": 80, "emp": "State House, Policy Groups"}, "design": {"raw": 95000, "jobs": 115, "emp": "IDEO, Communispace"}},
    "Washington, DC":     {"col": 158, "cs": {"raw": 140000, "jobs": 220, "emp": "Booz Allen, SAIC, Leidos"}, "nursing": {"raw": 88000, "jobs": 95, "emp": "NIH, MedStar, Georgetown"}, "finance": {"raw": 120000, "jobs": 160, "emp": "World Bank, IMF, Freddie Mac"}, "mecheng": {"raw": 115000, "jobs": 130, "emp": "Northrop, Raytheon, MITRE"}, "education": {"raw": 72000, "jobs": 70, "emp": "DCPS, Sidwell Friends"}, "biology": {"raw": 105000, "jobs": 140, "emp": "NIH, FDA, Georgetown Med"}, "marketing": {"raw": 88000, "jobs": 120, "emp": "Gartner, Advisory Board"}, "psychology": {"raw": 80000, "jobs": 95, "emp": "NIH, Walter Reed, APA"}, "polisci": {"raw": 95000, "jobs": 200, "emp": "Congress, Think Tanks, NGOs"}, "design": {"raw": 90000, "jobs": 100, "emp": "IDEO, ICF, Booz Allen DX"}},
    "Atlanta, GA":        {"col": 109, "cs": {"raw": 118000, "jobs": 175, "emp": "NCR, Georgia Tech, Cox"}, "nursing": {"raw": 68000, "jobs": 105, "emp": "Emory, Piedmont, Grady"}, "finance": {"raw": 100000, "jobs": 130, "emp": "SunTrust, Fiserv"}, "mecheng": {"raw": 100000, "jobs": 95, "emp": "Delta, Chick-fil-A, Georgia-Pacific"}, "education": {"raw": 50000, "jobs": 65, "emp": "APS, Fulton County Schools"}, "biology": {"raw": 80000, "jobs": 95, "emp": "Emory, CDC, Rollins SPH"}, "marketing": {"raw": 72000, "jobs": 120, "emp": "Coca-Cola, UPS, Home Depot"}, "psychology": {"raw": 58000, "jobs": 70, "emp": "Grady, Emory Healthcare"}, "polisci": {"raw": 68000, "jobs": 75, "emp": "GA State Gov, Consulates"}, "design": {"raw": 72000, "jobs": 90, "emp": "The Home Depot, Southpaw"}},
    "Los Angeles, CA":    {"col": 172, "cs": {"raw": 145000, "jobs": 210, "emp": "Snap, SpaceX, Riot Games"}, "nursing": {"raw": 98000, "jobs": 100, "emp": "Cedars-Sinai, Kaiser LA"}, "finance": {"raw": 115000, "jobs": 140, "emp": "JPM, City National Bank"}, "mecheng": {"raw": 120000, "jobs": 110, "emp": "SpaceX, Boeing, Northrop"}, "education": {"raw": 70000, "jobs": 55, "emp": "LAUSD, UCLA"}, "biology": {"raw": 105000, "jobs": 110, "emp": "UCLA Health, Amgen, BioAtla"}, "marketing": {"raw": 95000, "jobs": 160, "emp": "NBCUniversal, Disney, Warner"}, "psychology": {"raw": 72000, "jobs": 80, "emp": "UCLA, Cedars-Sinai"}, "polisci": {"raw": 80000, "jobs": 75, "emp": "LA City, Film Unions, Law"}, "design": {"raw": 100000, "jobs": 140, "emp": "Disney, Mattel, Beats"}},
    "Miami, FL":          {"col": 120, "cs": {"raw": 110000, "jobs": 135, "emp": "Chewy, Carnival Tech, World Fuel"}, "nursing": {"raw": 65000, "jobs": 105, "emp": "Jackson Health, Baptist Health"}, "finance": {"raw": 105000, "jobs": 155, "emp": "Bank of America, KKR (Miami)"}, "mecheng": {"raw": 95000, "jobs": 70, "emp": "Spirit Airlines, Harris Corp"}, "education": {"raw": 48000, "jobs": 55, "emp": "Miami-Dade Schools, FIU"}, "biology": {"raw": 78000, "jobs": 80, "emp": "UM Miller, Jackson Health"}, "marketing": {"raw": 72000, "jobs": 115, "emp": "Royal Caribbean, Lennar Corp"}, "psychology": {"raw": 58000, "jobs": 65, "emp": "Jackson Memorial, UM"}, "polisci": {"raw": 68000, "jobs": 65, "emp": "Miami-Dade Gov, OAS"}, "design": {"raw": 75000, "jobs": 95, "emp": "Telemundo, Carnival, MDCA"}},
    "Phoenix, AZ":        {"col": 106, "cs": {"raw": 112000, "jobs": 145, "emp": "Intel, GoDaddy, Microchip"}, "nursing": {"raw": 68000, "jobs": 110, "emp": "Banner Health, Dignity Health"}, "finance": {"raw": 92000, "jobs": 120, "emp": "Chase, Vanguard, American Express"}, "mecheng": {"raw": 100000, "jobs": 100, "emp": "Intel, Honeywell, Lucid Motors"}, "education": {"raw": 48000, "jobs": 60, "emp": "PVUSD, ASU, Grand Canyon Univ"}, "biology": {"raw": 78000, "jobs": 80, "emp": "Mayo Clinic AZ, Dignity Health"}, "marketing": {"raw": 70000, "jobs": 100, "emp": "Go Daddy, PetSmart, Avnet"}, "psychology": {"raw": 56000, "jobs": 65, "emp": "Banner Behavioral Health"}, "polisci": {"raw": 64000, "jobs": 55, "emp": "Arizona State Gov, ASU"}, "design": {"raw": 72000, "jobs": 88, "emp": "GoDaddy, Shamrock Foods"}},
    "Minneapolis, MN":    {"col": 110, "cs": {"raw": 118000, "jobs": 160, "emp": "Target, Best Buy, Optum"}, "nursing": {"raw": 75000, "jobs": 100, "emp": "Mayo Clinic, Allina Health"}, "finance": {"raw": 105000, "jobs": 130, "emp": "US Bancorp, Ameriprise, Allianz"}, "mecheng": {"raw": 100000, "jobs": 110, "emp": "3M, Medtronic, Emerson"}, "education": {"raw": 58000, "jobs": 60, "emp": "Minneapolis Public Schools"}, "biology": {"raw": 92000, "jobs": 105, "emp": "Medtronic, UCare, U of M"}, "marketing": {"raw": 80000, "jobs": 110, "emp": "Target, General Mills, Cargill"}, "psychology": {"raw": 64000, "jobs": 70, "emp": "Allina, Hennepin Healthcare"}, "polisci": {"raw": 72000, "jobs": 65, "emp": "MN Legislature, MSP NGOs"}, "design": {"raw": 82000, "jobs": 95, "emp": "Target, General Mills, Colle McVoy"}},
    "Pittsburgh, PA":     {"col": 97, "cs": {"raw": 110000, "jobs": 165, "emp": "Carnegie Mellon, Google, Uber ATG"}, "nursing": {"raw": 64000, "jobs": 95, "emp": "UPMC, AHN"}, "finance": {"raw": 88000, "jobs": 100, "emp": "PNC Bank, BNY Mellon"}, "mecheng": {"raw": 95000, "jobs": 110, "emp": "Westinghouse, ANSYS, Kennametal"}, "education": {"raw": 52000, "jobs": 55, "emp": "Pittsburgh City Schools, Pitt"}, "biology": {"raw": 82000, "jobs": 95, "emp": "UPMC, Pitt Med, Carnegie Mellon"}, "marketing": {"raw": 70000, "jobs": 85, "emp": "GNC, Highmark"}, "psychology": {"raw": 58000, "jobs": 68, "emp": "UPMC Western Psych, Pitt"}, "polisci": {"raw": 65000, "jobs": 55, "emp": "City of Pittsburgh, Allegheny Co."}, "design": {"raw": 72000, "jobs": 90, "emp": "CMU, Pittsburgh Ballet"}},
    "Raleigh-Durham, NC": {"col": 107, "cs": {"raw": 118000, "jobs": 170, "emp": "Cisco, IBM, Red Hat"}, "nursing": {"raw": 66000, "jobs": 95, "emp": "UNC Health, Duke Health"}, "finance": {"raw": 92000, "jobs": 105, "emp": "First Citizens Bank, Fidelity"}, "mecheng": {"raw": 98000, "jobs": 95, "emp": "Collins Aerospace, GE Aviation"}, "education": {"raw": 50000, "jobs": 60, "emp": "WCPSS, NC State, Duke"}, "biology": {"raw": 95000, "jobs": 130, "emp": "Biogen, Bayer, Duke Med"}, "marketing": {"raw": 72000, "jobs": 100, "emp": "Lenovo, SAS Institute"}, "psychology": {"raw": 60000, "jobs": 68, "emp": "UNC, Duke, Holly Hill Hospital"}, "polisci": {"raw": 68000, "jobs": 60, "emp": "NC State Gov, Research Triangle NGOs"}, "design": {"raw": 75000, "jobs": 88, "emp": "SAS, NetApp, Citrix"}},
}

# Map each school to its nearest metro area for career data
SCHOOL_TO_METRO = {
    "University of Alabama": "Atlanta, GA",
    "Auburn University": "Atlanta, GA",
    "University of Florida": "Miami, FL",
    "University of Georgia": "Atlanta, GA",
    "LSU": "Houston, TX",
    "University of Mississippi": "Atlanta, GA",
    "Mississippi State": "Atlanta, GA",
    "University of Tennessee": "Atlanta, GA",
    "Vanderbilt University": "Atlanta, GA",
    "University of Kentucky": "Atlanta, GA",
    "University of South Carolina": "Raleigh-Durham, NC",
    "University of Arkansas": "Austin, TX",
    "Texas A&M": "Houston, TX",
    "University of Missouri": "Chicago, IL",
    "University of Texas": "Austin, TX",
    "University of Oklahoma": "Austin, TX",
    "UCLA": "Los Angeles, CA",
    "UC Berkeley": "San Francisco, CA",
    "University of Michigan": "Chicago, IL",
    "University of Virginia": "Washington, DC",
    "UNC Chapel Hill": "Raleigh-Durham, NC",
    "University of Wisconsin": "Minneapolis, MN",
    "Ohio State University": "Pittsburgh, PA",
    "Penn State": "Pittsburgh, PA",
    "University of Illinois": "Chicago, IL",
    "Purdue University": "Chicago, IL",
    "Indiana University": "Chicago, IL",
    "University of Maryland": "Washington, DC",
    "University of Washington": "Seattle, WA",
    "University of Colorado Boulder": "Denver, CO",
    "Arizona State University": "Phoenix, AZ",
    "University of Arizona": "Phoenix, AZ",
    "University of Oregon": "Seattle, WA",
    "University of Minnesota": "Minneapolis, MN",
    "Rutgers University": "New York, NY",
    "Georgia Tech": "Atlanta, GA",
    "Virginia Tech": "Washington, DC",
    "NC State": "Raleigh-Durham, NC",
    "University of Pittsburgh": "Pittsburgh, PA",
    "Florida State University": "Miami, FL",
    "Clemson University": "Atlanta, GA",
    "Stanford University": "San Francisco, CA",
    "MIT": "Boston, MA",
    "Harvard University": "Boston, MA",
    "Yale University": "New York, NY",
    "Princeton University": "New York, NY",
    "Duke University": "Raleigh-Durham, NC",
    "Northwestern University": "Chicago, IL",
    "University of Chicago": "Chicago, IL",
    "Rice University": "Houston, TX",
    "Notre Dame": "Chicago, IL",
    "Boston University": "Boston, MA",
    "NYU": "New York, NY",
    "USC": "Los Angeles, CA",
    "Georgetown University": "Washington, DC",
    "Emory University": "Atlanta, GA",
    "Wake Forest University": "Raleigh-Durham, NC",
    "Tulane University": "Houston, TX",
}

# ---------------------------------------------------------------------------
# Quiz questions — each maps to one or more culture dimensions
# ---------------------------------------------------------------------------
QUIZ_QUESTIONS = [
    {
        "id": "q1",
        "question": "It's Friday night. What sounds like the best time?",
        "options": [
            {"label": "Huge house party or hitting the bars with a big group", "scores": {"social": 10, "spirit": 6}},
            {"label": "Chill hangout with close friends — board games, movie night", "scores": {"social": 4, "campus": 7}},
            {"label": "Studying at the library or working on a passion project", "scores": {"academics": 9, "career": 7}},
            {"label": "Going to a campus event — comedy show, concert, club meeting", "scores": {"social": 6, "diversity": 6, "campus": 6}},
        ],
    },
    {
        "id": "q2",
        "question": "What matters most to you in a professor?",
        "options": [
            {"label": "They challenge me and push me to think critically", "scores": {"academics": 10, "career": 6}},
            {"label": "They're approachable and actually care about students", "scores": {"academics": 7, "campus": 7}},
            {"label": "They have real industry experience and connections", "scores": {"career": 10, "academics": 5}},
            {"label": "They make class entertaining — I need to stay engaged", "scores": {"academics": 5, "social": 5}},
        ],
    },
    {
        "id": "q3",
        "question": "Pick your ideal Saturday in the fall:",
        "options": [
            {"label": "Tailgating at sunrise, face paint on, screaming in the student section", "scores": {"spirit": 10, "social": 8}},
            {"label": "Exploring the city — coffee shops, museums, street food", "scores": {"location": 10, "diversity": 6}},
            {"label": "Catching the game on TV while working on homework", "scores": {"spirit": 4, "academics": 7}},
            {"label": "Volunteering or going to a community event", "scores": {"diversity": 8, "religion": 5}},
        ],
    },
    {
        "id": "q4",
        "question": "When you think about life after college, what excites you most?",
        "options": [
            {"label": "Landing a top job — I want a strong alumni network and recruiting pipeline", "scores": {"career": 10, "academics": 6}},
            {"label": "The friendships and memories — college is about the experience", "scores": {"social": 8, "campus": 7}},
            {"label": "Grad school or research — I want to go deep in my field", "scores": {"academics": 10, "career": 5}},
            {"label": "Making a difference — I want to give back to my community", "scores": {"diversity": 8, "religion": 6}},
        ],
    },
    {
        "id": "q5",
        "question": "What kind of campus vibe do you want?",
        "options": [
            {"label": "Beautiful, classic campus with great dorms and dining", "scores": {"campus": 10, "social": 4}},
            {"label": "Right in a big city — the campus IS the city", "scores": {"location": 10, "diversity": 6}},
            {"label": "College town where everything revolves around the school", "scores": {"spirit": 8, "social": 7, "campus": 6}},
            {"label": "Doesn't matter as long as the academics are strong", "scores": {"academics": 9, "career": 7}},
        ],
    },
    {
        "id": "q6",
        "question": "How important is Greek life (fraternities/sororities) to you?",
        "options": [
            {"label": "Very — I'm definitely rushing", "scores": {"social": 10, "spirit": 6}},
            {"label": "I'd consider it but it's not make-or-break", "scores": {"social": 6, "spirit": 4}},
            {"label": "Not for me, but I don't mind if it's big on campus", "scores": {"social": 3, "academics": 5}},
            {"label": "I'd prefer a school where Greek life isn't dominant", "scores": {"social": 2, "diversity": 7, "academics": 7}},
        ],
    },
    {
        "id": "q7",
        "question": "How do you feel about diversity on campus?",
        "options": [
            {"label": "Extremely important — I want to be surrounded by people from all backgrounds", "scores": {"diversity": 10, "location": 6}},
            {"label": "Important but not my top priority", "scores": {"diversity": 6}},
            {"label": "I care more about finding my specific community or group", "scores": {"diversity": 4, "religion": 5, "social": 5}},
            {"label": "I just want a welcoming environment, however that looks", "scores": {"diversity": 5, "campus": 6}},
        ],
    },
    {
        "id": "q8",
        "question": "How do you feel about faith and religion on campus?",
        "options": [
            {"label": "Very important — I want a strong faith community", "scores": {"religion": 10, "spirit": 4}},
            {"label": "I'd like the option but don't need it to be central", "scores": {"religion": 5}},
            {"label": "Neutral — not something I think about", "scores": {"religion": 3}},
            {"label": "I prefer a more secular environment", "scores": {"religion": 1, "diversity": 6}},
        ],
    },
    {
        "id": "q9",
        "question": "You got a free elective. What are you taking?",
        "options": [
            {"label": "Something practical — business, coding, personal finance", "scores": {"career": 10, "academics": 5}},
            {"label": "Something fun — wine tasting, film studies, improv", "scores": {"social": 7, "campus": 6}},
            {"label": "Something challenging — advanced math, philosophy, research seminar", "scores": {"academics": 10}},
            {"label": "Something meaningful — social justice, environmental science, ethics", "scores": {"diversity": 8, "religion": 4}},
        ],
    },
    {
        "id": "q10",
        "question": "What's your ideal school size?",
        "options": [
            {"label": "Big — I want 40,000+ students, huge energy, endless options", "scores": {"social": 8, "spirit": 8}},
            {"label": "Medium — 15,000-30,000 feels right, not too big or small", "scores": {"campus": 6, "academics": 6}},
            {"label": "Smaller — under 15,000, where professors know my name", "scores": {"academics": 8, "campus": 8}},
            {"label": "Doesn't matter — I care about other things more", "scores": {}},
        ],
    },
]


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_profiles():
    """Load school culture profiles."""
    path = os.path.join(DATA_DIR, "school_profiles.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    # Fallback: load niche grades only and synthesize sentiment columns
    niche_path = os.path.join(DATA_DIR, "niche_grades.csv")
    df = pd.read_csv(niche_path)
    grade_map = {
        "A+": 4.3, "A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7,
        "C+": 2.3, "C": 2.0, "C-": 1.7, "D+": 1.3, "D": 1.0, "D-": 0.7, "F": 0.0,
    }
    grade_cols = [c for c in df.columns if c != "school"]
    for col in grade_cols:
        df[col] = df[col].map(grade_map) / 4.3
    df["social_sentiment"] = df["party_scene_grade"]
    df["academics_sentiment"] = df["academics_grade"]
    df["campus_sentiment"] = (df["campus_grade"] + df["food_grade"] + df["dorms_grade"]) / 3
    df["diversity_sentiment"] = df["diversity_grade"]
    df["career_sentiment"] = df["value_grade"]
    df["location_sentiment"] = df["campus_grade"]
    df["religion_sentiment"] = 0.5
    df["spirit_sentiment"] = df["athletics_grade"]
    for topic in ["social", "academics", "campus", "diversity", "career", "location", "religion", "spirit"]:
        df[f"{topic}_volume"] = 0.5
    return df


@st.cache_data
def load_quotes():
    """Load representative Reddit quotes if available."""
    path = os.path.join(DATA_DIR, "representative_quotes.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["school", "topic", "text", "sentiment", "score"])


@st.cache_data
def load_clubs():
    """Load club/organization data if available."""
    path = os.path.join(DATA_DIR, "clubs.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["school", "club_name", "category", "description"])


@st.cache_data
def fit_pca_model(profiles_json: str):
    """Fit PCA on profiles."""
    profiles = pd.read_json(profiles_json)
    features = [c for c in ALL_FEATURE_COLS if c in profiles.columns]
    X = profiles[features].fillna(0.5).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    coords = pd.DataFrame({
        "school": profiles["school"].values,
        "pc1": X_pca[:, 0],
        "pc2": X_pca[:, 1],
    })
    loadings = pd.DataFrame({
        "feature": features,
        "pc1_loading": pca.components_[0],
        "pc2_loading": pca.components_[1],
    })
    return pca, scaler, coords, loadings, features


def _pick_diverse_labels(sorted_features: pd.DataFrame, col: str, n: int, from_end: bool) -> list[str]:
    """Pick n feature labels avoiding near-duplicates (e.g. 'Academic Quality' and 'Academic Culture')."""
    if from_end:
        candidates = sorted_features["feature"].tolist()[::-1]  # highest first
    else:
        candidates = sorted_features["feature"].tolist()  # lowest first
    picked_labels: list[str] = []
    seen_roots: set[str] = set()
    for feat in candidates:
        label = FEATURE_LABELS.get(feat, feat)
        # Extract the first word as a "root" to detect duplicates like Academic Quality / Academic Culture
        root = label.split()[0].lower() if label else feat
        if root not in seen_roots:
            picked_labels.append(label)
            seen_roots.add(root)
        if len(picked_labels) >= n:
            break
    return picked_labels


def get_axis_labels(loadings: pd.DataFrame) -> tuple[str, str]:
    pc1_sorted = loadings.sort_values("pc1_loading")
    pc1_neg = _pick_diverse_labels(pc1_sorted, "pc1_loading", 2, from_end=False)
    pc1_pos = _pick_diverse_labels(pc1_sorted, "pc1_loading", 2, from_end=True)
    pc1_label = f"← {', '.join(pc1_neg)} | {', '.join(pc1_pos)} →"
    pc2_sorted = loadings.sort_values("pc2_loading")
    pc2_neg = _pick_diverse_labels(pc2_sorted, "pc2_loading", 2, from_end=False)
    pc2_pos = _pick_diverse_labels(pc2_sorted, "pc2_loading", 2, from_end=True)
    pc2_label = f"← {', '.join(pc2_neg)} | {', '.join(pc2_pos)} →"
    return pc1_label, pc2_label


def build_user_vector(survey: dict[str, float], features: list[str], profiles: pd.DataFrame) -> np.ndarray:
    """
    Build user preference vector scaled to the actual data range.
    Slider 1 = that feature's min in the dataset.
    Slider 10 = that feature's max in the dataset.
    This keeps the user point within the cloud of school points.
    """
    vec = np.full(len(features), 0.5)
    for dim, slider_val in survey.items():
        t = (slider_val - 1) / 9.0  # normalize 1-10 to 0-1
        if dim in SURVEY_TO_FEATURES:
            for feat in SURVEY_TO_FEATURES[dim]:
                if feat in features:
                    idx = features.index(feat)
                    feat_min = profiles[feat].min()
                    feat_max = profiles[feat].max()
                    # Interpolate within actual data range
                    vec[idx] = feat_min + t * (feat_max - feat_min)
    return vec


def compute_matches(user_vec: np.ndarray, profiles: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Score schools using Euclidean distance instead of cosine similarity.
    This gives much better spread in match percentages.
    """
    X = profiles[features].fillna(0.5).values
    # Euclidean distance from user to each school
    dists = np.sqrt(np.sum((X - user_vec) ** 2, axis=1))
    # Convert distance to a 0-100 match score
    # Closest school = 100%, furthest = lower bound
    max_dist = dists.max()
    min_dist = dists.min()
    if max_dist > min_dist:
        # Normalize to 0-1, invert (closer = higher), then scale to ~40-99 range
        norm_dists = (dists - min_dist) / (max_dist - min_dist)
        scores = (1 - norm_dists) * 60 + 40  # range: 40% to 100%
    else:
        scores = np.full(len(dists), 100.0)
    results = profiles.copy()
    results["match_score"] = scores
    return results.sort_values("match_score", ascending=False)


def make_radar_chart(school_row: pd.Series, features: list[str]) -> go.Figure:
    labels = [FEATURE_LABELS.get(f, f) for f in features if f in school_row.index]
    values = [school_row.get(f, 0.5) for f in features if f in school_row.index]
    labels = labels + [labels[0]]
    values = values + [values[0]]
    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=labels, fill="toself",
        fillcolor="rgba(99, 110, 250, 0.2)",
        line=dict(color="rgba(99, 110, 250, 0.8)"),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=250,
    )
    return fig


def get_strengths(school_name, profiles, features):
    row = profiles[profiles["school"] == school_name]
    if len(row) == 0:
        return ""
    row = row.iloc[0]
    scores = {FEATURE_LABELS.get(f, f): row.get(f, 0) for f in features}
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return ", ".join(n for n, _ in top)


def quiz_scores_to_survey(quiz_answers: dict[str, int]) -> dict[str, float]:
    """
    Convert quiz answers into dimension scores (1-10 scale).
    Each answer contributes weighted scores to dimensions.
    We average all contributions per dimension.
    """
    dim_totals: dict[str, list[float]] = {
        "social": [], "academics": [], "campus": [], "diversity": [],
        "career": [], "location": [], "religion": [], "spirit": [],
    }

    for q in QUIZ_QUESTIONS:
        qid = q["id"]
        if qid in quiz_answers:
            chosen_idx = quiz_answers[qid]
            if 0 <= chosen_idx < len(q["options"]):
                option_scores = q["options"][chosen_idx]["scores"]
                for dim, val in option_scores.items():
                    if dim in dim_totals:
                        dim_totals[dim].append(val)

    # Average scores per dimension, default to 5 if no data
    survey = {}
    for dim, values in dim_totals.items():
        if values:
            survey[dim] = sum(values) / len(values)
        else:
            survey[dim] = 5.0

    return survey


# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="College Culture Matcher",
    page_icon="🎓",
    layout="wide",
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "landing"  # landing | quiz | results | tangibles
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_complete" not in st.session_state:
    st.session_state.quiz_complete = False
if "survey_from_quiz" not in st.session_state:
    st.session_state.survey_from_quiz = None

# Load data (shared across pages)
profiles = load_profiles()
quotes_df = load_quotes()
clubs_df = load_clubs()
features = [c for c in ALL_FEATURE_COLS if c in profiles.columns]
pca_model, scaler, pca_coords, loadings, pca_features = fit_pca_model(profiles.to_json())
pc1_label, pc2_label = get_axis_labels(loadings)


# ---------------------------------------------------------------------------
# PAGE: Landing — choose Tangibles or Intangibles
# ---------------------------------------------------------------------------
def render_landing():
    st.markdown("""
    <style>
    .landing-title {
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        margin-top: 40px;
        margin-bottom: 5px;
    }
    .landing-sub {
        text-align: center;
        font-size: 1.15rem;
        color: #888;
        margin-bottom: 50px;
    }
    .path-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #333;
        border-radius: 16px;
        padding: 35px 25px;
        text-align: center;
        transition: all 0.2s ease;
        min-height: 320px;
    }
    .path-card:hover {
        border-color: #7c6cfa;
        transform: translateY(-2px);
    }
    .path-icon { font-size: 3.5rem; margin-bottom: 15px; }
    .path-name {
        font-family: 'Arial', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 10px;
        color: #ffffff;
    }
    .path-desc {
        font-size: 0.95rem;
        color: #ffffff;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="landing-title">🎓 College Culture Matcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-sub">Find your perfect college fit — by the numbers or by the vibe.</div>', unsafe_allow_html=True)

    col1, spacer, col2 = st.columns([1, 0.2, 1])

    with col1:
        st.markdown("""
        <div class="path-card">
            <div class="path-icon">📊</div>
            <div class="path-name">Tangibles</div>
            <div class="path-desc">
                Explore <b>salary, job markets, cost of living</b>, and top employers
                by major across 16 US metro areas. See where the jobs are for your field.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📊 Explore Career Data", use_container_width=True, type="primary"):
            st.session_state.page = "tangibles"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="path-card">
            <div class="path-icon">🎭</div>
            <div class="path-name">Intangibles</div>
            <div class="path-desc">
                Take a <b>10-question culture quiz</b> and get matched to schools based on
                social life, academics, diversity, campus feel, and more.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎭 Take the Culture Quiz", use_container_width=True, type="primary"):
            st.session_state.page = "quiz"
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#666;font-size:0.85rem;'>"
        "Built for <b>Bama Builds Hackathon</b> — Data Science Track<br>"
        "You can switch between Tangibles and Intangibles at any time."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# PAGE: Tangibles — MajorMap interactive map
# ---------------------------------------------------------------------------
def render_tangibles():
    # Switch button in top corner
    col_back, col_title, col_switch = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Home"):
            st.session_state.page = "landing"
            st.rerun()
    with col_title:
        st.title("📊 MajorMap — Career Data by Major")
    with col_switch:
        if st.button("🎭 Switch to Intangibles"):
            st.session_state.page = "quiz" if not st.session_state.quiz_complete else "results"
            st.rerun()

    # Load and embed the MajorMap HTML
    majormap_path = os.path.join(APP_DIR, "college-major-map.html")
    if os.path.exists(majormap_path):
        with open(majormap_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=False)
    else:
        st.error("MajorMap file not found. Make sure `college-major-map.html` is in the project root.")

    st.caption("Salary adjusted for regional cost of living. Circle size reflects job postings relative to metro population. Click a circle for details.")


# ---------------------------------------------------------------------------
# PAGE: Quiz
# ---------------------------------------------------------------------------
def render_quiz():
    col_back, col_title, col_switch = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Home", key="quiz_home"):
            st.session_state.page = "landing"
            st.rerun()
    with col_switch:
        if st.button("📊 Switch to Tangibles", key="quiz_tangibles"):
            st.session_state.page = "tangibles"
            st.rerun()

    st.title("🎓 College Culture Matcher")
    st.subheader("Find your perfect college fit in 10 questions")
    st.markdown("Answer honestly — there are no right or wrong answers. We'll match you to schools that fit *your* vibe.")
    st.markdown("---")

    # Progress bar
    answered = len(st.session_state.quiz_answers)
    total_q = len(QUIZ_QUESTIONS)
    st.progress(answered / total_q, text=f"Question {min(answered + 1, total_q)} of {total_q}")

    # Render each question
    for i, q in enumerate(QUIZ_QUESTIONS):
        qid = q["id"]
        st.markdown(f"### Q{i + 1}. {q['question']}")

        option_labels = [opt["label"] for opt in q["options"]]

        # Use radio buttons for selection
        choice = st.radio(
            f"Select your answer for Q{i + 1}:",
            options=range(len(option_labels)),
            format_func=lambda x, labels=option_labels: labels[x],
            key=f"quiz_{qid}",
            index=st.session_state.quiz_answers.get(qid, None),
            label_visibility="collapsed",
        )

        if choice is not None:
            st.session_state.quiz_answers[qid] = choice

        st.markdown("---")

    # Submit button
    all_answered = len(st.session_state.quiz_answers) == total_q

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if all_answered:
            if st.button("🎯 Find My Matches!", type="primary", use_container_width=True):
                st.session_state.survey_from_quiz = quiz_scores_to_survey(st.session_state.quiz_answers)
                st.session_state.quiz_complete = True
                st.session_state.page = "results"
                st.rerun()
        else:
            remaining = total_q - len(st.session_state.quiz_answers)
            st.info(f"Answer all questions to see your matches. {remaining} remaining.")


# ---------------------------------------------------------------------------
# PAGE: Results
# ---------------------------------------------------------------------------
def render_results():
    # Determine survey scores — from quiz or from sidebar sliders
    if st.session_state.survey_from_quiz:
        survey = st.session_state.survey_from_quiz.copy()
    else:
        survey = {
            "social": 5, "academics": 7, "campus": 6, "diversity": 5,
            "career": 7, "location": 5, "religion": 3, "spirit": 5,
        }

    # --- Sidebar: Fine-tune sliders + structural filters ---
    with st.sidebar:
        st.header("Fine-Tune Your Preferences")
        if st.session_state.quiz_complete:
            st.caption("Sliders pre-set from your quiz answers. Adjust to explore.")
        st.markdown("---")

        survey["social"] = st.slider("🎉 Social / Party Scene", 1, 10, int(round(survey["social"])), key="s_social")
        survey["academics"] = st.slider("📚 Academic Rigor", 1, 10, int(round(survey["academics"])), key="s_academics")
        survey["campus"] = st.slider("🏛️ Campus & Facilities", 1, 10, int(round(survey["campus"])), key="s_campus")
        survey["diversity"] = st.slider("🌍 Diversity & Inclusion", 1, 10, int(round(survey["diversity"])), key="s_diversity")
        survey["career"] = st.slider("💼 Career Outcomes", 1, 10, int(round(survey["career"])), key="s_career")
        survey["location"] = st.slider("📍 City vs College Town", 1, 10, int(round(survey["location"])), key="s_location")
        survey["religion"] = st.slider("⛪ Religious Community", 1, 10, int(round(survey["religion"])), key="s_religion")
        survey["spirit"] = st.slider("🏈 School Spirit & Athletics", 1, 10, int(round(survey["spirit"])), key="s_spirit")

        st.markdown("---")
        st.subheader("Structural Preferences")
        size_pref = st.selectbox(
            "Preferred School Size",
            ["No Preference", "Small (<5k)", "Medium (5k-15k)", "Large (15k-30k)", "Very Large (30k+)"],
        )
        region_pref = st.multiselect(
            "Region Preference",
            ["Northeast", "Southeast", "Midwest", "Southwest", "West Coast"],
            default=[],
        )

        st.markdown("---")
        st.subheader("💲 Tuition Budget")
        tuition_type = st.radio(
            "Residency Status",
            ["Out-of-State", "In-State"],
            horizontal=True,
            help="Private schools charge the same regardless of residency.",
        )
        tuition_key = "tuition_out" if tuition_type == "Out-of-State" else "tuition_in"
        max_tuition = st.slider(
            "Maximum Annual Tuition",
            min_value=5000,
            max_value=70000,
            value=70000,
            step=1000,
            format="$%d",
            help="Filter out schools above your budget. Slide left to set a cap.",
        )

        st.markdown("---")
        st.subheader("Career Outlook")
        major_options = ["None selected"] + [f"{m['icon']} {m['label']}" for m in MAJORS]
        selected_major_label = st.selectbox("Your Intended Major", major_options, index=0)
        if selected_major_label != "None selected":
            # Extract the major id from the label
            selected_major_id = None
            for m in MAJORS:
                if m["label"] in selected_major_label:
                    selected_major_id = m["id"]
                    break
        else:
            selected_major_id = None

        st.markdown("---")
        if st.button("📊 Switch to Tangibles", use_container_width=True):
            st.session_state.page = "tangibles"
            st.rerun()
        if st.button("🔄 Retake Quiz", use_container_width=True):
            st.session_state.quiz_answers = {}
            st.session_state.quiz_complete = False
            st.session_state.survey_from_quiz = None
            st.session_state.page = "quiz"
            st.rerun()
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()

        with st.expander("ℹ️ How It Works"):
            st.markdown("""
            **College Culture Matcher** analyzes real student opinions to build
            culture profiles for major US universities.

            **Data Sources:**
            - **Reddit**: Sentiment analysis of posts and comments from school subreddits
            - **Niche.com**: Letter grades across academics, campus life, diversity, etc.
            - **Rate My Professors**: Professor quality aggregates
            - **CampusLabs Engage**: Student organization/club directories
            - **MajorMap Career Data**: Salary, job volume, COL, and employers by major across 16 metro areas

            **Methodology:**
            1. Reddit text is filtered by topic keywords and scored with VADER sentiment analysis
            2. Scores are normalized and combined with Niche grades into 17-dimension culture vectors
            3. PCA reduces dimensions for the scatter visualization
            4. Euclidean distance matches your preferences to each school's profile
            5. Selecting a region preference **boosts** scores for schools in that region
            6. Tuition filter lets you set a budget cap (in-state or out-of-state)
            7. Career outlook adjusts salary for cost of living in each school's metro area

            *Built for Bama Builds Hackathon — Data Science Track*
            """)

    # --- Main content ---
    st.title("🎓 Your College Culture Matches")

    # Show user's culture profile summary
    if st.session_state.quiz_complete:
        with st.expander("📊 Your Culture Profile", expanded=True):
            # Build a radar chart of the user's preferences
            user_labels = ["Social", "Academics", "Campus", "Diversity", "Career", "Location", "Religion", "Spirit"]
            user_dims = ["social", "academics", "campus", "diversity", "career", "location", "religion", "spirit"]
            user_vals = [survey[d] / 10.0 for d in user_dims]
            user_vals_closed = user_vals + [user_vals[0]]
            user_labels_closed = user_labels + [user_labels[0]]

            fig_user = go.Figure(data=go.Scatterpolar(
                r=user_vals_closed, theta=user_labels_closed, fill="toself",
                fillcolor="rgba(255, 193, 7, 0.3)",
                line=dict(color="rgba(255, 152, 0, 0.9)", width=2),
                name="Your Preferences",
            ))
            fig_user.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                margin=dict(l=60, r=60, t=30, b=30),
                height=300,
            )
            st.plotly_chart(fig_user, use_container_width=True)

            # Top priorities callout
            sorted_dims = sorted(survey.items(), key=lambda x: x[1], reverse=True)
            top_3 = sorted_dims[:3]
            dim_emojis = {
                "social": "🎉", "academics": "📚", "campus": "🏛️", "diversity": "🌍",
                "career": "💼", "location": "📍", "religion": "⛪", "spirit": "🏈",
            }
            dim_names = {
                "social": "Social Life", "academics": "Academics", "campus": "Campus",
                "diversity": "Diversity", "career": "Career", "location": "Location",
                "religion": "Faith", "spirit": "School Spirit",
            }
            top_str = "   |   ".join(
                f"{dim_emojis.get(d, '')} **{dim_names.get(d, d)}** ({v:.0f}/10)"
                for d, v in top_3
            )
            st.markdown(f"**Your top priorities:** {top_str}")

    # Compute matches
    user_vec = build_user_vector(survey, features, profiles)
    matched = compute_matches(user_vec, profiles, features)

    # Apply structural filters
    SIZE_MAP = {
        "Small (<5k)": "Small",
        "Medium (5k-15k)": "Medium",
        "Large (15k-30k)": "Large",
        "Very Large (30k+)": "Very Large",
    }
    if size_pref != "No Preference":
        target_size = SIZE_MAP.get(size_pref)
        matched = matched[matched["school"].apply(
            lambda s: SCHOOL_META.get(s, {}).get("size") == target_size
        )]
    # Apply region preference as a score boost (not a hard filter)
    if region_pref:
        REGION_BOOST = 12  # points added to match score for preferred regions
        matched["match_score"] = matched.apply(
            lambda row: min(100.0, row["match_score"] + REGION_BOOST)
            if SCHOOL_META.get(row["school"], {}).get("region") in region_pref
            else row["match_score"],
            axis=1,
        )
        matched = matched.sort_values("match_score", ascending=False)
    # Apply tuition filter
    if max_tuition < 70000:
        matched = matched[matched["school"].apply(
            lambda s: SCHOOL_META.get(s, {}).get(tuition_key, 70000) <= max_tuition
        )]

    # PCA projection
    user_scaled = scaler.transform(user_vec.reshape(1, -1))
    user_pca = pca_model.transform(user_scaled)[0]
    all_matched = compute_matches(user_vec, profiles, features)
    # Apply region boost to all_matched as well (consistent with filtered results)
    if region_pref:
        all_matched["match_score"] = all_matched.apply(
            lambda row: min(100.0, row["match_score"] + REGION_BOOST)
            if SCHOOL_META.get(row["school"], {}).get("region") in region_pref
            else row["match_score"],
            axis=1,
        )
        all_matched = all_matched.sort_values("match_score", ascending=False)
    coords_with_scores = pca_coords.merge(all_matched[["school", "match_score"]], on="school")
    coords_with_scores["city"] = coords_with_scores["school"].map(
        lambda s: SCHOOL_META.get(s, {}).get("city", "")
    )
    coords_with_scores["enrollment"] = coords_with_scores["school"].map(
        lambda s: SCHOOL_META.get(s, {}).get("enrollment", 0)
    )
    coords_with_scores["strengths"] = coords_with_scores["school"].apply(
        lambda s: get_strengths(s, profiles, features)
    )

    # --- Visualization ---
    viz_mode = st.radio(
        "Visualization Mode",
        ["🗺️ Overall Fit Map (PCA)", "📊 Explore Dimensions"],
        horizontal=True,
    )

    if viz_mode == "🗺️ Overall Fit Map (PCA)":
        # Identify top 5 best-matching schools to label on the map
        top_5_best = coords_with_scores.nlargest(5, "match_score")

        fig = px.scatter(
            coords_with_scores, x="pc1", y="pc2",
            color="match_score",
            color_continuous_scale=["#ff4444", "#ffaa00", "#44bb44"],
            range_color=[coords_with_scores["match_score"].min(), coords_with_scores["match_score"].max()],
            hover_name="school",
            hover_data={"match_score": ":.1f", "city": True, "strengths": True, "pc1": False, "pc2": False},
            labels={"pc1": pc1_label, "pc2": pc2_label, "match_score": "Match %"},
            title="School Culture Map — Top 5 Best Matches Labeled",
        )

        # Add labels as annotations spread far apart in different directions
        positions = [
            {"ax": 120, "ay": -80},
            {"ax": -130, "ay": -70},
            {"ax": 140, "ay": 70},
            {"ax": -120, "ay": 80},
            {"ax": 0, "ay": -100},
        ]
        for i, (_, row) in enumerate(top_5_best.iterrows()):
            pos = positions[i % len(positions)]
            fig.add_annotation(
                x=row["pc1"], y=row["pc2"],
                text=f"<b>{row['school']}</b><br>{row['match_score']:.0f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=0.8,
                arrowwidth=1.5,
                arrowcolor="#555",
                ax=pos["ax"], ay=pos["ay"],
                bordercolor="#555",
                borderwidth=1,
                borderpad=3,
                bgcolor="rgba(255,255,255,0.9)",
                font=dict(family="Arial, sans-serif", size=11, color="#222"),
            )

        # Add user ideal point
        fig.add_trace(go.Scatter(
            x=[user_pca[0]], y=[user_pca[1]],
            mode="markers",
            marker=dict(size=20, color="gold", symbol="star", line=dict(width=2, color="black")),
            name="Your Ideal", hoverinfo="text", hovertext="Your ideal school profile",
        ))
        fig.add_annotation(
            x=user_pca[0], y=user_pca[1],
            text="<b>Your Ideal</b>",
            showarrow=True, arrowhead=2, arrowcolor="goldenrod",
            ax=0, ay=-35,
            font=dict(family="Arial, sans-serif", size=13, color="goldenrod"),
            bordercolor="goldenrod", borderwidth=1.5, borderpad=4,
            bgcolor="rgba(255,255,255,0.95)",
        )

        fig.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=60, b=20),
            font=dict(family="Arial, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        dim_options = {FEATURE_LABELS.get(f, f): f for f in features}
        with col1:
            x_label = st.selectbox("X-Axis Dimension", list(dim_options.keys()), index=0)
        with col2:
            y_label = st.selectbox("Y-Axis Dimension", list(dim_options.keys()), index=1)
        x_col = dim_options[x_label]
        y_col = dim_options[y_label]
        top_10_schools = matched.head(10)["school"].tolist()
        plot_df = profiles.copy()
        plot_df["match_score"] = all_matched.set_index("school")["match_score"]
        plot_df["is_top_10"] = plot_df["school"].isin(top_10_schools)
        plot_df["highlight"] = plot_df["is_top_10"].map({True: "Top 10 Match", False: "Other"})
        plot_df["city"] = plot_df["school"].map(lambda s: SCHOOL_META.get(s, {}).get("city", ""))
        fig = px.scatter(
            plot_df, x=x_col, y=y_col,
            color="highlight", color_discrete_map={"Top 10 Match": "#44bb44", "Other": "#cccccc"},
            hover_name="school",
            hover_data={"match_score": ":.1f", "city": True, "highlight": False},
            labels={x_col: x_label, y_col: y_label, "match_score": "Match %"},
            title=f"{x_label} vs {y_label}",
        )
        user_x = user_vec[features.index(x_col)] if x_col in features else 0.5
        user_y = user_vec[features.index(y_col)] if y_col in features else 0.5
        fig.add_trace(go.Scatter(
            x=[user_x], y=[user_y],
            mode="markers+text",
            marker=dict(size=18, color="gold", symbol="star", line=dict(width=2, color="black")),
            text=["You"], textposition="top center", name="Your Preference",
        ))
        fig.update_layout(height=600, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # --- Results Panel ---
    st.markdown("---")
    st.header("🏆 Your Top Matches")

    if len(matched) == 0:
        st.warning("No schools match your structural filters. Try widening your preferences.")
    else:
        top_schools = matched.head(10)
        for rank, (_, school_row) in enumerate(top_schools.iterrows(), 1):
            school_name = school_row["school"]
            match_pct = school_row["match_score"]
            meta = SCHOOL_META.get(school_name, {})

            with st.expander(
                f"**#{rank} — {school_name}** ({match_pct:.1f}% match)",
                expanded=(rank <= 3),
            ):
                col_info, col_radar = st.columns([1, 1])
                with col_info:
                    st.markdown(f"**Location:** {meta.get('city', 'N/A')}")
                    st.markdown(f"**Enrollment:** {meta.get('enrollment', 'N/A'):,}")
                    st.markdown(f"**Size:** {meta.get('size', 'N/A')}")
                    st.markdown(f"**Region:** {meta.get('region', 'N/A')}")
                    # Tuition display
                    t_in = meta.get("tuition_in")
                    t_out = meta.get("tuition_out")
                    if t_in and t_out:
                        if t_in == t_out:
                            st.markdown(f"**Tuition:** ${t_in:,}/yr (private)")
                        else:
                            st.markdown(f"**Tuition:** ${t_in:,} in-state · ${t_out:,} out-of-state")
                    strengths = get_strengths(school_name, profiles, features)
                    st.markdown(f"**Top Strengths:** {strengths}")
                    if "avg_rating" in school_row and pd.notna(school_row.get("avg_rating")):
                        st.markdown(f"**Avg Professor Rating:** {school_row['avg_rating']:.2f}")

                    # Career tangibles (if major selected)
                    if selected_major_id and school_name in SCHOOL_TO_METRO:
                        metro_name = SCHOOL_TO_METRO[school_name]
                        metro = METRO_DATA.get(metro_name, {})
                        major_data = metro.get(selected_major_id, {})
                        if major_data:
                            adj_salary = round(major_data["raw"] / (metro["col"] / 100))
                            major_label = next((m["label"] for m in MAJORS if m["id"] == selected_major_id), "")
                            st.markdown(f"**💰 {major_label} near {metro_name}:**")
                            st.markdown(
                                f"&nbsp;&nbsp;Adj. Salary: **${adj_salary:,}** · "
                                f"Jobs/100k: **{major_data['jobs']}** · "
                                f"COL: **{metro['col']}/100**"
                            )
                            st.caption(f"&nbsp;&nbsp;Top employers: {major_data['emp']}")

                    # Links
                    links = []
                    if meta.get("url"):
                        links.append(f"[🌐 Website]({meta['url']})")
                    if meta.get("clubs_url"):
                        links.append(f"[🏛️ Clubs & Orgs]({meta['clubs_url']})")
                    if links:
                        st.markdown(" · ".join(links))

                with col_radar:
                    radar_fig = make_radar_chart(school_row, features)
                    st.plotly_chart(radar_fig, use_container_width=True, key=f"radar_{school_name}")

                if len(quotes_df) > 0:
                    school_quotes = quotes_df[quotes_df["school"] == school_name]
                    if len(school_quotes) > 0:
                        st.markdown("**What students say:**")
                        for _, q in school_quotes.head(4).iterrows():
                            emoji = "👍" if q["sentiment"] == "positive" else "👎"
                            st.markdown(
                                f"{emoji} *({q['topic']})* \"{q['text'][:200]}...\" "
                                f"(⬆ {q['score']})"
                            )

    # --- Club / Organization Search ---
    st.markdown("---")
    st.header("🏛️ Club & Organization Finder")

    if len(clubs_df) == 0:
        st.info("Club data hasn't been scraped yet. Run `python data/scrape_clubs.py` to populate this section.")
    else:
        st.markdown("Search for a club, sport, or activity and see which schools have it — ranked by your culture match.")

        club_search = st.text_input(
            "Search for a club or activity",
            placeholder="e.g. ice skating, debate, robotics, chess, volleyball...",
            key="club_search",
        )

        if club_search and len(club_search.strip()) >= 2:
            query = club_search.strip().lower()
            # Search club names and descriptions for the query
            mask = (
                clubs_df["club_name"].str.lower().str.contains(query, na=False) |
                clubs_df["description"].str.lower().str.contains(query, na=False)
            )
            results = clubs_df[mask].copy()

            if len(results) == 0:
                st.warning(f"No clubs found matching **\"{club_search}\"** across any school. Try a broader term (e.g., \"skating\" instead of \"figure skating\").")
            else:
                # Summary: how many schools have this
                schools_with_club = results["school"].nunique()
                total_matches = len(results)
                total_schools_in_data = clubs_df["school"].nunique()
                st.success(
                    f"Found **{total_matches}** matching clubs/organizations across "
                    f"**{schools_with_club}** of {total_schools_in_data} schools with club data."
                )

                # Group by school, attach match score, sort by best match
                school_club_counts = (
                    results.groupby("school")
                    .agg(
                        num_matching=("club_name", "count"),
                        club_list=("club_name", lambda x: list(x)),
                    )
                    .reset_index()
                )

                # Merge in the match score so we can rank by best fit
                match_scores = all_matched[["school", "match_score"]].drop_duplicates()
                school_club_counts = school_club_counts.merge(match_scores, on="school", how="left")
                school_club_counts["match_score"] = school_club_counts["match_score"].fillna(0)
                school_club_counts = school_club_counts.sort_values("match_score", ascending=False)

                # Best-fit schools with this club
                st.markdown("#### Best Matches With This Club")
                st.caption("Schools ranked by how well they fit your culture preferences.")
                for rank, (_, row) in enumerate(school_club_counts.iterrows(), 1):
                    meta = SCHOOL_META.get(row["school"], {})
                    city = meta.get("city", "")
                    score = row["match_score"]
                    clubs_preview = row["club_list"][:5]
                    clubs_str = ", ".join(clubs_preview)
                    if len(row["club_list"]) > 5:
                        clubs_str += f" (+{len(row['club_list']) - 5} more)"

                    with st.expander(
                        f"**#{rank} — {row['school']}** ({city}) — {score:.0f}% match — "
                        f"{row['num_matching']} matching club(s)",
                        expanded=(rank <= 3),
                    ):
                        # Show match score bar
                        st.progress(score / 100, text=f"Culture match: {score:.1f}%")

                        # Links
                        club_links = []
                        if meta.get("url"):
                            club_links.append(f"[🌐 Website]({meta['url']})")
                        if meta.get("clubs_url"):
                            club_links.append(f"[🏛️ Browse All Clubs]({meta['clubs_url']})")
                        if club_links:
                            st.markdown(" · ".join(club_links))

                        # List matching clubs
                        for club_name in sorted(row["club_list"]):
                            club_row = results[
                                (results["school"] == row["school"]) &
                                (results["club_name"] == club_name)
                            ]
                            if len(club_row) > 0:
                                cat = club_row.iloc[0].get("category", "")
                                cat = str(cat) if pd.notna(cat) else ""
                                desc = club_row.iloc[0].get("description", "")
                                desc = str(desc) if pd.notna(desc) else ""
                                cat_str = f" *({cat})*" if cat else ""
                                st.markdown(f"- **{club_name}**{cat_str}")
                                if desc.strip() and len(desc.strip()) > 10:
                                    clean_desc = re.sub(r"<[^>]+>", " ", desc)
                                    clean_desc = re.sub(r"&[a-zA-Z]+;", " ", clean_desc)
                                    clean_desc = re.sub(r"\s+", " ", clean_desc).strip()
                                    if len(clean_desc) > 10:
                                        st.caption(f"  {clean_desc[:200]}")

                # Show schools WITHOUT this club (from those that have club data)
                schools_without = set(clubs_df["school"].unique()) - set(results["school"].unique())
                if schools_without:
                    with st.expander(f"Schools without matching clubs ({len(schools_without)} schools)"):
                        st.markdown(
                            ", ".join(sorted(schools_without))
                        )

                # Note about schools without club data
                all_schools = set(SCHOOL_META.keys())
                schools_no_data = all_schools - set(clubs_df["school"].unique())
                if schools_no_data:
                    st.caption(
                        f"Note: Club data unavailable for {len(schools_no_data)} schools "
                        f"({', '.join(sorted(list(schools_no_data)[:5]))}{'...' if len(schools_no_data) > 5 else ''})."
                    )

    # --- Career Outlook Section ---
    st.markdown("---")
    st.header("💰 Career Outlook by Major")

    if selected_major_id is None:
        st.info("Select your intended major in the sidebar to see salary, job market, and employer data for your top matches.")
    else:
        major_label = next((f"{m['icon']} {m['label']}" for m in MAJORS if m["id"] == selected_major_id), "")
        st.markdown(f"Showing career data for **{major_label}** in metro areas near your top matched schools.")
        st.caption("Salaries are adjusted for cost of living. Job volume = postings per 100k residents.")

        # Build career comparison table for top matched schools
        if len(matched) > 0:
            career_rows = []
            seen_metros = set()
            for _, school_row in matched.head(15).iterrows():
                school_name = school_row["school"]
                metro_name = SCHOOL_TO_METRO.get(school_name)
                if not metro_name or metro_name in seen_metros:
                    continue
                metro = METRO_DATA.get(metro_name, {})
                major_data = metro.get(selected_major_id, {})
                if major_data:
                    adj_salary = round(major_data["raw"] / (metro["col"] / 100))
                    # Which schools map to this metro?
                    schools_in_metro = [s for s, m in SCHOOL_TO_METRO.items() if m == metro_name and s in set(matched["school"])]
                    career_rows.append({
                        "Metro Area": metro_name,
                        "Schools Nearby": ", ".join(schools_in_metro[:3]),
                        "Adj. Salary": adj_salary,
                        "Raw Salary": major_data["raw"],
                        "Jobs/100k": major_data["jobs"],
                        "COL Index": metro["col"],
                        "Top Employers": major_data["emp"],
                    })
                    seen_metros.add(metro_name)

            if career_rows:
                career_df = pd.DataFrame(career_rows)
                career_df = career_df.sort_values("Adj. Salary", ascending=False)

                # Bar chart comparing adjusted salaries
                fig_salary = px.bar(
                    career_df,
                    x="Metro Area",
                    y="Adj. Salary",
                    color="Adj. Salary",
                    color_continuous_scale=["#ff6b6b", "#ffd93d", "#6bcb77"],
                    hover_data={"Schools Nearby": True, "Jobs/100k": True, "COL Index": True, "Top Employers": True},
                    labels={"Adj. Salary": "Adjusted Salary ($)"},
                    title=f"Cost-of-Living Adjusted {major_label.split(' ', 1)[-1]} Salary by Metro",
                )
                fig_salary.update_layout(
                    height=400,
                    font=dict(family="Arial, sans-serif"),
                    showlegend=False,
                    xaxis_tickangle=-30,
                )
                st.plotly_chart(fig_salary, use_container_width=True)

                # Detailed table
                st.markdown("#### Detailed Comparison")
                display_df = career_df.copy()
                display_df["Adj. Salary"] = display_df["Adj. Salary"].apply(lambda x: f"${x:,}")
                display_df["Raw Salary"] = display_df["Raw Salary"].apply(lambda x: f"${x:,}")
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Metro Area": st.column_config.TextColumn("Metro Area", width="medium"),
                        "Schools Nearby": st.column_config.TextColumn("Your Matches Nearby", width="large"),
                        "Adj. Salary": st.column_config.TextColumn("Adj. Salary", width="small"),
                        "Raw Salary": st.column_config.TextColumn("Raw Salary", width="small"),
                        "Jobs/100k": st.column_config.NumberColumn("Jobs/100k", width="small"),
                        "COL Index": st.column_config.NumberColumn("COL Index", width="small"),
                        "Top Employers": st.column_config.TextColumn("Top Employers", width="large"),
                    },
                )
            else:
                st.warning("No career data available for your matched schools' metro areas.")


# ---------------------------------------------------------------------------
# Page router
# ---------------------------------------------------------------------------
if st.session_state.page == "landing":
    render_landing()
elif st.session_state.page == "tangibles":
    render_tangibles()
elif st.session_state.page == "quiz" and not st.session_state.quiz_complete:
    render_quiz()
else:
    render_results()
