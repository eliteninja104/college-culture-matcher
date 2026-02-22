"""
Application Volume Trends — The "Shotgunning" Problem
Visuals for Bama Builds Hackathon presentation.

Tells the story: Students are applying to more schools than ever based on
prestige and rankings, not cultural fit. This drives acceptance rates down
and anxiety up — which is exactly the problem College Culture Matcher solves.

Run with:
    streamlit run application_trends.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="The Shotgunning Problem — College Application Trends",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# DATA: Application volumes & acceptance rates (~2014 vs ~2024)
# Sources: Common App reports, IPEDS, university admissions offices, news
# ---------------------------------------------------------------------------

# Applications per student over time (Common App data)
apps_per_student = pd.DataFrame({
    "Year": ["2013-14", "2014-15", "2015-16", "2016-17", "2017-18",
             "2018-19", "2019-20", "2020-21", "2021-22", "2022-23",
             "2023-24", "2024-25"],
    "Apps Per Student": [4.63, 4.72, 4.88, 5.01, 5.18,
                         5.32, 5.48, 5.62, 6.22, 6.41,
                         6.65, 6.80],
})

# Total Common App applications (millions)
total_apps = pd.DataFrame({
    "Year": ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"],
    "Total Applications (M)": [5.7, 5.9, 7.1, 8.6, 9.5, 10.2],
    "Unique Applicants (M)": [1.04, 1.08, 1.18, 1.33, 1.43, 1.50],
})

# School-level data: applications received & acceptance rates
# Compiled from university admissions offices, Common Data Sets, news reports
school_data = pd.DataFrame([
    # --- Elite Private (Ivies + peers) ---
    {"school": "Harvard",         "type": "Elite Private",    "apps_2014": 34295,  "apps_2024": 54008,  "rate_2014": 5.9,  "rate_2024": 3.6},
    {"school": "Yale",            "type": "Elite Private",    "apps_2014": 30932,  "apps_2024": 57000,  "rate_2014": 6.3,  "rate_2024": 3.7},
    {"school": "Princeton",       "type": "Elite Private",    "apps_2014": 26641,  "apps_2024": 32836,  "rate_2014": 7.3,  "rate_2024": 4.0},
    {"school": "Stanford",        "type": "Elite Private",    "apps_2014": 42167,  "apps_2024": 56000,  "rate_2014": 5.1,  "rate_2024": 3.6},
    {"school": "MIT",             "type": "Elite Private",    "apps_2014": 18357,  "apps_2024": 26000,  "rate_2014": 7.7,  "rate_2024": 4.0},
    {"school": "Duke",            "type": "Elite Private",    "apps_2014": 32506,  "apps_2024": 50927,  "rate_2014": 11.0, "rate_2024": 4.8},
    {"school": "Northwestern",    "type": "Elite Private",    "apps_2014": 32500,  "apps_2024": 53000,  "rate_2014": 13.1, "rate_2024": 7.0},
    {"school": "U. of Chicago",   "type": "Elite Private",    "apps_2014": 30369,  "apps_2024": 38800,  "rate_2014": 8.8,  "rate_2024": 5.2},
    {"school": "Rice",            "type": "Elite Private",    "apps_2014": 16500,  "apps_2024": 36777,  "rate_2014": 15.1, "rate_2024": 7.8},
    {"school": "Notre Dame",      "type": "Elite Private",    "apps_2014": 17600,  "apps_2024": 35401,  "rate_2014": 21.2, "rate_2024": 9.0},

    # --- Selective Private ---
    {"school": "Vanderbilt",      "type": "Selective Private", "apps_2014": 28700,  "apps_2024": 47000,  "rate_2014": 11.6, "rate_2024": 5.6},
    {"school": "Georgetown",      "type": "Selective Private", "apps_2014": 20500,  "apps_2024": 26841,  "rate_2014": 17.0, "rate_2024": 12.0},
    {"school": "Emory",           "type": "Selective Private", "apps_2014": 18500,  "apps_2024": 37855,  "rate_2014": 25.4, "rate_2024": 14.5},
    {"school": "Wake Forest",     "type": "Selective Private", "apps_2014": 10800,  "apps_2024": 20000,  "rate_2014": 33.4, "rate_2024": 21.5},
    {"school": "Tulane",          "type": "Selective Private", "apps_2014": 30000,  "apps_2024": 32609,  "rate_2014": 26.0, "rate_2024": 14.0},
    {"school": "NYU",             "type": "Selective Private", "apps_2014": 52727,  "apps_2024": 120000, "rate_2014": 32.0, "rate_2024": 7.7},
    {"school": "Boston Univ.",    "type": "Selective Private", "apps_2014": 55000,  "apps_2024": 78000,  "rate_2014": 34.5, "rate_2024": 11.0},
    {"school": "USC",             "type": "Selective Private", "apps_2014": 51800,  "apps_2024": 83488,  "rate_2014": 17.8, "rate_2024": 10.4},

    # --- Flagship Public (most relevant to our user base) ---
    {"school": "U. of Alabama",   "type": "Flagship Public",  "apps_2014": 32000,  "apps_2024": 56795,  "rate_2014": 57.0, "rate_2024": 77.0},
    {"school": "Auburn",          "type": "Flagship Public",  "apps_2014": 19000,  "apps_2024": 55056,  "rate_2014": 83.0, "rate_2024": 45.0},
    {"school": "U. of Florida",   "type": "Flagship Public",  "apps_2014": 29000,  "apps_2024": 65375,  "rate_2014": 47.0, "rate_2024": 23.0},
    {"school": "U. of Georgia",   "type": "Flagship Public",  "apps_2014": 21000,  "apps_2024": 47850,  "rate_2014": 53.0, "rate_2024": 39.0},
    {"school": "Ohio State",      "type": "Flagship Public",  "apps_2014": 40000,  "apps_2024": 72800,  "rate_2014": 62.0, "rate_2024": 53.0},
    {"school": "Penn State",      "type": "Flagship Public",  "apps_2014": 52000,  "apps_2024": 88478,  "rate_2014": 56.0, "rate_2024": 54.0},
    {"school": "U. of Michigan",  "type": "Flagship Public",  "apps_2014": 46000,  "apps_2024": 98310,  "rate_2014": 32.2, "rate_2024": 15.6},
    {"school": "UCLA",            "type": "Flagship Public",  "apps_2014": 92000,  "apps_2024": 145086, "rate_2014": 18.0, "rate_2024": 8.6},
    {"school": "UC Berkeley",     "type": "Flagship Public",  "apps_2014": 78000,  "apps_2024": 126796, "rate_2014": 16.9, "rate_2024": 11.4},
    {"school": "Georgia Tech",    "type": "Flagship Public",  "apps_2014": 28000,  "apps_2024": 66895,  "rate_2014": 32.2, "rate_2024": 12.7},
    {"school": "Virginia Tech",   "type": "Flagship Public",  "apps_2014": 22000,  "apps_2024": 57622,  "rate_2014": 70.0, "rate_2024": 55.0},
    {"school": "Clemson",         "type": "Flagship Public",  "apps_2014": 20000,  "apps_2024": 61517,  "rate_2014": 51.0, "rate_2024": 38.0},
    {"school": "Purdue",          "type": "Flagship Public",  "apps_2014": 40000,  "apps_2024": 78526,  "rate_2014": 60.0, "rate_2024": 50.0},
    {"school": "Texas A&M",       "type": "Flagship Public",  "apps_2014": 34000,  "apps_2024": 54905,  "rate_2014": 68.0, "rate_2024": 57.0},
    {"school": "U. of Wisconsin", "type": "Flagship Public",  "apps_2014": 32000,  "apps_2024": 60000,  "rate_2014": 50.0, "rate_2024": 43.0},
    {"school": "U. of Illinois",  "type": "Flagship Public",  "apps_2014": 38000,  "apps_2024": 67000,  "rate_2014": 59.0, "rate_2024": 43.0},
    {"school": "U. of Washington","type": "Flagship Public",  "apps_2014": 35000,  "apps_2024": 58000,  "rate_2014": 53.0, "rate_2024": 41.0},
    {"school": "Florida State",   "type": "Flagship Public",  "apps_2014": 30000,  "apps_2024": 70000,  "rate_2014": 51.0, "rate_2024": 24.0},
])

# Computed columns
school_data["pct_growth"] = ((school_data["apps_2024"] - school_data["apps_2014"]) / school_data["apps_2014"] * 100).round(0)
school_data["rate_drop"] = (school_data["rate_2014"] - school_data["rate_2024"]).round(1)
school_data["apps_increase"] = school_data["apps_2024"] - school_data["apps_2014"]

# ---------------------------------------------------------------------------
# COLOR PALETTE
# ---------------------------------------------------------------------------
TYPE_COLORS = {
    "Elite Private": "#e74c3c",
    "Selective Private": "#f39c12",
    "Flagship Public": "#3498db",
}

# ---------------------------------------------------------------------------
# STREAMLIT LAYOUT
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.big-stat { font-size: 2.8rem; font-weight: 800; color: #e74c3c; text-align: center; margin: 0; }
.stat-label { font-size: 1rem; color: #888; text-align: center; margin-top: -5px; margin-bottom: 20px; }
.narrative { font-size: 1.1rem; line-height: 1.8; max-width: 900px; margin: auto; }
</style>
""", unsafe_allow_html=True)

st.title("📈 The Shotgunning Problem")
st.markdown("### Why students are applying to more schools than ever — and why it isn't working")
st.markdown("---")

# ---- Hero Stats ----
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<p class="big-stat">47%</p>', unsafe_allow_html=True)
    st.markdown('<p class="stat-label">Increase in apps per student<br>(2014 → 2025)</p>', unsafe_allow_html=True)
with c2:
    st.markdown('<p class="big-stat">10.2M</p>', unsafe_allow_html=True)
    st.markdown('<p class="stat-label">Total Common App submissions<br>(2024-25 — first time over 10M)</p>', unsafe_allow_html=True)
with c3:
    st.markdown('<p class="big-stat">6.8</p>', unsafe_allow_html=True)
    st.markdown('<p class="stat-label">Avg. schools applied to<br>(up from 4.6 a decade ago)</p>', unsafe_allow_html=True)
with c4:
    st.markdown('<p class="big-stat">~60%</p>', unsafe_allow_html=True)
    st.markdown('<p class="stat-label">Drop in avg. top-10 acceptance rate<br>(16% in 2006 → 6.4% today)</p>', unsafe_allow_html=True)

st.markdown("---")

# ---- Narrative Opening ----
st.markdown("""
<div class="narrative">
<p>College admissions has become an <b>arms race</b>. Students are submitting more applications
than ever — not because they've found more schools that fit them, but because they're <b>shotgunning</b>:
blasting applications at every "prestigious" name hoping something sticks.</p>

<p>The data tells a clear story: <b>more applications, lower acceptance rates, more anxiety —
and no better outcomes for students finding where they actually belong.</b></p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ====== VISUAL 1: Apps Per Student Over Time ======
st.header("📊 Students Are Applying to More Schools Every Year")

fig_apps = px.line(
    apps_per_student, x="Year", y="Apps Per Student",
    markers=True,
    labels={"Apps Per Student": "Average Applications Per Student"},
)
fig_apps.update_traces(
    line=dict(color="#e74c3c", width=3),
    marker=dict(size=10),
)
fig_apps.add_annotation(
    x="2019-20", y=5.48,
    text="COVID / Test-Optional<br>policies begin →",
    showarrow=True, arrowhead=2, arrowcolor="#888",
    ax=-80, ay=-50,
    font=dict(size=11, color="#888"),
)
fig_apps.add_annotation(
    x="2024-25", y=6.80,
    text="<b>6.8 apps/student</b>",
    showarrow=True, arrowhead=2,
    ax=0, ay=-40,
    font=dict(size=13, color="#e74c3c"),
)
fig_apps.update_layout(
    height=400,
    yaxis=dict(range=[4, 7.5], title="Avg. Apps Per Student"),
    xaxis_title="",
    font=dict(family="Arial, sans-serif"),
)
st.plotly_chart(fig_apps, use_container_width=True)

st.caption("Source: Common App End-of-Season Reports, 2013-14 through 2024-25")

# ====== VISUAL 2: Application Volume Explosion — Grouped Bar ======
st.markdown("---")
st.header("🚀 Application Volumes Have Exploded")
st.markdown("Every school in our dataset has seen massive application growth. Many have **doubled or tripled** their applicant pools in just 10 years.")

# Sort by percent growth
sorted_data = school_data.sort_values("pct_growth", ascending=True)

fig_growth = go.Figure()
fig_growth.add_trace(go.Bar(
    y=sorted_data["school"],
    x=sorted_data["apps_2014"],
    name="~2014 Applications",
    orientation="h",
    marker_color="#bdc3c7",
    text=sorted_data["apps_2014"].apply(lambda x: f"{x:,.0f}"),
    textposition="inside",
))
fig_growth.add_trace(go.Bar(
    y=sorted_data["school"],
    x=sorted_data["apps_2024"],
    name="~2024 Applications",
    orientation="h",
    marker_color=sorted_data["type"].map(TYPE_COLORS),
    text=sorted_data.apply(lambda r: f"{r['apps_2024']:,.0f} (+{r['pct_growth']:.0f}%)", axis=1),
    textposition="inside",
))
fig_growth.update_layout(
    barmode="overlay",
    height=1200,
    font=dict(family="Arial, sans-serif", size=11),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    xaxis_title="Total Applications Received",
    yaxis_title="",
    margin=dict(l=140),
)
st.plotly_chart(fig_growth, use_container_width=True)

st.caption("Gray = ~2014 applications. Colored = ~2024 applications. Red = Elite Private, Orange = Selective Private, Blue = Flagship Public.")

# ====== VISUAL 3: Acceptance Rate Decline — Slope Chart ======
st.markdown("---")
st.header("📉 Acceptance Rates Are Plummeting")
st.markdown("As applications surge but seat counts stay flat, getting in becomes harder — even at schools that were once relatively accessible.")

# Use only schools where the rate actually dropped meaningfully
rate_data = school_data[school_data["rate_drop"] > 1].sort_values("rate_drop", ascending=False).head(25)

fig_slope = go.Figure()
for _, row in rate_data.iterrows():
    color = TYPE_COLORS.get(row["type"], "#95a5a6")
    fig_slope.add_trace(go.Scatter(
        x=["2014", "2024"],
        y=[row["rate_2014"], row["rate_2024"]],
        mode="lines+markers+text",
        line=dict(color=color, width=2),
        marker=dict(size=8),
        text=[f"{row['rate_2014']:.0f}%", f"{row['rate_2024']:.1f}%"],
        textposition=["middle left", "middle right"],
        textfont=dict(size=10),
        name=row["school"],
        hovertemplate=f"<b>{row['school']}</b><br>2014: {row['rate_2014']:.1f}%<br>2024: {row['rate_2024']:.1f}%<br>Drop: {row['rate_drop']:.1f} pts<extra></extra>",
    ))
fig_slope.update_layout(
    height=700,
    showlegend=False,
    xaxis=dict(
        tickvals=["2014", "2024"],
        tickfont=dict(size=14, color="#333"),
    ),
    yaxis=dict(title="Acceptance Rate (%)", range=[0, 90]),
    font=dict(family="Arial, sans-serif"),
    annotations=[
        dict(x="2014", y=92, text="<b>~2014</b>", showarrow=False, font=dict(size=14)),
        dict(x="2024", y=92, text="<b>~2024</b>", showarrow=False, font=dict(size=14)),
    ],
)
st.plotly_chart(fig_slope, use_container_width=True)

# ====== VISUAL 4: The Biggest Movers — Scatter ======
st.markdown("---")
st.header("🎯 Who Got Hit Hardest?")
st.markdown("Schools with the biggest jump in applications saw the steepest drops in acceptance rates.")

fig_scatter = px.scatter(
    school_data,
    x="pct_growth",
    y="rate_drop",
    color="type",
    color_discrete_map=TYPE_COLORS,
    size="apps_2024",
    size_max=40,
    hover_name="school",
    hover_data={
        "apps_2014": ":,.0f",
        "apps_2024": ":,.0f",
        "rate_2014": ":.1f",
        "rate_2024": ":.1f",
        "pct_growth": ":.0f",
        "rate_drop": ":.1f",
    },
    labels={
        "pct_growth": "Application Growth (%)",
        "rate_drop": "Acceptance Rate Drop (percentage points)",
        "type": "School Type",
        "apps_2024": "2024 Applications",
    },
)
fig_scatter.update_layout(
    height=550,
    font=dict(family="Arial, sans-serif"),
)
# Annotate notable outliers
for _, row in school_data[school_data["pct_growth"] > 120].iterrows():
    fig_scatter.add_annotation(
        x=row["pct_growth"], y=row["rate_drop"],
        text=f"<b>{row['school']}</b>",
        showarrow=True, arrowhead=2, arrowcolor="#888",
        ax=30, ay=-25,
        font=dict(size=10),
    )
st.plotly_chart(fig_scatter, use_container_width=True)

st.caption("Bubble size = total 2024 applications. Schools in the top-right had the largest application surges AND steepest acceptance rate drops.")

# ====== VISUAL 5: The Numbers Breakdown Table ======
st.markdown("---")
st.header("📋 Full Data: 2014 vs 2024")

display = school_data[["school", "type", "apps_2014", "apps_2024", "pct_growth", "rate_2014", "rate_2024", "rate_drop"]].copy()
display.columns = ["School", "Type", "Apps ~2014", "Apps ~2024", "Growth %", "Rate 2014", "Rate 2024", "Rate Drop (pts)"]
display = display.sort_values("Growth %", ascending=False)
display["Apps ~2014"] = display["Apps ~2014"].apply(lambda x: f"{x:,.0f}")
display["Apps ~2024"] = display["Apps ~2024"].apply(lambda x: f"{x:,.0f}")
display["Growth %"] = display["Growth %"].apply(lambda x: f"+{x:.0f}%")
display["Rate 2014"] = display["Rate 2014"].apply(lambda x: f"{x:.1f}%")
display["Rate 2024"] = display["Rate 2024"].apply(lambda x: f"{x:.1f}%")
display["Rate Drop (pts)"] = display["Rate Drop (pts)"].apply(lambda x: f"-{x:.1f}")

st.dataframe(display, use_container_width=True, hide_index=True, height=500)

# ====== CLOSING NARRATIVE ======
st.markdown("---")
st.header("💡 So What's the Solution?")

st.markdown("""
<div class="narrative">
<p>The current system rewards <b>volume over fit</b>. Students apply to 10, 15, even 20 schools —
not because those schools match their values, but because rankings told them to.</p>

<p>This creates a vicious cycle:</p>
<ul style="font-size: 1.05rem; line-height: 2;">
    <li>📈 More applications → lower acceptance rates</li>
    <li>📉 Lower acceptance rates → more anxiety → even more "safety" applications</li>
    <li>🔄 Repeat every year, with rates getting lower and stress getting higher</li>
</ul>

<p><b>College Culture Matcher</b> breaks this cycle by helping students find schools
that actually fit their personality, values, and lifestyle — not just their GPA bracket.
When you know which schools match your <i>culture</i>, you don't need to apply to 20 schools.
You need 5 good ones.</p>

<p><b>Better matches. Fewer applications. Less stress. More belonging.</b></p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Data compiled from Common App End-of-Season Reports (2013-2025), IPEDS, university admissions offices, IvyCoach, CollegeTransitions, and institutional Common Data Sets. Built for Bama Builds Hackathon — Data Science Track.")
