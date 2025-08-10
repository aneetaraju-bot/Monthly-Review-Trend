"""
Streamlit App: Vertical Health Zone Analysis
Filename: streamlit_vertical_analysis_app.py

Features:
- Upload CSV/XLSX or use built-in sample data
- Map columns (Vertical, Region, Batch Count)
- Choose metrics, per-metric weight and lower-is-better toggle
- Aggregate by Vertical (mean / median / weighted by Batch Count)
- Compute normalized health score and classify into Red / Watch / Healthy
- Download results and optionally save outputs to 'outputs/' with date
- Ready for GitHub + Streamlit Cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime

st.set_page_config(page_title="Vertical Health Zone Analysis", layout="wide")

# ---------------- Helpers ----------------

def load_sample_data() -> pd.DataFrame:
    data = {
        "Vertical": ["Coding","Coding","Data Science","Data Science","Teaching","Teaching"],
        "Region": ["North","South","North","South","North","South"],
        "Avg Consumption": [80, 65, 75, 45, 90, 85],
        "Avg Live Participation": [70, 60, 55, 40, 95, 80],
        "Overall Batch Health": [75, 62, 65, 42, 92, 83],
        "Batch Count": [10, 8, 6, 5, 12, 11]
    }
    return pd.DataFrame(data)

def infer_numeric_columns(df: pd.DataFrame):
    # Try to coerce object-like numeric columns to numeric and detect numeric dtype
    numeric_cols = []
    for c in df.columns:
        # skip Vertical/Region that are likely categorical
        if df[c].dtype == 'object':
            # try coercion
            coerced = pd.to_numeric(df[c], errors='coerce')
            if coerced.notna().sum() > 0:
                # we won't overwrite original here; just mark as numeric candidate
                numeric_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    return numeric_cols

def aggregate_by_vertical(df: pd.DataFrame, metric_cols, agg_method: str, batch_col: str = None):
    group = df.groupby('Vertical')
    if agg_method == 'Mean':
        return group[metric_cols].mean().reset_index()
    elif agg_method == 'Median':
        return group[metric_cols].median().reset_index()
    elif agg_method == 'Weighted mean' and batch_col and batch_col in df.columns:
        # Weighted mean using batch_col weights
        def weighted_mean(g):
            w = g[batch_col].astype(float)
            # avoid division by zero
            if w.sum() == 0:
                return g[metric_cols].mean()
            return (g[metric_cols].multiply(w, axis=0)).sum() / w.sum()
        agg = group.apply(weighted_mean)
        # group.apply returns a DataFrame-like object with 'Vertical' as index; reset index
        agg = agg.reset_index()
        return agg
    else:
        return group[metric_cols].mean().reset_index()

def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series(0.5, index=s.index)
    # scale to 0-1
    return (s - s.min()) / (s.max() - s.min())

def compute_health_score(agg_df: pd.DataFrame, metric_cols, weights: dict, lower_is_better: dict):
    # Build normalized matrix, inverting metrics where lower is better
    normed = pd.DataFrame(index=agg_df.index)
    for col in metric_cols:
        s = agg_df[col].copy().astype(float)
        s_norm = normalize_series(s)
        if lower_is_better.get(col, False):
            s_norm = 1.0 - s_norm
        normed[col] = s_norm.fillna(0.0)
    w = np.array([weights.get(col, 1.0) for col in metric_cols], dtype=float)
    if w.sum() == 0:
        w = np.ones_like(w)
    scores = normed.values.dot(w) / w.sum()
    return pd.Series(scores, index=agg_df.index), normed

def categorize_by_percentile(score_series: pd.Series, red_pct: float, healthy_pct: float):
    # red_pct and healthy_pct are given as integer percentages (0-100)
    low_thresh = np.nanpercentile(score_series, red_pct)
    high_thresh = np.nanpercentile(score_series, 100 - healthy_pct)
    def cat(s):
        if s <= low_thresh:
            return 'Red'
        elif s >= high_thresh:
            return 'Healthy'
        else:
            return 'Watch'
    return score_series.apply(cat), low_thresh, high_thresh

def categorize_by_fixed(score_series: pd.Series, red_cut: float, healthy_cut: float):
    def cat(s):
        if s <= red_cut:
            return 'Red'
        elif s >= healthy_cut:
            return 'Healthy'
        else:
            return 'Watch'
    return score_series.apply(cat)

# ---------------- UI ----------------

st.title("Vertical Health Zone — Streamlit Analyzer")
st.markdown("""
Upload your CSV/XLSX and the app will:
- Aggregate metrics by `Vertical`
- Compute a normalized health score (per-metric weights & lower-is-better supported)
- Classify verticals into **Red / Watch / Healthy**
""")

# File upload
uploaded = st.file_uploader("Upload CSV or XLSX (leave empty to use sample)", type=["csv","xlsx"])
if uploaded is None:
    df = load_sample_data()
    st.info("Using sample dataset — upload your file to analyze your data.")
else:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

with st.expander("Preview raw data (first 200 rows)"):
    st.dataframe(df.head(200))

# Column mapping
st.sidebar.header("Column mapping")
cols = df.columns.tolist()
vertical_col = st.sidebar.selectbox("Vertical column", options=cols, index=cols.index("Vertical") if "Vertical" in cols else 0)
region_col = st.sidebar.selectbox("Region column (optional)", options=[None] + cols, index=0)
batch_col = st.sidebar.selectbox("Batch count column (optional, for weighted agg)", options=[None] + cols, index=0)

# Normalize numeric detection
numeric_cols = infer_numeric_columns(df)
if len(numeric_cols) == 0:
    st.error("No numeric columns detected. Ensure your metrics are numeric or can be parsed as numbers.")
    st.stop()

st.sidebar.header("Metric selection")
selected_metrics = st.sidebar.multiselect("Pick numeric metrics to include (higher is better by default):",
                                          options=numeric_cols,
                                          default=numeric_cols[:3])
if not selected_metrics:
    st.error("Select at least one metric.")
    st.stop()

# Ensure vertical column is called 'Vertical' in df for grouping
if vertical_col != "Vertical":
    df = df.rename(columns={vertical_col: "Vertical"})

# Convert selected metrics to numeric in the main df (coerce)
for c in selected_metrics:
    df[c] = pd.to_numeric(df[c], errors="coerce")

if batch_col and batch_col in df.columns:
    df[batch_col] = pd.to_numeric(df[batch_col], errors="coerce").fillna(0)

# Aggregation options
st.sidebar.header("Aggregation")
agg_method = st.sidebar.selectbox("Aggregation method", options=["Mean","Median","Weighted mean"])

# Metric weights and lower-is-better toggles
st.sidebar.header("Metric weights and direction")
weights = {}
lower_is_better = {}
for m in selected_metrics:
    weights[m] = st.sidebar.number_input(f"Weight — {m}", min_value=0.0, value=1.0, step=0.1, key=f"w_{m}")
    lower_is_better[m] = st.sidebar.checkbox(f"Lower is better — {m}", value=False, key=f"inv_{m}")

# Thresholding
st.sidebar.header("Classification thresholds")
method = st.sidebar.radio("Thresholding method:", options=["Percentile-based", "Fixed cutoffs"])
if method == "Percentile-based":
    red_pct = st.sidebar.slider("Red zone bottom percentile (e.g. 20 => bottom 20%)", min_value=0, max_value=49, value=20, step=1)
    healthy_pct = st.sidebar.slider("Healthy zone top percentile (e.g. 30 => top 30%)", min_value=0, max_value=50, value=30, step=1)
    if red_pct + healthy_pct > 99:
        st.sidebar.warning("Sum of red and healthy percentiles is large; consider lowering so bands don't overlap.")
else:
    red_cut = st.sidebar.slider("Red if score <= (0 - 1)", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
    healthy_cut = st.sidebar.slider("Healthy if score >= (0 - 1)", min_value=0.0, max_value=1.0, value=0.70, step=0.01)
    if red_cut >= healthy_cut:
        st.sidebar.warning("Red cutoff should be less than Healthy cutoff.")

# Save outputs option (useful locally)
st.sidebar.header("Output options")
save_outputs = st.sidebar.checkbox("Save output CSV to 'outputs/' (server/local)", value=False)

# Compute aggregation
agg_df = aggregate_by_vertical(df, selected_metrics, agg_method, batch_col if batch_col else None)

# Compute health scores
scores, normed = compute_health_score(agg_df, selected_metrics, weights, lower_is_better)
agg_df["Health Score"] = scores

# Categorize
if method == "Percentile-based":
    zones, low_t, high_t = categorize_by_percentile(agg_df["Health Score"], red_pct, healthy_pct)
    agg_df["Zone"] = zones
else:
    agg_df["Zone"] = categorize_by_fixed(agg_df["Health Score"], red_cut, healthy_cut)

# Sorting
agg_df = agg_df.sort_values("Health Score", ascending=True).reset_index(drop=True)

# Save file if requested
if save_outputs:
    os.makedirs("outputs", exist_ok=True)
    fname = f"outputs/vertical_health_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv"
    try:
        agg_df.to_csv(fname, index=False)
        st.sidebar.success(f"Saved output: {fname}")
    except Exception as e:
        st.sidebar.error(f"Failed to save file: {e}")

# ---------------- Results UI ----------------
left, right = st.columns((2,3))

with left:
    st.subheader("Aggregated verticals and zones")
    st.dataframe(agg_df.style.format({c: "{:.2f}" for c in selected_metrics + ["Health Score"]}), height=450)

    csv = agg_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download results CSV", csv, file_name="vertical_health_zones.csv", mime="text/csv")

with right:
    st.subheader("Visuals")
    zone_color = {"Red": "#ff4d4d", "Watch": "#ffcc00", "Healthy": "#2ecc71"}
    if not agg_df.empty:
        fig = px.bar(agg_df, x="Vertical", y="Health Score", color="Zone", color_discrete_map=zone_color,
                     title="Health Score by Vertical", text=agg_df["Health Score"].round(2))
        st.plotly_chart(fig, use_container_width=True)
        if len(selected_metrics) >= 2:
            try:
                fig2 = px.parallel_coordinates(agg_df, dimensions=selected_metrics + ["Health Score"],
                                               color="Health Score")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                st.info("Parallel coordinates plot skipped (requires numeric columns without NaNs).")

# Summary counts
counts = agg_df["Zone"].value_counts().reindex(["Red","Watch","Healthy"]).fillna(0).astype(int)
col1, col2, col3 = st.columns(3)
col1.metric("Red zones", counts["Red"])
col2.metric("Watch zones", counts["Watch"])
col3.metric("Healthy zones", counts["Healthy"])

with st.expander("How scoring works & deployment steps"):
    st.markdown("""
    **Scoring**
    - Each selected metric is normalized 0–1 across verticals.
    - If `Lower is better` is checked for a metric, it is inverted (1 - normalized).
    - Weighted sum (using weights you set) produces the Health Score (0–1).
    - Zones are assigned by percentile bands or fixed cutoffs.

    **To run locally**
    1. Install dependencies (see requirements.txt).
    2. `streamlit run streamlit_vertical_analysis_app.py`

    **To deploy on Streamlit Cloud**
    1. Put this file + requirements.txt in a GitHub repo (see README).
    2. On https://share.streamlit.io click *New app* and connect the repo.
    """)
