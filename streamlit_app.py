# streamlit_app.py
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Monthly Review – All Verticals (Composite + Zones)", layout="wide")
st.title("📊 All Verticals in ONE Graph — Composite Trend with Red / Watch / Healthy Zones")

st.write("""
Upload your **pivot-style CSV** (row 0 = Verticals, row 1 = Metrics, rows 2+ = Months like Jan25).
Map your 6 KPIs once. We’ll compute a **Composite KPI Score (0–100)** for each vertical, per month, and plot **all verticals together** with traffic-light zones.
""")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

# Month labels like Jan25, Feb25, etc.
MONTH_PAT = re.compile(r'^[A-Za-z]{3}\d{2}$')

# The six KPIs we’ll use in the composite (you map your file’s metric names to these)
TARGET_KPIS = [
    "AVERAGE of Course completion %",
    "AVERAGE of NPS",
    "SUM of No of Placements(Monthly)",
    "AVERAGE of Reg to Placement %",
    "AVERAGE of Active Student %",
    "AVERAGE of Avg Mentor Rating",
]

# Default raw thresholds for each KPI (editable in sidebar)
DEFAULT_THRESHOLDS = {
    "AVERAGE of Course completion %": (50, 70),    # low, healthy
    "AVERAGE of NPS": (30, 60),
    "SUM of No of Placements(Monthly)": (10, 30),  # counts
    "AVERAGE of Reg to Placement %": (30, 60),
    "AVERAGE of Active Student %": (50, 75),
    "AVERAGE of Avg Mentor Rating": (3.5, 4.2),    # out of 5
}

# Composite zone bands (fixed): Red 0–50, Watch 50–75, Healthy 75–100
RED_BAND = (0, 50)
WATCH_BAND = (50, 75)
HEALTHY_BAND = (75, 100)

# -------------
# Parse pivot
# -------------
def parse_pivot(file) -> pd.DataFrame:
    """
    Parse pivot-style CSV into tidy long DF:
      Month | Vertical | Metric | Value
    Row 0: Vertical names (sparse → forward-filled)
    Row 1: Metric names
    Row 2+: Month + values
    """
    raw = pd.read_csv(file, header=None)

    if raw.shape[0] < 3 or raw.shape[1] < 2:
        return pd.DataFrame(columns=["Month", "Vertical", "Metric", "Value"])

    vertical_row = raw.iloc[0].astype(str).str.strip().replace({'nan': np.nan})
    metric_row   = raw.iloc[1].astype(str).str.strip().replace({'nan': np.nan})

    # Forward-fill verticals across columns
    verticals = []
    last_v = None
    for v in vertical_row:
        if pd.notna(v) and v != "":
            last_v = v
        verticals.append(last_v)

    records = []
    for j in range(1, raw.shape[1]):          # col 0 is months
        vertical = verticals[j]
        metric   = metric_row[j]
        if pd.isna(metric) and pd.isna(vertical):
            continue
        for i in range(2, raw.shape[0]):
            month = str(raw.iloc[i, 0]).strip()
            if not month or (month.lower() in ("nan", "none")):
                continue
            if not MONTH_PAT.match(month):
                continue
            val = raw.iloc[i, j]
            if isinstance(val, str):
                val = val.replace('%', '').replace(',', '').strip()
            try:
                val = float(val)
            except Exception:
                val = np.nan
            records.append([month, vertical, metric, val])

    df = pd.DataFrame(records, columns=["Month", "Vertical", "Metric", "Value"])
    df = df[df["Metric"].notna()].copy()
    return df

# -------------
# KPI helpers
# -------------
def kpi_guess_options(all_metrics):
    key_map = {
        "AVERAGE of Course completion %": ["completion"],
        "AVERAGE of NPS": ["nps"],
        "SUM of No of Placements(Monthly)": ["placement", "placements"],
        "AVERAGE of Reg to Placement %": ["reg to placement"],
        "AVERAGE of Active Student %": ["active student"],
        "AVERAGE of Avg Mentor Rating": ["mentor", "rating"],
    }
    preselect = {}
    for k, needles in key_map.items():
        picks = [m for m in all_metrics if any(n in m.lower() for n in needles)]
        preselect[k] = picks
    return preselect

def aggregate_by_vertical(tidy: pd.DataFrame, selections: dict) -> pd.DataFrame:
    """
    Build DF: Month | Vertical | KPI | Value (raw scales, per vertical)
      - Placements -> sum (within vertical/month if multiple columns selected)
      - Others     -> mean
    """
    rows = []
    for kpi, metric_names in selections.items():
        if not metric_names:
            continue
        sdf = tidy[tidy["Metric"].isin(metric_names)].copy()
        if sdf.empty:
            continue
        if "placement" in kpi.lower():
            grouped = sdf.groupby(["Month", "Vertical"], sort=False)["Value"].sum().reset_index()
        else:
            grouped = sdf.groupby(["Month", "Vertical"], sort=False)["Value"].mean().reset_index()
        grouped["KPI"] = kpi
        rows.append(grouped)

    if not rows:
        return pd.DataFrame(columns=["Month", "Vertical", "KPI", "Value"])

    out = pd.concat(rows, ignore_index=True)
    # Preserve month order
    cats = list(out["Month"].drop_duplicates())
    out["Month"] = pd.Categorical(out["Month"], categories=cats, ordered=True)
    out = out.sort_values(["KPI", "Vertical", "Month"])
    return out

# -------------
# Composite scoring (0–100) using thresholds
# -------------
def scale_with_thresholds(kpi_name: str, raw_values: pd.Series, low: float, healthy: float) -> pd.Series:
    """
    Map raw KPI values → 0..100 using two points:
      <= low      → 0 .. ~ (linear up to)
      >= healthy  → 100 (cap)
      between     → linear interpolation
    """
    s = raw_values.astype(float)
    # Handle degenerate thresholds
    if healthy is None or low is None or healthy <= low:
        return s * 0  # fallback, yields zeros

    # linear scale
    scaled = (s - low) / (healthy - low) * 100.0
    scaled = scaled.clip(lower=0, upper=100)
    return scaled

def compute_composite(vert_df: pd.DataFrame, thresholds: dict, weights: dict) -> pd.DataFrame:
    """
    Input DF: Month | Vertical | KPI | Value (raw)
    Output DF: Month | Vertical | Composite (0..100)
    Steps:
      1) For each KPI, scale raw values to 0..100 with its (low, healthy) thresholds
      2) Weighted mean across the 6 KPI scores → Composite
    """
    df = vert_df.copy()
    df["Score"] = np.nan

    for kpi, (lo, hi) in thresholds.items():
        mask = df["KPI"] == kpi
        if mask.any():
            df.loc[mask, "Score"] = scale_with_thresholds(kpi, df.loc[mask, "Value"], lo, hi)

    # If some KPIs missing for a vertical/month, average the available ones (weighting only what exists)
    # Pivot to sum weights per row
    df["Weight"] = df["KPI"].map(weights).fillna(0.0)
    # Keep only rows that got a score
    df_scored = df.dropna(subset=["Score"]).copy()

    # Normalize weights within each Month/Vertical (only across present KPIs)
    def _weighted_mean(g):
        w = g["Weight"].values
        x = g["Score"].values
        if w.sum() == 0:
            return np.nanmean(x) if len(x) else np.nan
        return np.average(x, weights=w)

    comp = (
        df_scored.groupby(["Month", "Vertical"], sort=False)
        .apply(_weighted_mean)
        .reset_index(name="Composite")
    )

    # Keep order of months
    cats = list(vert_df["Month"].drop_duplicates())
    comp["Month"] = pd.Categorical(comp["Month"], categories=cats, ordered=True)
    comp = comp.sort_values(["Vertical", "Month"])
    return comp

# -------------
# Plot: all verticals on one chart with zones
# -------------
def plot_all_verticals_one_chart(comp_df: pd.DataFrame):
    """
    Single chart: X = Month, Y = Composite (0..100), line per Vertical.
    Background zones: Red (0-50), Watch (50-75), Healthy (75-100).
    """
    if comp_df.empty:
        st.error("Composite table is empty. Check KPI mapping and thresholds.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Zone bands
    ax.axhspan(RED_BAND[0], RED_BAND[1], color="red", alpha=0.10, label="_redzone_")
    ax.axhspan(WATCH_BAND[0], WATCH_BAND[1], color="yellow", alpha=0.10, label="_watchzone_")
    ax.axhspan(HEALTHY_BAND[0], HEALTHY_BAND[1], color="green", alpha=0.10, label="_healthzone_")

    # Plot each vertical
    months = comp_df["Month"].cat.categories.tolist() if hasattr(comp_df["Month"], "categories") else comp_df["Month"].unique().tolist()
    for vertical in comp_df["Vertical"].unique():
        s = comp_df[comp_df["Vertical"] == vertical].set_index("Month")["Composite"]
        ax.plot(months, s.reindex(months), marker='o', label=vertical)

    ax.set_title("All Verticals — Composite KPI Trend (0–100)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Composite Score (0–100)")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([str(m) for m in months], rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)

# -------------
# Report
# -------------
def classify_band(score: float) -> str:
    if pd.isna(score):
        return "–"
    if score < RED_BAND[1]:
        return "🟥 Red Zone"
    if score < WATCH_BAND[1]:
        return "🟨 Watch Zone"
    return "🟩 Healthy Zone"

def build_vertical_composite_report(comp_df: pd.DataFrame) -> str:
    """
    Per vertical:
      - Start, End, Change
      - Average composite
      - Highest/Lowest month
      - Current Zone
      - Trend direction
    """
    lines = ["ALL VERTICALS — COMPOSITE REPORT", "="*60, ""]
    for vertical in comp_df["Vertical"].unique():
        v = comp_df[comp_df["Vertical"] == vertical].sort_values("Month")
        s = v["Composite"].astype(float)
        if s.empty:
            continue
        start_val, end_val = s.iloc[0], s.iloc[-1]
        change = end_val - start_val
        trend = "↑ Improving" if change > 0 else "↓ Declining" if change < 0 else "→ Stable"
        avg_val = s.mean()
        hi_idx = s.idxmax(); lo_idx = s.idxmin()
        hi_month = v.loc[hi_idx, "Month"]; lo_month = v.loc[lo_idx, "Month"]
        zone = classify_band(end_val)

        lines.append(f"**{vertical}** — {trend} | Current: {end_val:.1f} ({zone})")
        lines.append(f"  Start {start_val:.1f} → End {end_val:.1f} (Δ {change:+.1f}) | Avg {avg_val:.1f}")
        lines.append(f"  High {s.max():.1f} in {hi_month} | Low {s.min():.1f} in {lo_month}")
    return "\n".join(lines)

# -------------
# Sidebar controls (thresholds + weights)
# -------------
st.sidebar.header("Composite Settings")

thresholds = {}
weights = {}
for kpi in TARGET_KPIS:
    st.sidebar.subheader(kpi)
    default_lo, default_hi = DEFAULT_THRESHOLDS[kpi]
    lo = st.sidebar.number_input(f"{kpi} — Red↔Watch threshold (Low)", value=float(default_lo))
    hi = st.sidebar.number_input(f"{kpi} — Watch↔Healthy threshold (High)", value=float(default_hi))
    if hi <= lo:
        st.sidebar.warning("High must be greater than Low")
    thresholds[kpi] = (lo, hi)

    w = st.sidebar.slider(f"Weight for {kpi}", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    weights[kpi] = w

# -------------
# App flow
# -------------
if uploaded:
    tidy = parse_pivot(uploaded)
    if tidy.empty:
        st.error("Parsed 0 rows. Check export: row0=verticals, row1=metrics, rows 2+=months (e.g., Jan25).")
        st.stop()

    st.subheader("Detected metrics in your file")
    all_metrics = sorted(tidy["Metric"].dropna().unique().tolist())
    st.write(all_metrics)

    st.markdown("### Map your KPIs to the metric names in the file")
    guesses = kpi_guess_options(all_metrics)
    selections = {}
    for k in TARGET_KPIS:
        selections[k] = st.multiselect(f"Select columns for **{k}**", options=all_metrics, default=guesses.get(k, []))

    if st.button("Generate SINGLE GRAPH (All Verticals) + Report", type="primary"):
        # 1) Raw per-vertical KPI values
        vert_raw = aggregate_by_vertical(tidy, selections)
        st.caption(f"Tidy rows: {len(tidy):,} | Per-vertical KPI rows: {len(vert_raw):,}")

        if vert_raw.empty:
            st.error("Nothing aggregated. Adjust KPI selections or check your CSV.")
        else:
            # 2) Composite 0..100 per vertical/month
            comp_df = compute_composite(vert_raw, thresholds, weights)

            st.subheader("📈 All Verticals — ONE Chart (Composite 0–100)")
            plot_all_verticals_one_chart(comp_df)

            st.subheader("📝 Report (All Verticals — Composite)")
            report_txt = build_vertical_composite_report(comp_df)
            st.code(report_txt)
            st.download_button(
                "⬇️ Download composite_report.txt",
                report_txt.encode("utf-8"),
                file_name="composite_report.txt",
                mime="text/plain"
            )
else:
    st.info("Upload the pivot CSV to continue.")
