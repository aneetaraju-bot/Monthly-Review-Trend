# streamlit_app.py
import re
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Monthly Review – Combined KPIs", layout="wide")
st.title("📊 Combined KPI Trends (One Visual + Full Report)")
st.write("Upload your **pivot-style CSV** (row 0 = Verticals, row 1 = Metrics, row 2+ = Months).")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

# Detect month labels like Jan25, Feb25, etc.
MONTH_PAT = re.compile(r'^[A-Za-z]{3}\d{2}$')

# The six KPIs you want to see as a single combined visual
TARGET_KPIS = [
    "AVERAGE of Course completion %",
    "AVERAGE of NPS",
    "SUM of No of Placements(Monthly)",
    "AVERAGE of Reg to Placement %",
    "AVERAGE of Active Student %",
    "AVERAGE of Avg Mentor Rating",
]

# -----------------------
# Parsing & cleaning
# -----------------------
def parse_pivot(file) -> pd.DataFrame:
    """
    Parse your pivot-style CSV into a tidy long DF:
      Month | Vertical | Metric | Value
    - Row 0: vertical names (sparse → forward-filled)
    - Row 1: metric names
    - Row 2+: months + values
    """
    raw = pd.read_csv(file, header=None)

    # Guard: need min shape
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
    # Column 0 = month; other columns are metric columns under a vertical
    for j in range(1, raw.shape[1]):
        vertical = verticals[j]
        metric   = metric_row[j]
        # skip empty columns
        if pd.isna(metric) and pd.isna(vertical):
            continue
        for i in range(2, raw.shape[0]):
            month = str(raw.iloc[i, 0]).strip()
            if not month or (month.lower() in ("nan", "none")):
                continue
            # keep only rows that look like month labels (prevents trailing junk)
            if not MONTH_PAT.match(month):
                continue
            val = raw.iloc[i, j]
            # Clean numeric (strip %, commas)
            if isinstance(val, str):
                val = val.replace('%', '').replace(',', '').strip()
            try:
                val = float(val)
            except Exception:
                val = np.nan
            records.append([month, vertical, metric, val])

    df = pd.DataFrame(records, columns=["Month", "Vertical", "Metric", "Value"])
    # Drop rows with no metric or all-NaN Value
    df = df[df["Metric"].notna()].copy()
    return df

def kpi_guess_options(all_metrics):
    """
    Suggest metric selections for each KPI based on keywords.
    You can still adjust in the UI.
    """
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

# -----------------------
# Aggregation & analysis
# -----------------------
def aggregate_combined(df: pd.DataFrame, selections: dict) -> pd.DataFrame:
    """
    Aggregate across all verticals by KPI:
      - Placements -> sum
      - Others     -> mean
    Return: Month | KPI | Value
    """
    rows = []
    for kpi_name, metric_names in selections.items():
        if not metric_names:
            continue
        sdf = df[df["Metric"].isin(metric_names)].copy()
        if sdf.empty:
            continue
        if "placement" in kpi_name.lower():
            # counts
            agg = sdf.groupby("Month", sort=False)["Value"].sum().reset_index()
        else:
            # percentages/ratings
            agg = sdf.groupby("Month", sort=False)["Value"].mean().reset_index()
        agg["KPI"] = kpi_name
        rows.append(agg)

    if not rows:
        return pd.DataFrame(columns=["Month", "KPI", "Value"])

    out = pd.concat(rows, ignore_index=True)
    # preserve month order as it appears
    cats = list(out["Month"].drop_duplicates())
    out["Month"] = pd.Categorical(out["Month"], categories=cats, ordered=True)
    out = out.sort_values(["KPI", "Month"])
    return out

def plot_combined(agg_df: pd.DataFrame, normalize=False, rating_max=5.0):
    """
    Plot one combined line chart with all KPIs.
    Optional normalization:
      - Ratings scaled to % of rating_max
      - Placements scaled to % of max placements (across months)
      - Percent KPIs kept as numeric %
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Pivot to KPI columns
    piv = agg_df.pivot(index="Month", columns="KPI", values="Value")
    plot_df = piv.copy()

    if normalize:
        for col in plot_df.columns:
            name = col.lower()
            s = plot_df[col].astype(float)
            if "rating" in name:
                plot_df[col] = (s / rating_max) * 100.0
            elif "placement" in name:
                m = np.nanmax(s.values)
                plot_df[col] = (s / m) * 100.0 if m and m > 0 else s
            # percentages already numeric %

    months = plot_df.index.tolist()
    for kpi in plot_df.columns:
        ax.plot(months, plot_df[kpi], marker='o', label=kpi)

    ax.set_title("Combined KPI Trends")
    ax.set_xlabel("Month")
    ax.set_ylabel("Value" + (" (normalized to %)" if normalize else ""))
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([str(m) for m in months], rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)

def build_report(agg_df: pd.DataFrame) -> str:
    """Create text trend report per KPI."""
    lines = ["TREND REPORT", "=" * 50, ""]
    for kpi in agg_df["KPI"].unique():
        s = agg_df[agg_df["KPI"] == kpi].sort_values("Month")["Value"].astype(float)
        if s.empty:
            continue
        start_val, end_val = s.iloc[0], s.iloc[-1]
        change = end_val - start_val
        trend = "↑ Increasing" if change > 0 else "↓ Decreasing" if change < 0 else "→ Stable"
        avg_val = s.mean()
        high_idx = s.idxmax()
        low_idx  = s.idxmin()
        high_month = agg_df.loc[high_idx, "Month"]
        low_month  = agg_df.loc[low_idx,  "Month"]

        lines.append(f"{kpi}: {trend}")
        lines.append(f"  Start: {start_val:.2f}, End: {end_val:.2f} (Change: {change:+.2f})")
        lines.append(f"  Average: {avg_val:.2f}")
        lines.append(f"  Highest: {s.max():.2f} in {high_month}")
        lines.append(f"  Lowest: {s.min():.2f} in {low_month}")
        lines.append("  ✅ Ending above average – good momentum" if end_val > avg_val
                     else "  ⚠ Ending below average – needs attention")
        lines.append("")
    return "\n".join(lines)

# Backward-compatible alias (in case older code calls this name)
def generate_trend_report(agg_df: pd.DataFrame):
    return build_report(agg_df)

# -----------------------
# App flow
# -----------------------
if uploaded:
    try:
        tidy = parse_pivot(uploaded)

        if tidy.empty:
            st.error("Parsed 0 rows. Check the export: row0=verticals, row1=metrics, row2+=months.")
            st.stop()

        st.subheader("Detected metrics in your file")
        all_metrics = sorted(tidy["Metric"].dropna().unique().tolist())
        st.write(all_metrics)

        st.markdown("### Map the 6 KPIs to detected metric names")
        guesses = kpi_guess_options(all_metrics)
        selections = {}
        for k in TARGET_KPIS:
            selections[k] = st.multiselect(
                f"Select columns for **{k}**",
                options=all_metrics,
                default=guesses.get(k, [])
            )

        st.markdown("### Options")
        normalize = st.checkbox(
            "Normalize dissimilar scales to % (ratings & placements → % scale)",
            value=False
        )
        rating_max = st.number_input(
            "If normalizing, Mentor Rating max",
            min_value=1.0, max_value=10.0, value=5.0, step=0.5
        )

        if st.button("Generate Combined Visual + Report", type="primary"):
            agg = aggregate_combined(tidy, selections)

            # Diagnostics
            st.caption(f"Rows parsed: {len(tidy):,} | After aggregation: {len(agg):,}")
            missing = [k for k in TARGET_KPIS if k not in agg["KPI"].unique()]
            if missing:
                st.warning(f"No data aggregated for: {', '.join(missing)}. "
                           f"Adjust selections or check CSV contents.")

            if agg.empty:
                st.error("Nothing to plot. Please review the KPI selections above.")
            else:
                st.subheader("📈 Combined Chart")
                plot_combined(agg, normalize=normalize, rating_max=rating_max)

                st.subheader("📝 Trend Report")
                report_text = build_report(agg)
                st.code(report_text)
                st.download_button(
                    "⬇️ Download trend_report.txt",
                    report_text.encode("utf-8"),
                    file_name="trend_report.txt",
                    mime="text/plain"
                )

    except Exception as e:
        st.error("Something went wrong while processing your file.")
        st.exception(e)  # show full stacktrace inside app for easier debugging
else:
    st.info("Upload the pivot CSV to continue.")
