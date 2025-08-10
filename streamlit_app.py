# streamlit_app.py
import re, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Monthly Review – Combined KPIs", layout="wide")
st.title("📊 Combined KPI Trends (One Visual + Full Report)")

st.write("Upload your **pivot-style CSV** (row 0 = Verticals, row 1 = Metrics, row 2+ = Months)")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

MONTH_PAT = re.compile(r'^[A-Za-z]{3}\d{2}$')  # Jan25, Feb25, ...

TARGET_KPIS = [
    "AVERAGE of Course completion %",
    "AVERAGE of NPS",
    "SUM of No of Placements(Monthly)",
    "AVERAGE of Reg to Placement %",
    "AVERAGE of Active Student %",
    "AVERAGE of Avg Mentor Rating",
]

def parse_pivot(file) -> pd.DataFrame:
    """Return tidy long df: Month | Vertical | Metric | Value."""
    raw = pd.read_csv(file, header=None)

    # Guard: need at least 3 rows and 2 cols
    if raw.shape[0] < 3 or raw.shape[1] < 2:
        return pd.DataFrame(columns=["Month","Vertical","Metric","Value"])

    vertical_row = raw.iloc[0].astype(str).str.strip().replace({'nan': np.nan})
    metric_row   = raw.iloc[1].astype(str).str.strip().replace({'nan': np.nan})

    # Forward-fill vertical names across columns
    verticals = []
    last_v = None
    for v in vertical_row:
        if pd.notna(v) and v != '':
            last_v = v
        verticals.append(last_v)

    records = []
    # Start from col 1; col 0 is the Month column
    for j in range(1, raw.shape[1]):
        vertical = verticals[j]
        metric   = metric_row[j]
        if pd.isna(metric) and pd.isna(vertical):
            continue  # skip empty block columns
        for i in range(2, raw.shape[0]):
            month = str(raw.iloc[i, 0]).strip()
            if not month or (month.lower() in ("nan","none")):
                continue
            # accept only month-like labels (prevents blank tails)
            if not MONTH_PAT.match(month):
                continue
            val = raw.iloc[i, j]
            # Clean numeric
            if isinstance(val, str):
                val = val.replace('%','').replace(',','').strip()
            try:
                val = float(val)
            except:
                val = np.nan
            records.append([month, vertical, metric, val])

    df = pd.DataFrame(records, columns=["Month","Vertical","Metric","Value"])
    # Drop rows with no metric or all-NaN Value
    df = df[df["Metric"].notna()].copy()
    return df

def kpi_guess_options(all_metrics):
    """Heuristic suggestions for each KPI from detected metric names."""
    key_map = {
        "AVERAGE of Course completion %": ["completion"],
        "AVERAGE of NPS": ["nps"],
        "SUM of No of Placements(Monthly)": ["placement", "placements"],
        "AVERAGE of Reg to Placement %": ["reg to placement"],
        "AVERAGE of Active Student %": ["active student"],
        "AVERAGE of Avg Mentor Rating": ["mentor", "rating"],
    }
    preselect = {}
    low = [m.lower() for m in all_metrics]
    for k, needles in key_map.items():
        pick = []
        for m in all_metrics:
            ml = m.lower()
            if any(needle in ml for needle in needles):
                pick.append(m)
        preselect[k] = pick
    return preselect

def aggregate_combined(df: pd.DataFrame, selections: dict) -> pd.DataFrame:
    """Aggregate across all verticals by KPI type:
       - Placements -> sum
       - Others -> mean
       Returns: Month | KPI | Value
    """
    rows = []
    for kpi_name, metric_names in selections.items():
        if not metric_names:
            continue
        sdf = df[df["Metric"].isin(metric_names)].copy()
        if sdf.empty:
            continue
        # Decide agg
        if "placement" in kpi_name.lower():
            agg = sdf.groupby("Month")["Value"].sum().reset_index()
        else:
            agg = sdf.groupby("Month")["Value"].mean().reset_index()
        agg["KPI"] = kpi_name
        rows.append(agg)
    if not rows:
        return pd.DataFrame(columns=["Month","KPI","Value"])
    out = pd.concat(rows, ignore_index=True)
    # preserve input month order
    out["Month"] = pd.Categorical(out["Month"], categories=sorted(out["Month"].unique(), key=lambda x: x),
                                  ordered=True)
    out = out.sort_values(["KPI","Month"])
    return out

def plot_combined(agg_df: pd.DataFrame, normalize=False, rating_max=5.0):
    fig, ax = plt.subplots(figsize=(14, 8))
    months = agg_df["Month"].cat.categories.tolist() if hasattr(agg_df["Month"], "categories") else agg_df["Month"].unique().tolist()

    # Prepare plotting table pivoted as columns per KPI
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
            # percent KPIs already numeric %

    for kpi in plot_df.columns:
        ax.plot(months, plot_df[kpi], marker='o', label=kpi)

    ax.set_title("Combined KPI Trends")
    ax.set_xlabel("Month")
    ax.set_ylabel("Value" + (" (normalized to %)" if normalize else ""))
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)

def build_report(agg_df: pd.DataFrame) -> str:
    lines = ["TREND REPORT", "="*50, ""]
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

if uploaded:
    tidy = parse_pivot(uploaded)
    if tidy.empty:
        st.error("Parsed 0 rows. Likely your CSV has extra blank rows/headers. Export the pivot with only the table, or share a sample.")
    else:
        st.subheader("Detected metrics in your file")
        all_metrics = sorted(tidy["Metric"].dropna().unique().tolist())
        st.write(all_metrics)

        st.markdown("### Map the 6 KPIs to detected metric names")
        guesses = kpi_guess_options(all_metrics)
        selections = {}
        for k in TARGET_KPIS:
            selections[k] = st.multiselect(f"Select columns for **{k}**", options=all_metrics, default=guesses.get(k, []))

        st.markdown("### Options")
        normalize = st.checkbox("Normalize dissimilar scales to % (ratings & placements → % scale)", value=False)
        rating_max = st.number_input("If normalizing, Mentor Rating max", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

        if st.button("Generate Combined Visual + Report", type="primary"):
            agg = aggregate_combined(tidy, selections)

            # Diagnostics
            st.caption(f"Rows parsed: {len(tidy):,} | After aggregation: {len(agg):,}")
            empty_kpis = [k for k in TARGET_KPIS if k not in agg["KPI"].unique()]
            if empty_kpis:
                st.warning(f"No data for: {', '.join(empty_kpis)}. Check your selections or CSV export.")

            if agg.empty:
                st.error("Nothing to plot. Adjust the KPI selections above.")
            else:
                st.subheader("📈 Combined Chart")
                plot_combined(agg, normalize=normalize, rating_max=rating_max)

                st.subheader("📝 Trend Report")
                report = build_report(agg)
                st.code(report)
                st.download_button("⬇️ Download trend_report.txt", report.encode("utf-8"),
                                   file_name="trend_report.txt", mime="text/plain")

else:
    st.info("Upload the pivot CSV to continue.")

    st.subheader("📝 Trend Report")
    report_text = generate_trend_report(agg_df)
    st.text(report_text)

    st.download_button("Download Trend Report", report_text, file_name="trend_report.txt")
