# streamlit_app.py
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Monthly Review – KPI by Vertical", layout="wide")
st.title("📊 KPI by Vertical – Trends with Red/Base/Healthy Zones (No Normalization)")
st.write("Upload your **pivot-style CSV** (row 0 = Verticals, row 1 = Metrics, rows 2+ = Months like Jan25).")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

# Month labels like Jan25, Feb25, etc.
MONTH_PAT = re.compile(r'^[A-Za-z]{3}\d{2}$')

# The six KPIs we’ll analyze (you map your file’s metric names to these)
TARGET_KPIS = [
    "AVERAGE of Course completion %",
    "AVERAGE of NPS",
    "SUM of No of Placements(Monthly)",
    "AVERAGE of Reg to Placement %",
    "AVERAGE of Active Student %",
    "AVERAGE of Avg Mentor Rating",
]

# Thresholds for zones (raw values; adjust as needed)
# Format: KPI -> (low_threshold, high_threshold)
ZONE_THRESHOLDS = {
    "AVERAGE of Course completion %": (50, 70),
    "AVERAGE of NPS": (30, 60),
    "SUM of No of Placements(Monthly)": (10, 30),
    "AVERAGE of Reg to Placement %": (30, 60),
    "AVERAGE of Active Student %": (50, 75),
    "AVERAGE of Avg Mentor Rating": (3.5, 4.2),
}

# -----------------------
# Parsing the pivot
# -----------------------
def parse_pivot(file) -> pd.DataFrame:
    """
    Parse pivot-style CSV into tidy long DF with columns:
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

# -----------------------
# KPI selection helpers
# -----------------------
def kpi_guess_options(all_metrics):
    """
    Suggest metric selections for each KPI based on keywords.
    You can adjust in the UI.
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
# Aggregation per KPI & vertical (NO normalization)
# -----------------------
def aggregate_by_vertical(tidy: pd.DataFrame, selections: dict) -> pd.DataFrame:
    """
    Build a DF: Month | Vertical | KPI | Value
    - For each KPI, reduce multiple matched metric columns (if selected) within each vertical/month:
        * Placements -> sum
        * Others     -> mean
    - Keep raw scales (no normalization).
    """
    out_rows = []
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
        out_rows.append(grouped)

    if not out_rows:
        return pd.DataFrame(columns=["Month", "Vertical", "KPI", "Value"])

    df = pd.concat(out_rows, ignore_index=True)
    # Preserve month order (as encountered)
    cats = list(df["Month"].drop_duplicates())
    df["Month"] = pd.Categorical(df["Month"], categories=cats, ordered=True)
    df = df.sort_values(["KPI", "Vertical", "Month"])
    return df

# -----------------------
# Plot per KPI (lines = verticals) with zones
# -----------------------
def plot_kpi_by_vertical(df_kpi: pd.DataFrame, kpi_name: str):
    """
    Plot raw values for a single KPI, with one line per Vertical.
    Adds zone shading based on ZONE_THRESHOLDS[kpi_name].
    """
    sub = df_kpi[df_kpi["KPI"] == kpi_name].copy()
    if sub.empty:
        st.warning(f"No data to plot for: {kpi_name}")
        return

    piv = sub.pivot(index="Month", columns="Vertical", values="Value")
    piv = piv.sort_index()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Zone shading (raw thresholds)
    low, high = ZONE_THRESHOLDS.get(kpi_name, (None, None))
    if low is not None and high is not None:
        ymax = np.nanmax(piv.values) if np.isfinite(np.nanmax(piv.values)) else high
        # Safeguard if ymax < high
        ymax = max(ymax, high)
        ax.axhspan(0, low, color="red", alpha=0.10, label="_redzone_")
        ax.axhspan(low, high, color="yellow", alpha=0.10, label="_basezone_")
        ax.axhspan(high, ymax, color="green", alpha=0.10, label="_healthzone_")

    # Plot a line for each Vertical
    for vertical in piv.columns:
        ax.plot(piv.index.tolist(), piv[vertical], marker='o', label=vertical)

    ax.set_title(f"{kpi_name} — Vertical Comparison (Raw Values)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Value")
    ax.set_xticks(range(len(piv.index)))
    ax.set_xticklabels([str(m) for m in piv.index], rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')  # legend outside
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)

# -----------------------
# Report (per KPI & vertical)
# -----------------------
def classify_zone(kpi_name: str, value: float) -> str:
    low, high = ZONE_THRESHOLDS.get(kpi_name, (None, None))
    if low is None or high is None or pd.isna(value):
        return "–"
    if value < low:
        return "🟥 Red Zone"
    elif value < high:
        return "🟨 Base Zone"
    else:
        return "🟩 Healthy Zone"

def build_vertical_report(df: pd.DataFrame) -> str:
    """
    Text report per KPI & Vertical:
      - start, end, change
      - average, high/low month
      - zone for latest value
    """
    lines = ["TREND REPORT BY VERTICAL", "="*60, ""]
    for kpi in df["KPI"].unique():
        lines.append(f"## {kpi}")
        kdf = df[df["KPI"] == kpi].copy()
        for vertical in kdf["Vertical"].unique():
            vdf = kdf[kdf["Vertical"] == vertical].sort_values("Month")
            s = vdf["Value"].astype(float)
            if s.empty:
                continue
            start_val, end_val = s.iloc[0], s.iloc[-1]
            change = end_val - start_val
            trend = "↑ Increasing" if change > 0 else "↓ Decreasing" if change < 0 else "→ Stable"
            avg_val = s.mean()
            high_idx = s.idxmax()
            low_idx  = s.idxmin()
            high_month = vdf.loc[high_idx, "Month"]
            low_month  = vdf.loc[low_idx,  "Month"]
            zone = classify_zone(kpi, end_val)

            lines.append(f"- **{vertical}**: {trend}")
            lines.append(f"  Start {start_val:.2f} → End {end_val:.2f} (Δ {change:+.2f}) | Avg {avg_val:.2f}")
            lines.append(f"  High {s.max():.2f} in {high_month} | Low {s.min():.2f} in {low_month} | {zone}")
        lines.append("")  # blank after each KPI block
    return "\n".join(lines)

# -----------------------
# App flow
# -----------------------
if uploaded:
    tidy = parse_pivot(uploaded)

    if tidy.empty:
        st.error("Parsed 0 rows. Check export: row0=verticals, row1=metrics, rows 2+=months (e.g., Jan25).")
        st.stop()

    st.subheader("Detected metrics in your file")
    all_metrics = sorted(tidy["Metric"].dropna().unique().tolist())
    st.write(all_metrics)

    st.markdown("### Map each KPI to the metric names from your file")
    guesses = kpi_guess_options(all_metrics)
    selections = {}
    for k in TARGET_KPIS:
        selections[k] = st.multiselect(
            f"Select columns for **{k}**",
            options=all_metrics,
            default=guesses.get(k, [])
        )

    if st.button("Generate Vertical Comparisons", type="primary"):
        agg_vert = aggregate_by_vertical(tidy, selections)

        # Diagnostics
        st.caption(f"Tidy rows: {len(tidy):,} | After vertical aggregation: {len(agg_vert):,}")
        missing = [k for k in TARGET_KPIS if k not in agg_vert["KPI"].unique()]
        if missing:
            st.warning("No aggregated data for: " + ", ".join(missing))

        if agg_vert.empty:
            st.error("Nothing to plot. Adjust KPI selections or check your CSV.")
        else:
            # One chart per KPI (lines = verticals), with zone shading
            for kpi in TARGET_KPIS:
                if kpi in agg_vert["KPI"].unique():
                    st.subheader(f"📈 {kpi}")
                    plot_kpi_by_vertical(agg_vert, kpi)

            # Report
            st.subheader("📝 Trend Report (by Vertical)")
            report_txt = build_vertical_report(agg_vert)
            st.code(report_txt)
            st.download_button(
                "⬇️ Download trend_report_by_vertical.txt",
                report_txt.encode("utf-8"),
                file_name="trend_report_by_vertical.txt",
                mime="text/plain"
            )
else:
    st.info("Upload the pivot CSV to continue.")
