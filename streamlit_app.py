# app.py
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Monthly Review – Simple Analysis", layout="wide")
st.title("📊 Monthly Review – All Verticals per KPI with Red / Watch / Healthy Zones")
st.write("Upload your **pivot-style CSV** (row 0 = Verticals, row 1 = KPIs, rows 2+ = Months like Jan25).")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

# Month labels like Jan25, Feb25, etc.
MONTH_PAT = re.compile(r'^[A-Za-z]{3}\d{2}$')

# ----- Zone thresholds (edit to your targets) -----
THRESHOLDS = {
    "AVERAGE of Course completion %": (50, 70),          # (<50 Red, 50–70 Watch, >70 Healthy)
    "AVERAGE of NPS": (30, 60),
    "SUM of No of Placements(Monthly)": (10, 30),        # counts
    "AVERAGE of Reg to Placement %": (30, 60),
    "AVERAGE of Active Student %": (50, 75),
    "AVERAGE of Avg Mentor Rating": (3.5, 4.2),          # out of 5
}
TARGET_KPIS = list(THRESHOLDS.keys())

# -----------------------
# Parse pivot → tidy long
# -----------------------
def parse_pivot(file) -> pd.DataFrame:
    """
    Return tidy long DF: Month | Vertical | KPI | Value
    Row 0: Vertical names (sparse → forward-filled)
    Row 1: KPI names
    Row 2+: Month + values
    """
    raw = pd.read_csv(file, header=None)

    if raw.shape[0] < 3 or raw.shape[1] < 2:
        return pd.DataFrame(columns=["Month","Vertical","KPI","Value"])

    vertical_row = raw.iloc[0].astype(str).str.strip().replace({'nan': np.nan})
    metric_row   = raw.iloc[1].astype(str).str.strip().replace({'nan': np.nan})

    # forward-fill vertical names across columns
    verticals = []
    last_v = None
    for v in vertical_row:
        if pd.notna(v) and v != "":
            last_v = v
        verticals.append(last_v)

    records = []
    for j in range(1, raw.shape[1]):   # col 0 is months
        vertical = verticals[j]
        kpi      = metric_row[j]
        if pd.isna(kpi) and pd.isna(vertical):
            continue
        for i in range(2, raw.shape[0]):
            month = str(raw.iloc[i, 0]).strip()
            if not month or (month.lower() in ("nan","none")):
                continue
            if not MONTH_PAT.match(month):
                continue
            val = raw.iloc[i, j]
            if isinstance(val, str):
                val = val.replace('%','').replace(',','').strip()
            try:
                val = float(val)
            except Exception:
                val = np.nan
            records.append([month, vertical, kpi, val])

    df = pd.DataFrame(records, columns=["Month","Vertical","KPI","Value"])
    return df.dropna(subset=["KPI"]).reset_index(drop=True)

# -----------------------
# KPI selection helpers
# -----------------------
def kpi_guess_options(all_metrics):
    # Suggest likely matches; you can adjust in UI
    key_map = {
        "AVERAGE of Course completion %": ["completion"],
        "AVERAGE of NPS": ["nps"],
        "SUM of No of Placements(Monthly)": ["placement", "placements"],
        "AVERAGE of Reg to Placement %": ["reg to placement"],
        "AVERAGE of Active Student %": ["active student"],
        "AVERAGE of Avg Mentor Rating": ["mentor", "rating"],
    }
    pre = {}
    for k, needles in key_map.items():
        picks = [m for m in all_metrics if any(n in m.lower() for n in needles)]
        pre[k] = picks
    return pre

# -----------------------
# Aggregate per KPI & vertical (NO normalization)
# -----------------------
def agg_vertical_kpi(tidy: pd.DataFrame, selections: dict) -> pd.DataFrame:
    """
    Return Month | Vertical | KPI | Value (raw)
      - Placements -> sum if multiple cols selected
      - Others     -> mean
    """
    out = []
    for kpi, cols in selections.items():
        if not cols:
            continue
        sub = tidy[tidy["KPI"].isin(cols)].copy()
        if sub.empty:
            continue
        if "placement" in kpi.lower():
            g = sub.groupby(["Month","Vertical"], sort=False)["Value"].sum().reset_index()
        else:
            g = sub.groupby(["Month","Vertical"], sort=False)["Value"].mean().reset_index()
        g["KPI"] = kpi
        out.append(g)
    if not out:
        return pd.DataFrame(columns=["Month","Vertical","KPI","Value"])
    df = pd.concat(out, ignore_index=True)
    cats = list(df["Month"].drop_duplicates())
    df["Month"] = pd.Categorical(df["Month"], categories=cats, ordered=True)
    return df.sort_values(["KPI","Vertical","Month"])

# -----------------------
# Plot one KPI (all verticals) with zones
# -----------------------
def plot_kpi_all_verticals(df: pd.DataFrame, kpi: str):
    sub = df[df["KPI"] == kpi]
    if sub.empty:
        st.warning(f"No data for KPI: {kpi}")
        return
    piv = sub.pivot(index="Month", columns="Vertical", values="Value").sort_index()
    fig, ax = plt.subplots(figsize=(12,6))
    low, high = THRESHOLDS[kpi]
    ymax = np.nanmax(piv.values) if np.isfinite(np.nanmax(piv.values)) else high
    ymax = max(ymax, high)

    # zones
    ax.axhspan(0, low,  color="red",    alpha=0.10, label="_red_")
    ax.axhspan(low, high, color="yellow", alpha=0.10, label="_watch_")
    ax.axhspan(high, ymax, color="green",  alpha=0.10, label="_healthy_")

    for v in piv.columns:
        ax.plot(piv.index.tolist(), piv[v], marker='o', label=v)

    ax.set_title(f"{kpi} — All Verticals (Raw Values)")
    ax.set_xlabel("Month"); ax.set_ylabel("Value")
    ax.set_xticks(range(len(piv.index)))
    ax.set_xticklabels([str(m) for m in piv.index], rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.02,1), loc='upper left')
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)

# -----------------------
# Simple summary (latest month per vertical & KPI)
# -----------------------
def quick_summary(df: pd.DataFrame):
    st.subheader("📌 Quick Zone Summary (latest month per vertical & KPI)")
    def last_row(g):
        return g.sort_values("Month").iloc[-1]
    latest = df.groupby(["KPI","Vertical"], group_keys=False).apply(last_row)

    # Build a plain-text report and show it
    lines = []
    for kpi in THRESHOLDS.keys():
        lines.append(f"### {kpi}")
        low, high = THRESHOLDS[kpi]
        rows = latest[latest["KPI"] == kpi]
        if rows.empty:
            lines.append("  (no rows for this KPI after aggregation)")
            continue
        for _, r in rows.iterrows():
            val = r["Value"]
            if pd.isna(val):
                zone = "–"
            elif val < low:
                zone = "🟥 Red Zone"
            elif val < high:
                zone = "🟨 Watch Zone"
            else:
                zone = "🟩 Healthy Zone"
            lines.append(f"- {r['Vertical']}: {zone} (Latest: {val:.2f})")
        lines.append("")
    report_text = "\n".join(lines)
    st.code(report_text)
    st.download_button("⬇️ Download summary.txt", report_text.encode("utf-8"),
                       file_name="summary.txt", mime="text/plain")

# -----------------------
# App flow
# -----------------------
if uploaded:
    tidy = parse_pivot(uploaded)
    if tidy.empty:
        st.error("Parsed 0 rows. Confirm CSV has row0=verticals, row1=KPIs, and month labels like Jan25 in first column.")
        st.stop()

    st.subheader("Detected metrics")
    all_m = sorted(tidy["KPI"].dropna().unique().tolist())
    st.write(all_m)

    st.markdown("### Map each target KPI to the metric names from your file")
    guesses = {
        # simple keyword guesses:
        "AVERAGE of Course completion %": [m for m in all_m if "completion" in m.lower()],
        "AVERAGE of NPS": [m for m in all_m if "nps" in m.lower()],
        "SUM of No of Placements(Monthly)": [m for m in all_m if "placement" in m.lower()],
        "AVERAGE of Reg to Placement %": [m for m in all_m if "reg to placement" in m.lower()],
        "AVERAGE of Active Student %": [m for m in all_m if "active student" in m.lower()],
        "AVERAGE of Avg Mentor Rating": [m for m in all_m if ("mentor" in m.lower() or "rating" in m.lower())],
    }
    selections = {}
    for k in TARGET_KPIS:
        selections[k] = st.multiselect(f"Select columns for **{k}**", options=all_m, default=guesses.get(k, []))

    if st.button("Generate Charts + Summary", type="primary"):
        agg = agg_vertical_kpi(tidy, selections)
        st.caption(f"Tidy rows: {len(tidy):,} | Aggregated rows: {len(agg):,}")
        if agg.empty:
            st.error("Nothing aggregated. Adjust KPI selections or check your CSV.")
        else:
            for k in TARGET_KPIS:
                if k in agg["KPI"].unique():
                    st.subheader(f"📈 {k}")
                    plot_kpi_all_verticals(agg, k)
            quick_summary(agg)
else:
    st.info("Upload the pivot CSV to continue.")
