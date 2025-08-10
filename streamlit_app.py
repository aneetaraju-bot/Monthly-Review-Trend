# streamlit_app.py
import io
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Combined KPI Trends", layout="wide")
st.title("Combined KPI Trends – Single Visual + Full Report")

st.markdown("""
**Upload one CSV** containing your KPIs over time.  
This app will let you **pick** which column is the Month and which columns are your KPIs.
""")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

def clean_headers(cols):
    # trim spaces & unify for display, but KEEP originals for selection
    return [c.strip() for c in cols]

def coerce_numeric(series: pd.Series) -> pd.Series:
    # remove % and commas, then convert to number
    s = (series.astype(str)
               .str.replace('%', '', regex=False)
               .str.replace(',', '', regex=False)
               .str.strip())
    return pd.to_numeric(s, errors='coerce')

def generate_report(df: pd.DataFrame, month_col: str, metric_cols):
    lines = ["TREND REPORT", "="*50, ""]
    for metric in metric_cols:
        s = pd.to_numeric(df[metric], errors='coerce').dropna()
        if s.empty:
            continue
        start_val, end_val = s.iloc[0], s.iloc[-1]
        change = end_val - start_val
        trend = "↑ Increasing" if change > 0 else "↓ Decreasing" if change < 0 else "→ Stable"
        avg_val = s.mean()
        high_idx = s.idxmax()
        low_idx = s.idxmin()
        high_month = df.loc[high_idx, month_col]
        low_month = df.loc[low_idx, month_col]

        lines.append(f"{metric}: {trend}")
        lines.append(f"  Start: {start_val:.2f}, End: {end_val:.2f} (Change: {change:+.2f})")
        lines.append(f"  Average: {avg_val:.2f}")
        lines.append(f"  Highest: {s.max():.2f} in {high_month}")
        lines.append(f"  Lowest: {s.min():.2f} in {low_month}")
        lines.append("  ✅ Ending above average – good momentum" if end_val > avg_val
                     else "  ⚠ Ending below average – needs attention")
        lines.append("")
    return "\n".join(lines)

def plot_combined(df: pd.DataFrame, month_col: str, metric_cols):
    months = df[month_col].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(14, 8))
    for metric in metric_cols:
        ax.plot(months, df[metric], marker='o', label=metric)

    ax.set_title("Combined KPI Trends")
    ax.set_xlabel("Month")
    ax.set_ylabel("Value")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')  # legend outside
    ax.grid(True)

    st.pyplot(fig, clear_figure=True)

if uploaded:
    # Read CSV and normalize headers (whitespace)
    df_raw = pd.read_csv(uploaded)
    df_raw.columns = clean_headers(df_raw.columns)

    st.subheader("Preview")
    st.dataframe(df_raw.head(20), use_container_width=True, hide_index=True)

    # Guess a month-like column (e.g., "Month", "Helper Date", etc.)
    candidates = []
    for c in df_raw.columns:
        c_low = c.lower()
        if ("month" in c_low) or ("date" in c_low):
            candidates.append(c)
    default_month = candidates[0] if candidates else df_raw.columns[0]

    st.markdown("### Select columns")
    month_col = st.selectbox("Month column", options=list(df_raw.columns), index=list(df_raw.columns).index(default_month))

    # Preselect KPI columns if the common names exist, otherwise allow multi-select
    suggested = [n for n in df_raw.columns if any(k in n.lower() for k in [
        "completion", "nps", "placement", "reg to placement", "active student", "mentor"
    ])]
    metric_cols = st.multiselect(
        "KPI columns (select your six metrics)",
        options=list(df_raw.columns),
        default=suggested[:6] if suggested else []
    )

    if st.button("Generate Visual + Report", type="primary"):

        if not month_col or not metric_cols:
            st.error("Please select the Month column and at least one KPI column.")
        else:
            df = df_raw.copy()

            # Clean KPI columns to numeric (handles '33.5%' etc.)
            for col in metric_cols:
                df[col] = coerce_numeric(df[col])

            # Drop rows with no month or all KPIs missing
            df = df[df[month_col].notna()].reset_index(drop=True)
            if df.empty:
                st.error("No valid data rows after cleaning. Check your file.")
            else:
                plot_combined(df, month_col, metric_cols)

                report = generate_report(df, month_col, metric_cols)
                st.subheader("Trend Report")
                st.code(report)

                st.download_button(
                    label="⬇️ Download trend_report.txt",
                    data=report.encode("utf-8"),
                    file_name="trend_report.txt",
                    mime="text/plain"
                )
else:
    st.info("Upload a CSV to continue.")
