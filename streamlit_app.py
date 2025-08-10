# streamlit_app.py
import io, re, traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Combined KPI Trends", layout="wide")
st.title("Combined KPI Trends — One Visual + Full Report")

st.markdown("""
Upload **one CSV** and select the Month column and **your KPI columns**.
Works with any header names (e.g., *AVERAGE of Course completion %*, *AVERAGE of NPS*, etc.).
""")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

def clean_headers(cols):
    # keep original text but trim surrounding spaces for matching
    return [c.strip() for c in cols]

def coerce_numeric(series: pd.Series) -> pd.Series:
    # remove %, commas, and spaces; convert to number
    s = (series.astype(str)
               .str.replace('%', '', regex=False)
               .str.replace(',', '', regex=False)
               .str.strip())
    return pd.to_numeric(s, errors='coerce')

def guess_month_columns(df: pd.DataFrame):
    # try header-name guess
    name_hits = [c for c in df.columns if any(k in c.lower() for k in ["month", "date", "helper"])]
    # try value-pattern guess (Jan25, Feb25, etc.)
    pat = re.compile(r'^[A-Za-z]{3}\d{2}$')
    value_hits = []
    for c in df.columns:
        try:
            vals = df[c].astype(str).dropna().head(10)
            if (vals.str.match(pat)).mean() > 0.5:
                value_hits.append(c)
        except Exception:
            pass
    # de‑duplicate, preserve order (name hits first)
    seen, out = set(), []
    for c in name_hits + value_hits + list(df.columns):
        if c not in seen:
            out.append(c); seen.add(c)
    return out

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

def plot_combined(df: pd.DataFrame, month_col: str, metric_cols, normalize=False, rating_max=5.0):
    plot_df = df.copy()
    if normalize:
        for col in metric_cols:
            name = col.lower()
            s = plot_df[col].astype(float)
            if "rating" in name:
                plot_df[col] = (s / rating_max) * 100.0
            elif "placement" in name and "%" not in name:
                # counts → scale to 0–100 by max for readability
                m = np.nanmax(s.values)
                plot_df[col] = (s / m) * 100.0 if m and m > 0 else s
            # percentages are already numeric percentages after cleaning

    months = plot_df[month_col].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(14, 8))
    for metric in metric_cols:
        ax.plot(months, plot_df[metric], marker='o', label=metric)

    ax.set_title("Combined KPI Trends")
    ax.set_xlabel("Month")
    ax.set_ylabel("Value" + (" (normalized to %)" if normalize else ""))
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)

if uploaded:
    try:
        df_raw = pd.read_csv(uploaded)
        df_raw.columns = clean_headers(df_raw.columns)

        st.subheader("Preview")
        st.dataframe(df_raw.head(20), use_container_width=True, hide_index=True)

        st.markdown("### Select columns")
        # Month select (smart defaults)
        month_options = guess_month_columns(df_raw)
        month_col = st.selectbox("Month column", options=list(df_raw.columns),
                                 index=list(df_raw.columns).index(month_options[0]) if month_options else 0)

        # Suggest metric columns by keyword
        suggested = [n for n in df_raw.columns if any(k in n.lower() for k in [
            "completion", "nps", "placement", "reg to placement", "active", "mentor", "rating"
        ])]
        metric_cols = st.multiselect(
            "KPI columns (pick your six)",
            options=list(df_raw.columns),
            default=suggested[:6] if suggested else []
        )

        st.caption("Tip: Your six KPIs are usually named like: "
                   "‘AVERAGE of Course completion %’, ‘AVERAGE of NPS’, "
                   "‘SUM of No of Placements(Monthly)’, ‘AVERAGE of Reg to Placement %’, "
                   "‘AVERAGE of Active Student %’, ‘AVERAGE of Avg Mentor Rating’.")

        normalize = st.checkbox("Normalize dissimilar scales to % (ratings & counts → % for a single scale)", value=False)
        rating_max = st.number_input("If normalizing, Mentor Rating max is", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

        if st.button("Generate Visual + Report", type="primary"):
            if not month_col or not metric_cols:
                st.error("Select the Month column and at least one KPI column.")
                st.stop()

            df = df_raw.copy()
            # Clean KPI columns to numeric
            for col in metric_cols:
                df[col] = coerce_numeric(df[col])

            # Drop rows missing month or all metrics
            df = df[df[month_col].notna()].reset_index(drop=True)
            if df.empty:
                st.error("No valid data rows after cleaning — check your file contents.")
                st.stop()

            # Render chart
            plot_combined(df, month_col, metric_cols, normalize=normalize, rating_max=rating_max)

            # Build & download report (always uses raw values, not normalized)
            report = generate_report(df, month_col, metric_cols)
            st.subheader("Trend Report")
            st.code(report)
            st.download_button("⬇️ Download trend_report.txt", report.encode("utf-8"),
                               file_name="trend_report.txt", mime="text/plain")

    except Exception as e:
        st.error("Something went wrong while processing your file.")
        st.exception(e)  # show full stacktrace inside the app
else:
    st.info("Upload a CSV to continue.")
