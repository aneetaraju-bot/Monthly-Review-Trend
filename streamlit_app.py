import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Combined KPI Trends", layout="wide")
st.title("Combined KPI Trends – Single Visual + Full Report")

st.markdown("""
Upload a **single CSV** with these columns (exact names recommended):  
**Month, Course Completion %, NPS %, No of Placements, Reg to Placement %, Active Student %, Avg Mentor Rating**
""")

uploaded = st.file_uploader("Upload combined KPI CSV", type=["csv"])

def make_chart_and_report(df: pd.DataFrame):
    # Ensure Month is str and keep metric columns in order
    df['Month'] = df['Month'].astype(str)
    metrics = [c for c in df.columns if c != 'Month']

    # --- Chart: combined line chart (clean & readable) ---
    fig, ax = plt.subplots(figsize=(14, 8))
    for metric in metrics:
        ax.plot(df['Month'], df[metric], marker='o', label=metric)

    ax.set_title("Combined KPI Trends")
    ax.set_xlabel("Month")
    ax.set_ylabel("Value")
    ax.set_xticks(range(len(df['Month'])))
    ax.set_xticklabels(df['Month'], rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)

    # --- Report ---
    lines = ["TREND REPORT", "="*50, ""]
    for metric in metrics:
        s = pd.to_numeric(df[metric], errors='coerce').dropna()
        if s.empty:
            continue
        start_val, end_val = s.iloc[0], s.iloc[-1]
        change = end_val - start_val
        trend = "↑ Increasing" if change > 0 else "↓ Decreasing" if change < 0 else "→ Stable"
        avg_val = s.mean()
        high_idx = s.idxmax()
        low_idx = s.idxmin()
        high_month = df.loc[high_idx, 'Month']
        low_month = df.loc[low_idx, 'Month']

        lines.append(f"{metric}: {trend}")
        lines.append(f"  Start: {start_val:.2f}, End: {end_val:.2f} (Change: {change:+.2f})")
        lines.append(f"  Average: {avg_val:.2f}")
        lines.append(f"  Highest: {s.max():.2f} in {high_month}")
        lines.append(f"  Lowest: {s.min():.2f} in {low_month}")
        lines.append("  ✅ Ending above average – good momentum" if end_val > avg_val
                     else "  ⚠ Ending below average – needs attention")
        lines.append("")

    report_text = "\n".join(lines)
    st.subheader("Trend Report")
    st.code(report_text)

    # Download button
    st.download_button(
        label="⬇️ Download trend_report.txt",
        data=report_text.encode("utf-8"),
        file_name="trend_report.txt",
        mime="text/plain"
    )

if uploaded:
    df = pd.read_csv(uploaded)
    # light cleanup if CSV includes % signs for percentage columns
    for col in df.columns:
        if col != "Month":
            df[col] = (df[col].astype(str).str.replace('%','', regex=False)
                                   .str.replace(',','', regex=False))
            df[col] = pd.to_numeric(df[col], errors='coerce')
    st.subheader("Preview")
    st.dataframe(df, use_container_width=True, hide_index=True)
    make_chart_and_report(df)
else:
    st.info("Please upload the combined KPI CSV to generate the visual and report.")
