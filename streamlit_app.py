import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monthly Business Review", layout="wide")

st.title("📊 Monthly Business Review - Combined KPI Trends")

uploaded_file = st.file_uploader("Upload pivot-style CSV", type=["csv"])

def parse_pivot_csv(file):
    raw = pd.read_csv(file, header=None)
    vertical_row = raw.iloc[0]
    metric_row = raw.iloc[1]

    # Forward-fill verticals
    verticals = []
    current_vertical = None
    for v in vertical_row:
        if pd.notna(v):
            current_vertical = v
        verticals.append(current_vertical)

    records = []
    for col_idx in range(1, raw.shape[1]):
        vertical = verticals[col_idx]
        metric = metric_row[col_idx]
        for row_idx in range(2, raw.shape[0]):
            month = str(raw.iloc[row_idx, 0])
            value = raw.iloc[row_idx, col_idx]
            if isinstance(value, str):
                value = value.replace('%', '').replace(',', '').strip()
            try:
                value = float(value)
            except:
                value = np.nan
            records.append([month, vertical, metric, value])

    df = pd.DataFrame(records, columns=["Month", "Vertical", "Metric", "Value"])
    return df

def aggregate_kpis(df):
    keep_metrics = [
        "AVERAGE of Course completion %",
        "AVERAGE of NPS",
        "SUM of No of Placements(Monthly)",
        "AVERAGE of Reg to Placement %",
        "AVERAGE of Active Student %",
        "AVERAGE of Avg Mentor Rating"
    ]
    df = df[df["Metric"].isin(keep_metrics)]

    # Decide aggregation
    agg_rules = {
        "SUM of No of Placements(Monthly)": "sum"
    }
    for m in keep_metrics:
        if m not in agg_rules:
            agg_rules[m] = "mean"

    agg_df = df.groupby(["Month", "Metric"]).agg({"Value": agg_rules.get}).reset_index()

    # Sort months if needed
    agg_df["Month"] = pd.Categorical(agg_df["Month"], categories=agg_df["Month"].unique(), ordered=True)
    return agg_df

def plot_combined_chart(agg_df):
    fig, ax = plt.subplots(figsize=(14, 8))
    for metric in agg_df["Metric"].unique():
        subset = agg_df[agg_df["Metric"] == metric]
        ax.plot(subset["Month"], subset["Value"], marker='o', label=metric)

    ax.set_xlabel("Month")
    ax.set_ylabel("Value")
    ax.set_title("Combined KPI Trends")
    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=45)
    st.pyplot(fig)

def generate_trend_report(agg_df):
    report_lines = []
    for metric in agg_df["Metric"].unique():
        subset = agg_df[agg_df["Metric"] == metric]
        start = subset["Value"].iloc[0]
        end = subset["Value"].iloc[-1]
        change = end - start
        avg_val = subset["Value"].mean()
        high_val = subset["Value"].max()
        low_val = subset["Value"].min()
        status = "✅ Above Avg" if end >= avg_val else "⚠ Below Avg"
        report_lines.append(f"{metric}: Start={start:.2f}, End={end:.2f}, Change={change:.2f}, "
                            f"Avg={avg_val:.2f}, High={high_val:.2f}, Low={low_val:.2f} → {status}")
    return "\n".join(report_lines)

if uploaded_file:
    df = parse_pivot_csv(uploaded_file)
    agg_df = aggregate_kpis(df)

    st.subheader("📈 Combined KPI Chart")
    plot_combined_chart(agg_df)

    st.subheader("📝 Trend Report")
    report_text = generate_trend_report(agg_df)
    st.text(report_text)

    st.download_button("Download Trend Report", report_text, file_name="trend_report.txt")
