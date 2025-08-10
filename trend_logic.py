import re
import io
import zipfile
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MONTH_RE = re.compile(r'^[A-Za-z]{3}\d{2}$')

def load_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a pivot-style CSV where:
      - Row 0 after col0 contains vertical names
      - First column contains 'Helper Date', 'Vertical', and month labels like Jan25
    Returns tidy numeric DataFrame with columns: Month, <verticals...> (floats).
    """
    # Build header from row 0
    cols = ['Month'] + df_raw.iloc[0, 1:].tolist()
    df = df_raw.copy()
    df.columns = cols

    # Keep only true month rows (Jan25, Feb25, ...)
    df = df[df['Month'].astype(str).str.match(MONTH_RE, na=False)].reset_index(drop=True)

    # Convert percentage strings → float
    for col in df.columns[1:]:
        df[col] = (
            df[col].astype(str)
                  .str.replace('%', '', regex=False)
                  .str.replace(',', '', regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop fully empty columns (rare)
    empties = [c for c in df.columns[1:] if df[c].notna().sum() == 0]
    if empties:
        df = df.drop(columns=empties)

    return df

def get_verticals(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ['Month', 'Grand Total']]

def make_grouped_bar(df: pd.DataFrame) -> io.BytesIO:
    months = df['Month'].tolist()
    verticals = get_verticals(df)
    x = np.arange(len(months))
    bw = 0.8 / max(len(verticals), 1)

    fig = plt.figure(figsize=(12, 6))
    for i, v in enumerate(verticals):
        plt.bar(x + i*bw, df[v], width=bw, label=v)
    plt.xticks(x + (len(verticals)-1)*bw/2, months)
    plt.title("Monthly Course Completion % – All Verticals")
    plt.xlabel("Month"); plt.ylabel("Completion %"); plt.legend(); plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def make_single_bar(df: pd.DataFrame, vertical: str) -> io.BytesIO:
    months = df['Month'].tolist()
    fig = plt.figure(figsize=(9, 4.5))
    plt.bar(months, df[vertical])
    plt.title(f"{vertical} – Monthly Course Completion %")
    plt.xlabel("Month"); plt.ylabel("Completion %"); plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def build_trend_report(df: pd.DataFrame) -> str:
    verticals = get_verticals(df)
    lines = ["TREND REPORT", "="*50, ""]
    for v in verticals:
        s = df[v].dropna()
        if s.empty:
            continue
        start_val, end_val = s.iloc[0], s.iloc[-1]
        change = end_val - start_val
        trend = "↑ Increasing" if change > 0 else "↓ Decreasing" if change < 0 else "→ Stable"
        avg_val = s.mean()
        high_m = df.loc[s.idxmax(), 'Month']
        low_m = df.loc[s.idxmin(), 'Month']

        lines.append(f"{v}: {trend}")
        lines.append(f"  Start: {start_val:.2f}%, End: {end_val:.2f}% (Change: {change:+.2f}%)")
        lines.append(f"  Average: {avg_val:.2f}%")
        lines.append(f"  Highest: {s.max():.2f}% in {high_m}")
        lines.append(f"  Lowest: {s.min():.2f}% in {low_m}")
        lines.append("  ✅ Ending above average – good momentum" if end_val > avg_val
                     else "  ⚠ Ending below average – needs attention")
        lines.append("")
    return "\n".join(lines)

def package_downloads(df: pd.DataFrame) -> Tuple[io.BytesIO, Dict[str, io.BytesIO]]:
    """
    Returns:
      - a ZIP (BytesIO) containing all PNGs + trend_report.txt
      - a dict for individual downloads: {"combined_trend.png": buf, "<v>_trend.png": buf, "trend_report.txt": buf_txt}
    """
    files: Dict[str, io.BytesIO] = {}

    # Combined chart
    combined_buf = make_grouped_bar(df)
    files["combined_trend.png"] = combined_buf

    # Per-vertical charts
    for v in get_verticals(df):
        vbuf = make_single_bar(df, v)
        safe = v.replace(" ", "_").replace("/", "_")
        files[f"{safe}_trend.png"] = vbuf

    # Report
    report = build_trend_report(df)
    report_buf = io.BytesIO(report.encode("utf-8"))
    files["trend_report.txt"] = report_buf

    # ZIP
    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, "w", zipfile.ZIP_DEFLATED) as z:
        for name, b in files.items():
            z.writestr(name, b.getvalue())
    zip_bytes.seek(0)

    return zip_bytes, files
