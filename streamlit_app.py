import io
import pandas as pd
import streamlit as st
from trend_logic import load_and_clean, package_downloads

st.set_page_config(page_title="MBR Trend Charts & Report", layout="wide")
st.title("Monthly Business Review – Trend Charts & Downloadable Report")

st.write("Upload the **combined CSV** with all verticals (same format you shared).")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

# Optional: show a sample note for expected format
with st.expander("Expected CSV format (pivot export)"):
    st.markdown("""
    - First row after the first column contains vertical names.
    - First column has `Helper Date`, `Vertical`, then month labels like `Jan25`, `Feb25`, ...
    - Each cell is a percentage like `33.50%`.
    """)

if uploaded:
    raw = pd.read_csv(uploaded, header=0)
    df = load_and_clean(raw)

    st.subheader("Cleaned Data Preview")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Generate all charts + report and prepare downloads
    zip_bytes, files = package_downloads(df)

    st.subheader("Downloads")
    st.download_button(
        label="⬇️ Download ALL (ZIP: charts + report)",
        data=zip_bytes,
        file_name="mbr_trends_bundle.zip",
        mime="application/zip"
    )

    # Individual downloads + inline preview
    st.write("Or download individually:")
    for name, buf in files.items():
        if name.endswith(".png"):
            st.image(buf, caption=name)
        st.download_button(
            label=f"Download {name}",
            data=buf.getvalue(),
            file_name=name
        )
else:
    st.info("Please upload your combined CSV to generate charts and the trend report.")
