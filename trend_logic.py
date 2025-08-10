# in trend_logic.py, replace make_grouped_bar(...) with this:
def make_grouped_bar(df: pd.DataFrame, title: str) -> io.BytesIO:
    months = df['Month'].tolist()
    verticals = get_verticals(df)

    fig, ax = plt.subplots(figsize=(14, 8))  # larger canvas
    for v in verticals:
        ax.plot(months, df[v], marker='o', label=v)  # line chart for readability

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Value")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha='right')  # no overlap
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')  # legend outside
    ax.grid(True)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

        for name, b in files.items():
            z.writestr(name, b.getvalue())
    zip_bytes.seek(0)

    return zip_bytes, files
