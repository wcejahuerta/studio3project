import json, re
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------- USAA THEME ----------
USAA_NAVY = "#002F6C"
USAA_GOLD = "#CC9900"
USAA_BABY_BLUE = "#A7C7E7"
USAA_SLATE = "#4D4D4F"
USAA_LIGHT_GRAY = "#A7A8AA"

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": USAA_SLATE,
    "axes.labelcolor": USAA_SLATE,
    "xtick.color": USAA_SLATE,
    "ytick.color": USAA_SLATE,
    "grid.color": USAA_LIGHT_GRAY,
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "axes.grid": False,        
    "font.size": 12,
    "axes.titleweight": "bold",
})

# Heatmap colormap (navy → baby blue → white → gold)
USAA_CMAP = LinearSegmentedColormap.from_list(
    "usaa_cmap", [USAA_NAVY, USAA_BABY_BLUE, "white", USAA_GOLD]
)
# ---------------------------------

# ---------- CONFIG ----------
JSON_PATH   = Path("fraud_results.json")

TEXT_COL    = "cleaned_text"
CLASS_COL   = "fraud_related"
REASON_COL  = "fraud_reason"
DATE_COL    = "date"
CLUSTER_COL = "kmeans_cluster"

EXTRA_STOPS = {
    "occ","fdic","frs","federal","reserve","treasury","office","department",
    "united","states","u","s","section","bank","banks","banking","institution",
    "institutions","agency","agencies","newsroom","pdf","page","pages","date",
    "bulletin","press","release","public","policy","regulation","regulatory",
    "comment","comments","docket","system","board","governors"
}

# LOB keyword lists (from teammate)
insurance_keywords = ["claim", "payout", "collision", "adjuster", "policyholder"]
banking_keywords   = ["zelle", "wire", "account takeover", "credit card", "ach"]
investing_keywords = ["retirement", "ira", "rollover", "brokerage", "portfolio"]

# Trend buckets
TREND_MAP = {
    "phish": "Phishing / Social engineering",
    "impersonat": "Identity theft / ATO",
    "identity": "Identity theft / ATO",
    "account takeover": "Identity theft / ATO",
    "check": "Check fraud",
    "peer": "P2P payment scams",
    "zelle": "P2P payment scams",
    "invest": "Investment / Crypto scams",
    "crypto": "Investment / Crypto scams",
    "wire": "Wire / Transfer scams",
    "ach": "ACH / Transfer fraud",
    "elder": "Elder financial exploitation",
    "scam": "General scams",
}

# ---------- HELPERS ----------

def load_any_json(path: Path):
    """Load JSON file into Python object."""
    with path.open("r", encoding="utf-8") as f:
        return json.loads(f.read().strip())

def assign_trend(reason_text: str) -> str:
    """Map free-text fraud_reason to a trend bucket."""
    t = (reason_text or "").lower()
    for needle, label in TREND_MAP.items():
        if needle in t:
            return label
    return "Other / General fraud"

def detect_lob(text: str) -> str:
    """Roughly classify article into Banking / Insurance / Investing."""
    txt = (text or "").lower()
    scores = {
        "Insurance": sum(word in txt for word in insurance_keywords),
        "Banking":   sum(word in txt for word in banking_keywords),
        "Investing": sum(word in txt for word in investing_keywords),
    }
    # Return the LOB with the highest score (ties are fine)
    return max(scores, key=scores.get)

# ---------- MAIN ----------

def main():
    # ---- Load & normalize ----
    records = load_any_json(JSON_PATH)
    df = pd.json_normalize(records, sep=".")
    print("Columns:", df.columns.tolist())

    # ---- Parse dates -> year ----
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(
            df[DATE_COL],
            format="%B %d, %Y",   # e.g. "December 4, 2024"
            errors="coerce"
        )
        df["year"] = df[DATE_COL].dt.year.astype("Int64")
    else:
        print("⚠️ No 'date' column found; skipping year-based analysis.")

    # ---- Keep only fraud-related ----
    if CLASS_COL in df.columns:
        df = df[df[CLASS_COL] == True]

    if df.empty:
        print("⚠️ No fraud-related rows found. Nothing to plot.")
        return

    # ---- Detect Line of Business (LOB) ----
    df["LOB"] = df[TEXT_COL].astype(str).apply(detect_lob)
    print("\nLOB counts:\n", df["LOB"].value_counts())

    # ============================================================
    # 1) TOP KEYWORDS BY LOB (USAA styled) + LEGEND
    #    (x-axis shows ONLY keywords, color shows LOB)
    # ============================================================
    lob_top = {}

    for lob, rows in df.groupby("LOB"):
        text = " ".join(rows[TEXT_COL].astype(str)).lower()
        words = re.findall(r"\b[a-z]{3,}\b", text)
        stopset = ENGLISH_STOP_WORDS.union(EXTRA_STOPS)
        toks = [w for w in words if w not in stopset and not w.isdigit()]
        counts = Counter(toks).most_common(7)
        lob_top[lob] = counts

    # Map LOB -> color
    lob_colors = {
        "Banking":   USAA_NAVY,
        "Insurance": USAA_GOLD,
        "Investing": USAA_BABY_BLUE,
    }

    # Flatten into single list so x-axis labels are ONLY keywords
    all_keywords = []
    all_values = []
    all_colors = []

    for lob, items in lob_top.items():
        for kw, count in items:
            all_keywords.append(kw)
            all_values.append(count)
            all_colors.append(lob_colors.get(lob, USAA_SLATE))

    x = range(len(all_keywords))

    plt.figure(figsize=(11, 6))
    plt.grid(True, axis="y")

    plt.bar(
        x,
        all_values,
        color=all_colors,
        edgecolor=USAA_SLATE,
        linewidth=1.0,
        alpha=0.9,
    )

    # X-axis labels are JUST the keywords now
    plt.xticks(x, all_keywords, rotation=45, ha="right")

    # Legend that always shows all 3 LOBs
    handles = [
        plt.Line2D([0], [0], color=USAA_NAVY,      lw=10, label="Banking"),
        plt.Line2D([0], [0], color=USAA_GOLD,      lw=10, label="Insurance"),
        plt.Line2D([0], [0], color=USAA_BABY_BLUE, lw=10, label="Investing"),
    ]
    plt.legend(handles=handles, title="Line of Business", loc="upper right")

    plt.title("Top Keywords by Line of Business")
    plt.ylabel("Keyword Frequency")
    plt.tight_layout()
    plt.savefig("lob_keywords.png", dpi=260)
    print("Saved: lob_keywords.png")

    # ============================================================
    # 2) FRAUD TRENDS (from LLM reasons)
    # ============================================================
    if REASON_COL in df.columns:
        trend_series = df[REASON_COL].astype(str).apply(assign_trend)
        trend_counts = trend_series.value_counts()
        trend_counts.to_csv("trend_counts.csv")
        print("\nTrend buckets:\n", trend_counts)

        plt.figure()
        plt.grid(True, axis="y")
        trend_counts.plot(
            kind="bar",
            color=USAA_NAVY,
            edgecolor=USAA_SLATE,
            linewidth=1.2
        )
        plt.title("Fraud Trends (from LLM Reasons)")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig("top_trends.png", dpi=260)
        print("Saved: top_trends.png")
    else:
        print("\n⚠️ No REASON column; skipping trend chart.")

    # ============================================================
    # 3) ARTICLES BY YEAR
    # ============================================================
    if "year" in df.columns and df["year"].notna().any():
        year_counts = (
            df["year"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )
        year_counts.to_csv("articles_by_year.csv")
        print("\nArticles by year:\n", year_counts)

        plt.figure()
        plt.grid(True, axis="y")
        plt.bar(
            year_counts.index.astype(str),
            year_counts.values,
            color=USAA_NAVY,
            edgecolor=USAA_SLATE,
            linewidth=1.2
        )
        plt.title("Articles by Year (Scraped Dates)")
        plt.ylabel("Number of Articles")
        plt.xlabel("Year")
        plt.tight_layout()
        plt.savefig("articles_by_year.png", dpi=260)
        print("Saved: articles_by_year.png")
    else:
        print("\n⚠️ No valid years; skipping articles-by-year chart.")

    # ============================================================
    # 4) TREND-BY-YEAR HEATMAP
    # ============================================================
    if (
        "year" in df.columns
        and df["year"].notna().any()
        and REASON_COL in df.columns
    ):
        trend_series = df[REASON_COL].astype(str).apply(assign_trend)
        valid_mask = df["year"].notna()
        table = pd.crosstab(df.loc[valid_mask, "year"], trend_series[valid_mask])
        print("\nTrend by year table:\n", table)

        if not table.empty:
            plt.figure(figsize=(11, 5))
            plt.grid(False)

            plt.imshow(
                table.values,
                aspect="auto",
                cmap=USAA_CMAP,
                interpolation="nearest"   # clean blocks, no extra lines
            )

            plt.xticks(
                range(len(table.columns)),
                table.columns,
                rotation=45,
                ha="right"
            )
            plt.yticks(
                range(len(table.index)),
                table.index
            )
            plt.colorbar(label="Number of Articles")
            plt.title("Fraud Trends by Year")
            plt.xlabel("Trend Bucket")
            plt.ylabel("Year")
            plt.tight_layout()
            plt.savefig("trend_by_year_heatmap.png", dpi=260)
            print("Saved: trend_by_year_heatmap.png")
    else:
        print("\n⚠️ Skipping heatmap (need both year and fraud_reason).")

if __name__ == "__main__":
    main()
