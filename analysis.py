#   - Loads fraud_results.json
#   - Cleans text and finds top keywords
#   - Buckets fraud reasons into trend categories
#   - Uses scraped article dates to show trends over time

import json, re
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

# ---------- CONFIG ----------
JSON_PATH   = Path("fraud_results.json")

TEXT_COL    = "cleaned_text"
CLASS_COL   = "fraud_related"      # boolean True/False (optional)
REASON_COL  = "fraud_reason"       # text explaining why it's fraud (optional)
CLUSTER_COL = "kmeans_cluster"     # optional
DATE_COL    = "date"               # <-- scraped from the website, e.g. "December 4, 2024"

# Extra stop-words (domain-specific junk you don't want in Top 5)
EXTRA_STOPS = {
    "occ","fdic","frs","federal","reserve","treasury","office","department",
    "united","states","u","s","section","bank","banks","banking","institution",
    "institutions","agency","agencies","newsroom","pdf","page","pages","date",
    "bulletin","press","release","public","policy","regulation","regulatory",
    "comment","comments","docket","system","board","governors"
}

# Trend bucketing: maps substrings -> human label
TREND_MAP = {
    "phish": "Phishing / Social engineering",
    "impersonat": "Identity theft / ATO",
    "identity": "Identity theft / ATO",
    "account takeover": "Identity theft / ATO",
    "check": "Check fraud",
    "peer": "P2P payment scams",      # catches "peer-to-peer"
    "zelle": "P2P payment scams",
    "invest": "Investment / Crypto scams",
    "crypto": "Investment / Crypto scams",
    "wire": "Wire / Transfer scams",
    "ach": "ACH / Transfer fraud",
    "elder": "Elder financial exploitation",
    "scam": "General scams"
}
# ---------------------------------------------------------------

def load_any_json(path: Path):
    """Load JSON that could be a list, dict, or NDJSON; return list of records."""
    with path.open("r", encoding="utf-8") as f:
        txt = f.read().strip()
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        # NDJSON fallback
        records = []
        for line in txt.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # common container keys
        for k in ["results","data","items","records","rows"]:
            if isinstance(obj.get(k), list):
                return obj[k]
        # dict of id -> record or a single record
        if all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())
        return [obj]
    raise ValueError("Unsupported JSON structure")

def assign_trend(reason_text: str) -> str:
    """Map free-text fraud_reason to a trend bucket."""
    t = (reason_text or "").lower()
    for needle, label in TREND_MAP.items():
        if needle in t:
            return label
    return "Other / General fraud"

def main():
    # ---- Load & normalize
    records = load_any_json(JSON_PATH)
    df = pd.json_normalize(records, sep=".")
    print("Columns:", df.columns.tolist())

    # ---- Date handling (uses scraped "date" column)
    if DATE_COL in df.columns:
        # Your JSON has dates like "December 4, 2024"
        df[DATE_COL] = pd.to_datetime(
            df[DATE_COL],
            format="%B %d, %Y",      # MonthName Day, Year
            errors="coerce"
        )
        df["year"] = df[DATE_COL].dt.year
    else:
        print(f"⚠️ Column '{DATE_COL}' not found; skipping date-based analysis.")

    # ---- Basic column presence checks
    for col in [TEXT_COL, REASON_COL, CLASS_COL]:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found. Available columns: {df.columns.tolist()}")
    if TEXT_COL not in df.columns:
        raise KeyError(f"Required text column '{TEXT_COL}' is missing.")

    # ---- Keep only fraud-related rows if available; else use all rows
    work = df
    if CLASS_COL in df.columns:
        mask = df[CLASS_COL].astype(str).str.lower().isin(["true","1","yes"])
        if mask.any():
            work = df[mask]

    # ============================================================
    # 1) TOP KEYWORDS (single words)
    # ============================================================
    fraud_text = " ".join(work[TEXT_COL].astype(str)).lower()
    words = re.findall(r"\b[a-z]{3,}\b", fraud_text)

    stopset = ENGLISH_STOP_WORDS.union(EXTRA_STOPS)
    tokens = [w for w in words if w not in stopset and not w.isdigit()]

    counts = Counter(tokens)
    top5 = counts.most_common(5)
    top20 = counts.most_common(20)
    print("\nTop 5 keywords (stopwords removed):", top5)

    # Save keyword tables
    pd.DataFrame(top5, columns=["Keyword","Count"]).to_csv("top5_keywords.csv", index=False)
    pd.DataFrame(top20, columns=["Keyword","Count"]).to_csv("top20_keywords.csv", index=False)
    print("Saved: top5_keywords.csv, top20_keywords.csv")

    # Chart: Top 5 keywords
    if top5:
        labels = [w for w,_ in top5]
        values = [c for _,c in top5]
        plt.figure()
        plt.bar(labels, values)
        plt.title("Top 5 Keywords (fraud-related articles)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("top5_keywords.png", dpi=200)
        print("Saved: top5_keywords.png")

    # ============================================================
    # 2) PHRASES (bigrams/trigrams) – just to print some examples
    # ============================================================
    try:
        corpus = df[TEXT_COL].astype(str).tolist()
        cv = CountVectorizer(ngram_range=(2,3), max_features=10, stop_words="english")
        X = cv.fit_transform(corpus)
        print("\nSample top phrases:", list(cv.get_feature_names_out())[:5])
    except Exception as e:
        print("\n(Skipping phrase extraction):", e)

    # ============================================================
    # 3) TREND BUCKETS (bar chart) – from LLM fraud_reason
    # ============================================================
    trend_counts = pd.Series(dtype=int)
    trends = None
    if REASON_COL in df.columns:
        trends = df[REASON_COL].astype(str).apply(assign_trend)
        trend_counts = trends.value_counts().sort_values(ascending=False)
        print("\nTrend buckets:\n", trend_counts)
        trend_counts.to_csv("trend_counts.csv")
        print("Saved: trend_counts.csv")

        # Chart: Trends
        plt.figure()
        trend_counts.plot(kind="bar")
        plt.title("Fraud Trends (from LLM reasons)")
        plt.ylabel("Number of Articles")
        plt.tight_layout()
        plt.savefig("top_trends.png", dpi=200)
        print("Saved: top_trends.png")
    else:
        print("\nNo REASON column found; skipping trend chart.")

    # ============================================================
    # 4) ARTICLES BY YEAR (using scraped date)
    # ============================================================
    if DATE_COL in df.columns and df[DATE_COL].notna().any():
        year_counts = df["year"].value_counts().sort_index()
        print("\nArticles by year:\n", year_counts)

        year_counts.to_csv("articles_by_year.csv")
        print("Saved: articles_by_year.csv")

        plt.figure()
        year_counts.plot(kind="bar")
        plt.title("Articles by Year (from scraped dates)")
        plt.ylabel("Number of Articles")
        plt.xlabel("Year")
        plt.tight_layout()
        plt.savefig("articles_by_year.png", dpi=200)
        print("Saved: articles_by_year.png")
    else:
        print("\nNo valid dates found; skipping articles-by-year chart.")

    # ============================================================
    # 5) TREND-BY-YEAR TABLE + HEATMAP
    #    (crosses trend buckets with years)
    # ============================================================
    if (
        DATE_COL in df.columns
        and df[DATE_COL].notna().any()
        and REASON_COL in df.columns
    ):
        # Use the same trend mapping, but drop rows with no year
        trend_series = df[REASON_COL].astype(str).apply(assign_trend)
        valid_mask = df["year"].notna()
        trend_by_year = pd.crosstab(df.loc[valid_mask, "year"], trend_series[valid_mask])

        print("\nTrend by year (table):\n", trend_by_year)
        trend_by_year.to_csv("trend_by_year.csv")
        print("Saved: trend_by_year.csv")

        # Heatmap-style visualization using matplotlib
        if not trend_by_year.empty:
            plt.figure(figsize=(8, 4))
            plt.imshow(trend_by_year.values, aspect="auto")
            plt.colorbar(label="Number of Articles")

            plt.xticks(
                ticks=range(len(trend_by_year.columns)),
                labels=trend_by_year.columns,
                rotation=45,
                ha="right"
            )
            plt.yticks(
                ticks=range(len(trend_by_year.index)),
                labels=trend_by_year.index
            )
            plt.title("Fraud Trends by Year (Heatmap)")
            plt.xlabel("Trend Bucket")
            plt.ylabel("Year")
            plt.tight_layout()
            plt.savefig("trend_by_year_heatmap.png", dpi=200)
            print("Saved: trend_by_year_heatmap.png")
    else:
        print("\nSkipping trend-by-year heatmap (need both DATE and REASON columns).")

    # ============================================================
    # 6) Optional: Cluster breakdown (if kmeans_cluster exists)
    # ============================================================
    if CLUSTER_COL in df.columns:
        cluster_counts = df[CLUSTER_COL].value_counts().sort_index()
        print("\nCluster counts:\n", cluster_counts)

if __name__ == "__main__":
    main()
