import json, re
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

# ---------- CONFIG (adjust if your column names change) ----------
JSON_PATH = Path("fraud_results.json")

# Your JSON fields from the snippet you shared:
TEXT_COL        = "cleaned_text"
CLASS_COL       = "fraud_related"      # boolean True/False
REASON_COL      = "fraud_reason"
CLUSTER_COL     = "kmeans_cluster"     # optional

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

    # ---- Tokenize text
    fraud_text = " ".join(work[TEXT_COL].astype(str)).lower()
    words = re.findall(r"\b[a-z]{3,}\b", fraud_text)

    # ---- Stop-words filter
    stopset = ENGLISH_STOP_WORDS.union(EXTRA_STOPS)
    tokens = [w for w in words if w not in stopset and not w.isdigit()]

    # ---- Top keywords
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
        plt.tight_layout()
        plt.savefig("top5_keywords.png", dpi=200)
        print("Saved: top5_keywords.png")

    # ---- Phrases (bigrams/trigrams) – optional
    try:
        corpus = df[TEXT_COL].astype(str).tolist()
        cv = CountVectorizer(ngram_range=(2,3), max_features=10, stop_words="english")
        X = cv.fit_transform(corpus)
        print("\nSample top phrases:", list(cv.get_feature_names_out())[:5])
    except Exception as e:
        print("\n(Skipping phrase extraction):", e)

    # ---- Trend buckets from LLM reasons
    trend_counts = pd.Series(dtype=int)
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
        plt.ylabel("Articles")
        plt.tight_layout()
        plt.savefig("top_trends.png", dpi=200)
        print("Saved: top_trends.png")
    else:
        print("\nNo REASON column found; skipping trend chart.")

    # ---- (Optional) Cluster breakdown
    if CLUSTER_COL in df.columns:
        cluster_counts = df[CLUSTER_COL].value_counts().sort_index()
        print("\nCluster counts:\n", cluster_counts)

if __name__ == "__main__":
    main()
