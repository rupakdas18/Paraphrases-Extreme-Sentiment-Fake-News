import numpy as np
import pandas as pd
import argparse


"""

Effect size calculations for sentiment analysis scores.
Helpers for computing paired effect sizes between human and paraphrased sentiment scores.

HOW TO USE:
1) Configure the columns below to match your CSV file.
2) Run this script with the dataset name and paraphraser as arguments.
   Example:
     python effect_size.py isot gemini

"""

# Optional: if your file has an ID column and you want to sanity-check alignment
# ID_COL = "id"   # set to None if not needed
ID_COL = None

def parse_args():
    parser = argparse.ArgumentParser(description="Fake news classification + effect size")

    parser.add_argument(
        "dataset",
        type=str,
        choices=["kaggle", "liar", "covid-19", "TALLIP", "election_2024", "isot"],
        help="Dataset name"
    )

    parser.add_argument(
        "paraphraser",
        type=str,
        choices=["gemini", "bard", "parrot", "gpt", "pegasus", "llama", "human"],
        help="Paraphrasing model"
    )

    return parser.parse_args()

# ----------------------------
# 2) Effect size functions
# ----------------------------
def cohen_dz(x: np.ndarray, y: np.ndarray) -> float:
    """
    Paired-samples Cohen's d (a.k.a. dz):
      d_z = mean(x - y) / sd(x - y)
    """
    d = x - y
    sd = d.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return d.mean() / sd

def hedges_gz(x: np.ndarray, y: np.ndarray) -> float:
    """
    Small-sample corrected paired effect size:
      g_z = J * d_z, where J = 1 - 3/(4*df - 1), df = n - 1
    """
    dz = cohen_dz(x, y)
    n = np.sum(~np.isnan(x - y))
    if n <= 1 or np.isnan(dz):
        return np.nan
    df = n - 1
    J = 1 - (3 / (4 * df - 1)) if (4 * df - 1) != 0 else np.nan
    return J * dz

def rank_biserial_from_paired(x: np.ndarray, y: np.ndarray) -> float:
    """
    Rank-biserial correlation for paired data (nonparametric effect size).
    Uses sign of pairwise differences, ignores ties.
      r_rb = (n_pos - n_neg) / (n_pos + n_neg)
    """
    d = x - y
    d = d[~np.isnan(d)]
    pos = np.sum(d > 0)
    neg = np.sum(d < 0)
    denom = pos + neg
    if denom == 0:
        return np.nan
    return (pos - neg) / denom

def summarize_pair(df: pd.DataFrame, a: str, b: str) -> dict:
    # Pairwise drop missing
    pair = df[[a, b]].dropna()
    x = pair[a].to_numpy(dtype=float)
    y = pair[b].to_numpy(dtype=float)

    diff = x - y
    return {
        "comparison": f"{a} vs {b}",
        "n": len(diff),
        "mean_diff_(Δ)": float(np.mean(diff)) if len(diff) else np.nan,
        "sd_diff": float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan,
        "cohen_dz": float(cohen_dz(x, y)),
        "hedges_gz": float(hedges_gz(x, y)),
        "rank_biserial_r": float(rank_biserial_from_paired(x, y)),
    }

# ----------------------------
# 3) Load + compute
# ----------------------------

# ----------------------------
# 1) Configure these columns
# ----------------------------

args = parse_args()
dataset_name = args.dataset
paraphraser = args.paraphraser

CSV_PATH = f"results/sentiment/{dataset_name}_{paraphraser}_vader_scores.csv"

COL_HUMAN = "Human_sentiment"
COL_POS   = "Positive_sentiment"
COL_NEG   = "Negative_sentiment"
COL_PRES  = "Preserved_sentiment"


df = pd.read_csv(CSV_PATH)

if ID_COL is not None and ID_COL in df.columns:
    # Sort to ensure deterministic alignment if needed
    df = df.sort_values(ID_COL).reset_index(drop=True)

required = [COL_HUMAN, COL_POS, COL_NEG, COL_PRES]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}\nAvailable: {list(df.columns)}")

results = []
results.append(summarize_pair(df, COL_HUMAN, COL_PRES))
results.append(summarize_pair(df, COL_HUMAN, COL_POS))
results.append(summarize_pair(df, COL_HUMAN, COL_NEG))

out = pd.DataFrame(results)
pd.set_option("display.max_columns", None)
print(out.to_string(index=False))
