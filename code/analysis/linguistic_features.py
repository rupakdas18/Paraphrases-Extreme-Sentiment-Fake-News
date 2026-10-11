"""
Linguistic features of human-written texts and their sentiment-shifted paraphrases.

For one dataset and paraphraser, loads the human texts and the positive, negative and
neutral (sentiment-preserved) paraphrases with load_data.data_load() (train + test + valid),
and compares three features:
    readability          Flesch Reading Ease (higher = easier to read)
    complex_word_ratio   difficult words / words (textstat.difficult_words)
    dependency_depth     depth of the deepest dependency parse tree among the text's sentences

Each paraphrase file also contains the human text it was made from, so every comparison
is paired: the human text and its own paraphrase. For each feature and sentiment the
script reports a paired t-test, a Wilcoxon signed-rank test (robust to the skewed,
long-tailed feature distributions) and the paired effect size (Cohen's d_z).

Outputs, in results/linguistic/{dataset}/{paraphraser}/:
    {dataset}_{paraphraser}_{feature}.csv           one row per text pair, the three
                                                    sentiments stacked (column "sentiment")
    {dataset}_{paraphraser}_{feature}_summary.csv   statistics per sentiment
    {dataset}_{paraphraser}_{feature}.pdf           histograms: human + three sentiments

The plots leave out extreme values (see PLOT_LIMITS) so the bulk of each distribution is
visible; the CSVs and statistics use all values.

How to run (from the project root; set DATASET and PARAPHRASER below):
    python coding/analysis/linguistic_features.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import textstat
from scipy import stats
import load_data


DATASET = "election_2024"        # 'isot', 'liar', 'kaggle', 'covid-19', 'election_2024'
PARAPHRASER = "gemini"     # 'gpt', 'gemini', 'llama'
LANGUAGE = "English"
NUM_CLASSES = 6 if DATASET == "liar" else 2
FEATURES = ["readability", "complex_word_ratio", "dependency_depth"]  # dependency_depth is the slow one
LIMIT = None            # e.g. 200 to try the script on the first rows of each file

SENTIMENTS = ["positive", "negative", "neutral"]
HUMAN_COLUMN = "human"
PARAPHRASE_COLUMN = PARAPHRASER

# Same colors as SentenceTransformer.py and factual_entailment.py; human in blue
COLORS = {"human": "#1f77b4", "positive": "#ff7f0e", "negative": "#2ca02c", "neutral": "#d62728"}
# Keys are the file suffixes used by load_data; the "neutral" files hold the sentiment-preserved paraphrases
TITLES = {"human": "Human-written text", "positive": "Positive sentiment",
          "negative": "Negative sentiment", "neutral": "Preserved sentiment"}
AXIS_LABELS = {"readability": "Flesch Reading Ease Score",
               "complex_word_ratio": "Complex word ratio",
               "dependency_depth": "Depth of parse tree"}
# Plot range only (min, max); values outside are left out of the histograms, not the statistics
PLOT_LIMITS = {"readability": (-100, None), "complex_word_ratio": (0, 1), "dependency_depth": (0, 25)}

OUT_DIR = os.path.join("results", "linguistic", DATASET, PARAPHRASER)

nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
nlp.max_length = 2_000_000


# ============================================================
# FEATURES
# ============================================================

def readability(texts):
    return np.array([textstat.flesch_reading_ease(t) if t.strip() else np.nan for t in texts])


def complex_word_ratio(texts):
    def ratio(text):
        words = len(text.split())
        return textstat.difficult_words(text) / words if words else np.nan
    return np.array([ratio(t) for t in texts])


def dependency_depth(texts):
    """Depth (in tokens, root = 1) of the deepest parse tree among each text's sentences."""
    depths = []
    for doc in nlp.pipe(texts, batch_size=32):
        # A token's depth is its number of ancestors + 1; the tree depth is the deepest token
        depth = max((sum(1 for _ in tok.ancestors) + 1 for tok in doc if not tok.is_space), default=np.nan)
        depths.append(depth)
    return np.array(depths, dtype=float)


FEATURE_FUNCTIONS = {"readability": readability, "complex_word_ratio": complex_word_ratio,
                     "dependency_depth": dependency_depth}


def compute(feature, texts, cache):
    """Feature values for a list of texts; values are cached per text (the human texts repeat)."""
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if todo:
        cache.update(zip(todo, FEATURE_FUNCTIONS[feature](todo)))
    return np.array([cache[t] for t in texts], dtype=float)


# ============================================================
# STATISTICS
# ============================================================

def paired_statistics(human, paraphrased):
    """Descriptive statistics and paired tests for one feature and sentiment."""
    keep = ~(np.isnan(human) | np.isnan(paraphrased))
    h, p = human[keep], paraphrased[keep]
    diff = p - h
    t_stat, t_p = stats.ttest_rel(h, p)
    w_stat, w_p = stats.wilcoxon(h, p) if np.any(diff != 0) else (np.nan, 1.0)
    return {
        "n_pairs": int(keep.sum()),
        "human_mean": h.mean(), "human_std": h.std(ddof=1), "human_median": np.median(h),
        "human_min": h.min(), "human_max": h.max(),
        "paraphrased_mean": p.mean(), "paraphrased_std": p.std(ddof=1), "paraphrased_median": np.median(p),
        "paraphrased_min": p.min(), "paraphrased_max": p.max(),
        "mean_difference": diff.mean(),                      # paraphrased - human
        "cohens_dz": diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0,
        "t_statistic": t_stat, "t_test_p": t_p,
        "wilcoxon_statistic": w_stat, "wilcoxon_p": w_p,
    }


def print_statistics(feature, sentiment, s, alpha=0.05):
    print(f"\n[{feature} | {sentiment}] n = {s['n_pairs']}")
    print(f"  human:       mean {s['human_mean']:.4f}  sd {s['human_std']:.4f}  median {s['human_median']:.4f}")
    print(f"  paraphrased: mean {s['paraphrased_mean']:.4f}  sd {s['paraphrased_std']:.4f}  median {s['paraphrased_median']:.4f}")
    print(f"  difference (paraphrased - human): {s['mean_difference']:+.4f}, Cohen's d_z = {s['cohens_dz']:+.3f}")
    verdict = "significant" if s["wilcoxon_p"] < alpha else "not significant"
    print(f"  paired t-test p = {s['t_test_p']:.3g} | Wilcoxon p = {s['wilcoxon_p']:.3g} ({verdict} at {alpha})")


# ============================================================
# PLOT
# ============================================================

def plot_distributions(feature, values, path):
    """Four histograms (human + three sentiments) on shared bins and axes."""
    low, high = PLOT_LIMITS[feature]
    shown = {}
    for name, v in values.items():
        v = v[~np.isnan(v)]
        if low is not None:
            v = v[v >= low]
        if high is not None:
            v = v[v <= high]
        shown[name] = v
    all_values = np.concatenate(list(shown.values()))
    if feature == "dependency_depth":
        bins = np.arange(all_values.min() - 0.5, all_values.max() + 1.5)  # one bin per integer depth
    else:
        bins = np.linspace(all_values.min(), all_values.max(), 31)

    fig, axes = plt.subplots(len(shown), 1, figsize=(12, 10), sharex=True, sharey=True)
    for ax, (name, v) in zip(axes, shown.items()):
        ax.hist(v, bins=bins, color=COLORS[name], edgecolor="black", alpha=0.7)
        ax.axvline(np.median(v), color="black", linestyle=":", linewidth=2,
                   label=f"Median = {np.median(v):.3f}")
        ax.set_title(TITLES[name], fontsize=20)
        ax.set_ylabel("Frequency", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend(fontsize=14, loc="upper right")
    axes[-1].set_xlabel(AXIS_LABELS[feature], fontsize=20)

    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def load_version(paraphraser, sentiment):
    full_df, _, _, _ = load_data.data_load(DATASET, NUM_CLASSES, paraphraser, LANGUAGE, "n/a", sentiment)
    full_df = full_df.dropna(subset=[c for c in (HUMAN_COLUMN, PARAPHRASE_COLUMN) if c in full_df.columns])
    full_df = full_df.reset_index(drop=True)  # train/test/valid each start at 0; make rows unique
    return full_df.head(LIMIT) if LIMIT else full_df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    human_df = load_version("human", "")
    variants = {s: load_version(PARAPHRASER, s) for s in SENTIMENTS}
    print(f"Rows: human {len(human_df)}, " + ", ".join(f"{s} {len(d)}" for s, d in variants.items()))

    for feature in FEATURES:
        print(f"\n###################### {feature} ######################")
        cache = {}
        human_texts = human_df[HUMAN_COLUMN].astype(str).tolist()
        plot_values = {"human": compute(feature, human_texts, cache)}
        pairs, summary = [], []

        for sentiment, df in variants.items():
            human = compute(feature, df[HUMAN_COLUMN].astype(str).tolist(), cache)
            paraphrased = compute(feature, df[PARAPHRASE_COLUMN].astype(str).tolist(), cache)
            plot_values[sentiment] = paraphrased

            pairs.append(pd.DataFrame({
                "sentiment": sentiment,
                "label": df["label"].to_numpy() if "label" in df.columns else np.nan,
                f"human_{feature}": human,
                f"paraphrased_{feature}": paraphrased,
                "human_text": df[HUMAN_COLUMN].to_numpy(),
                "paraphrased_text": df[PARAPHRASE_COLUMN].to_numpy(),
            }))
            s = paired_statistics(human, paraphrased)
            print_statistics(feature, sentiment, s)
            summary.append({"feature": feature, "sentiment": sentiment, **s})

        base = os.path.join(OUT_DIR, f"{DATASET}_{PARAPHRASER}_{feature}")
        pd.concat(pairs, ignore_index=True).to_csv(base + ".csv", index=False)
        pd.DataFrame(summary).to_csv(base + "_summary.csv", index=False)
        plot_distributions(feature, plot_values, base + ".pdf")
        print(f"\nSaved {base}.csv, {base}_summary.csv and {base}.pdf")


if __name__ == "__main__":
    main()
