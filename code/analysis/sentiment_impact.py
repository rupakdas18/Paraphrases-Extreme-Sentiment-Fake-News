"""
Impact of sentiment shift on classification performance.

Every test article exists in three paraphrased versions written by the same
paraphraser: sentiment-preserved ("neutral"), shifted positive and shifted
negative. Because all three describe the same article, comparing a model's
results on the positive/negative versions against the neutral version of the
SAME articles isolates the effect of the sentiment shift from the effect of
paraphrasing (and from how hard each article is).

For every model in the combined results file:
  1. Performance per version (human, neutral, positive, negative): accuracy,
     macro-F1, and recall on fake (0) and true (1) news.
  2. Paired effect of the sentiment shift: change in accuracy and macro-F1 of
     the positive/negative version relative to neutral, with 95% bootstrap
     confidence intervals (resampling articles), and an exact McNemar test on
     the articles where the two versions disagree. p-values are Holm-adjusted
     across all model x variant tests.
  3. Direction: change in recall per class. Shows whether a sentiment shift
     pushes predictions toward "fake" or toward "true".
  4. Dose-response: the accuracy drop relative to neutral for articles split
     into thirds by how far the paraphrase moved the sentiment (VADER compound
     score change from the human text). A drop that grows with the shift
     points to sentiment rather than paraphrasing as the cause.

HOW TO USE (from the project root, in the dl environment):
    python coding/analysis/sentiment_impact.py isot gemini
    python coding/analysis/sentiment_impact.py isot gemini --models CNN BERT

Outputs go to results/sentiment_impact/{dataset}_{paraphraser}/:
    performance.csv        metrics per model and version
    paired_effects.csv     shift effects vs neutral with CIs and McNemar tests
    class_recall.csv       recall change per class vs neutral
    dose_response.csv      accuracy change vs neutral by sentiment-shift size
    accuracy_change.pdf/.png   figure of the paired accuracy change per model
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results")
OUT_ROOT = os.path.join(RESULTS, "sentiment_impact")

# Column names in results/combined/{dataset}_{paraphraser}_all_results.csv
TEXT_COLUMNS = {
    "human": "human_written",
    "neutral": "neutral_sentiment",
    "positive": "positive_sentiment",
    "negative": "negative_sentiment",
}
VERSIONS = ["human", "neutral", "positive", "negative"]
SHIFTED = ["positive", "negative"]
CONTROL = "neutral"
LABEL_COLUMN = "True_label"
ID_COLUMN = "id"

# Reference categorical palette, slots 1 and 2 (blue, orange)
COLORS = {"positive": "#2a78d6", "negative": "#eb6834"}
MARKERS = {"positive": "o", "negative": "s"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def parse_args():
    parser = argparse.ArgumentParser(description="Impact of sentiment shift on classification performance")
    parser.add_argument("dataset", type=str, help="Dataset name, e.g. isot")
    parser.add_argument("paraphraser", type=str, help="Paraphraser name, e.g. gemini")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to analyse (default: every model in the combined file)")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for confidence intervals")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--combined", type=str, default=None,
                        help="Path to a different combined results CSV (default: "
                             "results/combined/{dataset}_{paraphraser}_all_results.csv)")
    return parser.parse_args()


def load_combined(dataset, paraphraser, path=None):
    path = path or os.path.join(RESULTS, "combined", f"{dataset}_{paraphraser}_all_results.csv")
    if not os.path.exists(path):
        sys.exit(f"Combined results file not found: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except OSError as err:
        sys.exit(f"Could not read {path}: {err}\nIf this is an online-only OneDrive file, open the folder in "
                 "File Explorer, right-click the file, choose 'Always keep on this device' and wait for the "
                 "download to finish (green check mark).")
    missing = [c for c in [ID_COLUMN, LABEL_COLUMN] + list(TEXT_COLUMNS.values()) if c not in df.columns]
    if missing:
        sys.exit(f"{path} is missing columns {missing}")
    return df


def available_models(df):
    """Models that have a prediction column for every version."""
    prefixes = [c[: -len("_human")] for c in df.columns if c.endswith("_human") and c != "human_written"]
    return [m for m in prefixes if all(f"{m}_{v}" in df.columns for v in VERSIONS)]


def sentiment_shift(df, dataset, paraphraser):
    """VADER compound score change from the human text to each shifted version, cached."""
    cache = os.path.join(OUT_ROOT, f"{dataset}_{paraphraser}_vader_scores.csv")
    ids = df[ID_COLUMN].astype(str)
    if os.path.exists(cache):
        scores = pd.read_csv(cache, dtype={ID_COLUMN: str}).set_index(ID_COLUMN)
        if set(ids) <= set(scores.index):
            scores = scores.loc[ids]
            return pd.DataFrame({v: (scores[v] - scores["human"]).to_numpy() for v in SHIFTED})

    print("Computing VADER sentiment for all texts (cached for later runs)...")
    analyzer = SentimentIntensityAnalyzer()
    scores = pd.DataFrame({ID_COLUMN: ids})
    for version in ["human"] + SHIFTED:
        scores[version] = [analyzer.polarity_scores(str(t))["compound"] for t in df[TEXT_COLUMNS[version]]]
    os.makedirs(OUT_ROOT, exist_ok=True)
    scores.to_csv(cache, index=False)
    return pd.DataFrame({v: (scores[v] - scores["human"]).to_numpy() for v in SHIFTED})


def metrics(y, pred):
    correct = pred == y
    return {
        "accuracy": correct.mean(),
        "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0),
        "recall_fake": correct[y == 0].mean(),
        "recall_true": correct[y == 1].mean(),
    }


def paired_effect(y, pred_variant, pred_control, n_boot, rng):
    """Variant minus control for accuracy and macro-F1, bootstrapped over articles, plus exact McNemar."""
    n = len(y)
    acc_v, acc_c = (pred_variant == y), (pred_control == y)
    boot_acc, boot_f1 = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        boot_acc[b] = acc_v[i].mean() - acc_c[i].mean()
        boot_f1[b] = (f1_score(y[i], pred_variant[i], average="macro", labels=[0, 1], zero_division=0)
                      - f1_score(y[i], pred_control[i], average="macro", labels=[0, 1], zero_division=0))
    lost = int((acc_c & ~acc_v).sum())    # right on neutral, wrong on the shifted version
    gained = int((~acc_c & acc_v).sum())  # wrong on neutral, right on the shifted version
    p = binomtest(lost, lost + gained, 0.5).pvalue if lost + gained else 1.0
    return {
        "delta_accuracy": acc_v.mean() - acc_c.mean(),
        "acc_ci_low": np.percentile(boot_acc, 2.5), "acc_ci_high": np.percentile(boot_acc, 97.5),
        "delta_macro_f1": (f1_score(y, pred_variant, average="macro", labels=[0, 1], zero_division=0)
                           - f1_score(y, pred_control, average="macro", labels=[0, 1], zero_division=0)),
        "f1_ci_low": np.percentile(boot_f1, 2.5), "f1_ci_high": np.percentile(boot_f1, 97.5),
        "articles_lost": lost, "articles_gained": gained, "mcnemar_p": p,
    }


def holm(p_values):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def dose_response(y, preds, shift, model):
    """Paired accuracy change vs neutral within thirds of the sentiment-shift size."""
    rows = []
    control_correct = preds[CONTROL] == y
    for v in SHIFTED:
        size = np.abs(shift[v].to_numpy())
        thirds = pd.qcut(size, 3, labels=["small", "medium", "large"])
        for t in ["small", "medium", "large"]:
            m = np.asarray(thirds == t)
            rows.append({
                "model": model, "variant": v, "shift_size": t, "articles": int(m.sum()),
                "mean_abs_shift": size[m].mean(),
                "delta_accuracy": (preds[v][m] == y[m]).mean() - control_correct[m].mean(),
            })
    return rows


def plot_accuracy_change(effects, path, title):
    """Dot plot of the paired accuracy change, fine-tuned and zero-shot models in separate panels.

    Zero-shot models swing by tens of points; on a shared scale they would hide
    the fine-tuned models' effects.
    """
    models = list(dict.fromkeys(effects["model"]))
    groups = [(name, ms) for name, ms in [
        ("Fine-tuned models", [m for m in models if not m.startswith("zs_")]),
        ("Zero-shot models", [m for m in models if m.startswith("zs_")])] if ms]
    fig, axes = plt.subplots(len(groups), 1, squeeze=False,
                             figsize=(6.4, 0.5 * len(models) + 1.0 + 0.9 * len(groups)),
                             gridspec_kw={"height_ratios": [len(ms) for _, ms in groups]})
    offsets = {"positive": -0.14, "negative": 0.14}
    for ax, (name, group) in zip(axes[:, 0], groups):
        for v in SHIFTED:
            e = effects[effects["variant"] == v].set_index("model").loc[group]
            ax.errorbar(100 * e["delta_accuracy"], np.arange(len(group)) + offsets[v],
                        xerr=[100 * (e["delta_accuracy"] - e["acc_ci_low"]),
                              100 * (e["acc_ci_high"] - e["delta_accuracy"])],
                        fmt=MARKERS[v], color=COLORS[v], ms=6, elinewidth=2, capsize=0,
                        label=f"{v} vs neutral")
        ax.axvline(0, color=INK_MUTED, lw=1)
        ax.set_yticks(np.arange(len(group)), group)
        ax.set_ylim(len(group) - 0.5, -0.5)
        ax.set_title(name, color=INK_MUTED, loc="left", fontsize=9)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_MUTED, length=0)
    axes[-1, 0].set_xlabel("Change in accuracy vs neutral (percentage points), 95% CI", color=INK)
    fig.suptitle(title, color=INK, x=0.02, ha="left", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.95), ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = os.path.join(OUT_ROOT, f"{args.dataset}_{args.paraphraser}")
    os.makedirs(out_dir, exist_ok=True)

    df = load_combined(args.dataset, args.paraphraser, args.combined)
    models = args.models or available_models(df)
    unknown = [m for m in models if m not in available_models(df)]
    if unknown:
        sys.exit(f"No complete prediction columns for {unknown}. Available: {available_models(df)}")
    y = df[LABEL_COLUMN].to_numpy()
    shift = sentiment_shift(df, args.dataset, args.paraphraser)

    performance, effects, recalls, doses = [], [], [], []
    for model in models:
        # Anything other than 0/1 (e.g. an unparsable LLM answer) counts as wrong
        preds = {v: df[f"{model}_{v}"].to_numpy() for v in VERSIONS}
        invalid = {v: int((~np.isin(p, [0, 1])).sum()) for v, p in preds.items()}
        for v in VERSIONS:
            performance.append({"model": model, "version": v, **metrics(y, preds[v]), "invalid_predictions": invalid[v]})
        control = metrics(y, preds[CONTROL])
        for v in SHIFTED:
            effects.append({"model": model, "variant": v, **paired_effect(y, preds[v], preds[CONTROL], args.bootstrap, rng)})
            m = metrics(y, preds[v])
            recalls.append({"model": model, "variant": v,
                            "delta_recall_fake": m["recall_fake"] - control["recall_fake"],
                            "delta_recall_true": m["recall_true"] - control["recall_true"]})
        doses.extend(dose_response(y, preds, shift, model))
        print(f"Done: {model}")

    performance = pd.DataFrame(performance)
    effects = pd.DataFrame(effects)
    effects["mcnemar_p_holm"] = holm(effects["mcnemar_p"])
    recalls, doses = pd.DataFrame(recalls), pd.DataFrame(doses)

    performance.to_csv(os.path.join(out_dir, "performance.csv"), index=False)
    effects.to_csv(os.path.join(out_dir, "paired_effects.csv"), index=False)
    recalls.to_csv(os.path.join(out_dir, "class_recall.csv"), index=False)
    doses.to_csv(os.path.join(out_dir, "dose_response.csv"), index=False)
    plot_accuracy_change(effects, os.path.join(out_dir, "accuracy_change"),
                         f"{args.dataset} / {args.paraphraser}: effect of sentiment shift on accuracy")

    pp = lambda x: f"{100 * x:+.2f}"
    print("\nAccuracy per version:")
    print(performance.pivot(index="model", columns="version", values="accuracy")[VERSIONS]
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nPaired change vs neutral (percentage points), Holm-adjusted McNemar p:")
    for _, r in effects.iterrows():
        print(f"  {r['model']:<11} {r['variant']:<8} acc {pp(r['delta_accuracy'])} "
              f"[{pp(r['acc_ci_low'])}, {pp(r['acc_ci_high'])}]  F1 {pp(r['delta_macro_f1'])}  "
              f"lost {r['articles_lost']:>4} / gained {r['articles_gained']:>4}  p={r['mcnemar_p_holm']:.3g}")
    print("\nRecall change vs neutral (percentage points):")
    print((recalls.set_index(["model", "variant"]) * 100).round(2).to_string())
    print("\nDose-response: accuracy change vs neutral (pp) by sentiment-shift size:")
    print((doses.pivot_table(index=["model", "variant"], columns="shift_size", values="delta_accuracy")
           [["small", "medium", "large"]] * 100).round(2).to_string())
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
