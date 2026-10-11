"""Test whether classifiers perform differently on human and sentiment variants.

For every model in a combined-results CSV, the script compares its predictions
on the human-written text against the neutral (``preserved``), positive, and
negative versions.  Every pair is evaluated on the same news items, so both
available tests are paired; choose one with --metric:

  --metric f1 (default)
      Paired permutation test (approximate randomization) of the F1 score: for
      each item the human and variant predictions are swapped at random, the F1
      difference is recomputed, and the p-value is the share of random swaps
      whose difference is at least as large as the observed one.  F1 averaging
      defaults to "weighted", as in confident_interval.py; use --average weighted
      for macro-F1.  Accuracy is reported as well.
  --metric accuracy
      McNemar's exact test on paired correct/incorrect outcomes.  (McNemar's
      test does not apply to F1, which is not an average of per-item outcomes.)

Effect sizes are reported with each test:
  f1        the F1 difference (human - variant) with a 95% paired bootstrap CI
            (--bootstrap resamples), and the relative change (variant - human) / human.
  accuracy  the odds ratio of the discordant pairs (human-only correct /
            variant-only correct) with a 95% CI, and Cohen's g (signed;
            |g| < .05 negligible, < .15 small, < .25 medium, otherwise large).

Results are saved to results/analysis/{dataset}_{paraphraser}_prediction_significance.csv
for accuracy and ..._prediction_significance_f1.csv for F1.

Example:
    python coding/analysis/stas_sig_pred.py isot llama
    python coding/analysis/stas_sig_pred.py liar gpt --average macro
    python coding/analysis/stas_sig_pred.py isot llama --metric accuracy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


VARIANT_COLUMNS = {
    # Older combined files use ``preserved`` for the neutral version.
    "preserved": ("neutral",),
    "positive": ("positive",),
    "negative": ("negative",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired significance tests of human versus sentiment-variant predictions."
    )
    parser.add_argument("dataset", help="Dataset name, e.g. isot")
    parser.add_argument("paraphraser", help="Paraphraser name, e.g. llama")
    parser.add_argument(
        "--metric", choices=["f1", "accuracy"], default="f1",
        help="f1: paired permutation test of F1 (default); accuracy: McNemar's exact test.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level (default: 0.05)."
    )
    parser.add_argument(
        "--average", choices=["weighted", "macro"], default="weighted",
        help="F1 only: averaging over classes (default: weighted).",
    )
    parser.add_argument(
        "--permutations", type=int, default=10000, help="F1 only: random swaps per test (default: 10000)."
    )
    parser.add_argument(
        "--bootstrap", type=int, default=2000,
        help="F1 only: bootstrap resamples for the effect-size CI (default: 2000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional combined-results CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )
    return parser.parse_args()


def prediction_column(columns: pd.Index, model: str, variant: str) -> str | None:
    """Find a model prediction column for a sentiment variant."""
    for suffix in VARIANT_COLUMNS[variant]:
        column = f"{model}_{suffix}"
        if column in columns:
            return column
    return None


def find_models(columns: pd.Index) -> list[str]:
    """Return model prefixes that have a human prediction column."""
    models = []
    for column in columns:
        if column.endswith("_human"):
            model = column[: -len("_human")]
            if all(prediction_column(columns, model, variant) for variant in VARIANT_COLUMNS):
                models.append(model)
            else:
                print(f"Skipping {model}: one or more variant prediction columns are missing.")
    return models


def paired_mcnemar(true_labels: pd.Series, human: pd.Series, variant: pd.Series) -> dict:
    """Return paired accuracies, discordant counts, and an exact McNemar p-value."""
    paired = pd.DataFrame({"truth": true_labels, "human": human, "variant": variant}).dropna()
    if paired.empty:
        raise ValueError("No rows contain labels and predictions for this comparison.")

    human_correct = paired["human"].eq(paired["truth"])
    variant_correct = paired["variant"].eq(paired["truth"])

    human_only = int((human_correct & ~variant_correct).sum())
    variant_only = int((~human_correct & variant_correct).sum())
    discordant = human_only + variant_only

    # With no discordant pairs, the two versions have identical accuracy.
    # The exact McNemar test is a two-sided binomial test of the discordant
    # pairs, with the null hypothesis that either version wins equally often.
    p_value = 1.0 if discordant == 0 else float(
        binomtest(human_only, n=discordant, p=0.5, alternative="two-sided").pvalue
    )

    # Effect sizes for McNemar's test, from the discordant pairs:
    #   odds ratio = human_only / variant_only (> 1: human version correct more often),
    #     with a 95% CI on the log scale (0.5 added to both counts if either is 0);
    #   Cohen's g = human_only / discordant - 0.5 (signed; |g| < .05 negligible,
    #     < .15 small, < .25 medium, otherwise large).
    b, c = (human_only, variant_only) if human_only and variant_only else (human_only + 0.5, variant_only + 0.5)
    log_or, se = np.log(b / c), np.sqrt(1 / b + 1 / c)
    cohens_g = human_only / discordant - 0.5 if discordant else 0.0

    return {
        "n_pairs": len(paired),
        "human_accuracy": human_correct.mean(),
        "variant_accuracy": variant_correct.mean(),
        "accuracy_difference": human_correct.mean() - variant_correct.mean(),
        "human_only_correct": human_only,
        "variant_only_correct": variant_only,
        "odds_ratio": float(np.exp(log_or)),
        "odds_ratio_ci_low": float(np.exp(log_or - 1.96 * se)),
        "odds_ratio_ci_high": float(np.exp(log_or + 1.96 * se)),
        "cohens_g": cohens_g,
        "effect_magnitude": magnitude_g(cohens_g),
        "p_value": p_value,
    }


def magnitude_g(g: float) -> str:
    """Cohen's (1988) labels for Cohen's g."""
    size = abs(g)
    return "negligible" if size < 0.05 else "small" if size < 0.15 else "medium" if size < 0.25 else "large"


def item_counts(truth: np.ndarray, pred: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-item contributions to each class's true positives and predicted count (n x k)."""
    predicted = pred[:, None] == classes[None, :]
    true_positive = predicted & (truth[:, None] == classes[None, :])
    return true_positive.astype(float), predicted.astype(float)


def f1_from_counts(tp: np.ndarray, predicted: np.ndarray, support: np.ndarray, average: str) -> np.ndarray:
    """F1 from per-class counts; works on a single row or a batch of rows (b x k).

    Matches sklearn's f1_score with zero_division=0 over the classes present in the labels.
    """
    denominator = predicted + support
    per_class = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    if average == "macro":
        return per_class.mean(axis=-1)  # unweighted mean over classes
    # weighted: each class by its support (support may also be a batch, b x k)
    return (per_class * support).sum(axis=-1) / support.sum(axis=-1)


def paired_permutation_f1(
    true_labels: pd.Series, human: pd.Series, variant: pd.Series,
    average: str, permutations: int, rng: np.random.Generator, bootstrap: int,
) -> dict:
    """Return paired F1 and accuracy, a two-sided paired permutation p-value for the F1
    difference, and its effect size: the difference with a 95% paired bootstrap CI."""
    paired = pd.DataFrame({"truth": true_labels, "human": human, "variant": variant}).dropna()
    if paired.empty:
        raise ValueError("No rows contain labels and predictions for this comparison.")

    truth = paired["truth"].to_numpy()
    classes = np.unique(truth)
    support = (truth[:, None] == classes[None, :]).sum(axis=0).astype(float)
    tp_h, pred_h = item_counts(truth, paired["human"].to_numpy(), classes)
    tp_v, pred_v = item_counts(truth, paired["variant"].to_numpy(), classes)

    human_f1 = float(f1_from_counts(tp_h.sum(0), pred_h.sum(0), support, average))
    variant_f1 = float(f1_from_counts(tp_v.sum(0), pred_v.sum(0), support, average))
    observed = human_f1 - variant_f1

    # Swapping item i moves its counts from one side to the other, so the counts
    # after a swap mask S are: human' = human + S @ (variant - human), variant' = variant - S @ (...)
    d_tp, d_pred = tp_v - tp_h, pred_v - pred_h
    exceed, done, batch = 0, 0, 500
    while done < permutations:
        size = min(batch, permutations - done)
        swap = (rng.random((size, len(truth))) < 0.5).astype(float)
        shift_tp, shift_pred = swap @ d_tp, swap @ d_pred
        diff = (f1_from_counts(tp_h.sum(0) + shift_tp, pred_h.sum(0) + shift_pred, support, average)
                - f1_from_counts(tp_v.sum(0) - shift_tp, pred_v.sum(0) - shift_pred, support, average))
        exceed += int((np.abs(diff) >= abs(observed) - 1e-12).sum())
        done += size
    p_value = (exceed + 1) / (permutations + 1)

    # Effect size: the F1 difference, with a 95% paired bootstrap CI (items resampled
    # with replacement, human and variant predictions kept together). Each resample is
    # a vector of item weights, so the class counts are weights @ per-item counts.
    true_onehot = (truth[:, None] == classes[None, :]).astype(float)
    boot_diffs, done = [], 0
    while done < bootstrap:
        size = min(batch, bootstrap - done)
        weights = rng.multinomial(len(truth), np.full(len(truth), 1 / len(truth)), size=size).astype(float)
        boot_support = weights @ true_onehot
        boot_diffs.append(f1_from_counts(weights @ tp_h, weights @ pred_h, boot_support, average)
                          - f1_from_counts(weights @ tp_v, weights @ pred_v, boot_support, average))
        done += size
    ci_low, ci_high = np.percentile(np.concatenate(boot_diffs), [2.5, 97.5])

    human_correct = paired["human"].eq(paired["truth"])
    variant_correct = paired["variant"].eq(paired["truth"])
    return {
        "n_pairs": len(paired),
        "human_f1": human_f1,
        "variant_f1": variant_f1,
        "f1_difference": observed,
        "f1_difference_ci_low": ci_low,
        "f1_difference_ci_high": ci_high,
        "relative_f1_change": -observed / human_f1 if human_f1 else np.nan,  # (variant - human) / human
        "human_accuracy": human_correct.mean(),
        "variant_accuracy": variant_correct.mean(),
        "accuracy_difference": human_correct.mean() - variant_correct.mean(),
        "p_value": p_value,
    }


def holm_adjust(p_values: pd.Series, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Holm-adjusted p-values and their rejection decisions."""
    values = p_values.to_numpy(dtype=float)
    count = len(values)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        [(count - rank) * values[index] for rank, index in enumerate(order)]
    ).clip(0, 1)
    adjusted = np.empty(count)
    adjusted[order] = adjusted_sorted
    return adjusted <= alpha, adjusted


def main() -> None:
    args = parse_args()
    input_path = args.input or Path(
        f"results/combined/{args.dataset}_{args.paraphraser}_all_results.csv"
    )
    suffix = "_f1" if args.metric == "f1" else ""  # accuracy keeps the original file name
    output_path = args.output or Path(
        f"results/analysis/{args.dataset}_{args.paraphraser}_prediction_significance{suffix}.csv"
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"Combined-results file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "True_label" not in df.columns:
        raise ValueError("The combined-results CSV must contain a 'True_label' column.")

    models = find_models(df.columns)
    if not models:
        raise ValueError("No complete model prediction sets were found in the combined-results CSV.")

    rng = np.random.default_rng(args.seed)
    results = []
    for model in models:
        for variant_name in VARIANT_COLUMNS:
            variant_column = prediction_column(df.columns, model, variant_name)
            if args.metric == "f1":
                result = paired_permutation_f1(
                    df["True_label"], df[f"{model}_human"], df[variant_column],
                    args.average, args.permutations, rng, args.bootstrap,
                )
            else:
                result = paired_mcnemar(df["True_label"], df[f"{model}_human"], df[variant_column])
            results.append({"model": model, "variant": variant_name, **result})

    results_df = pd.DataFrame(results)
    if args.metric == "f1":
        results_df.insert(2, "f1_average", args.average)
    rejected, adjusted_pvalues = holm_adjust(results_df["p_value"], args.alpha)
    results_df["p_value_holm"] = adjusted_pvalues
    results_df["significantly_different"] = rejected
    results_df["direction"] = results_df[f"{args.metric}_difference"].map(
        lambda difference: "human higher" if difference > 0 else "variant higher" if difference < 0 else "equal"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    if args.metric == "f1":
        print(f"F1 ({args.average}), paired permutation test with {args.permutations} swaps, Holm-adjusted:")
    else:
        print("Accuracy, McNemar's exact test, Holm-adjusted:")
    if args.metric == "f1":
        effect_columns = ["f1_difference", "f1_difference_ci_low", "f1_difference_ci_high"]
    else:
        effect_columns = ["accuracy_difference", "odds_ratio", "cohens_g", "effect_magnitude"]
    display_columns = [
        "model", "variant", f"human_{args.metric}", f"variant_{args.metric}",
        *effect_columns, "p_value_holm", "significantly_different", "direction",
    ]
    print(results_df[display_columns].to_string(index=False, float_format="{:.4f}".format))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
