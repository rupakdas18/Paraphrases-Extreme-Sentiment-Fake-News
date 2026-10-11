


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import variables

# =========================
# 1) Load your CSV
# =========================

variables.dataset_name = 'election_2024' # 'TALLIP''liar_2','liar_6', 'kaggle', 'covid-19', 'election_2024
variables.paraphraser = 'gpt'
CSV_PATH = f"results/combined/{variables.dataset_name}_{variables.paraphraser}_all_results.csv"   # <-- change if needed
df = pd.read_csv(CSV_PATH)
print("columns in combined results:", df.columns)

# =========================

# 2) Settings
# =========================

# List the models you want plotted, in any combination.
# Deep-learning / transformer-based: CNN, LSTM, GPT, BERT, Llama
# Zero-shot LLMs:                    zs_mistral, zs_phi35, zs_qwen
# Supervised ML classifiers:         LR, DT, RF, SVM, XG
# NOTE: supervised ML predictions (LR/DT/RF/SVM/XG) live in
# results/supervised/{dataset}_{paraphraser}{sentiment}_classification_results.xlsx
# (columns like "LR_{paraphraser}_{feature}"), not in the combined CSV this
# script reads -- including them here will print a warning and skip those
# models until that merge is added to combine_results.py.
MODELS = ["CNN", "LSTM", "BERT", "Llama", "GPT",'SVM_tfidf','SVM_cv','SVM_w2v_we','SVM_glove_we','SVM_bert_we']  # <-- change if needed

CONDS  = ["Human", "Preserved", "Positive", "Negative"]

# Column names in YOUR csv
TRUE_COL = "True_label"
PRED_COL = {
    "Human":   {m: f"{m}_human"   for m in MODELS},
    "Preserved": {m: f"{m}_neutral" for m in MODELS},
    "Positive":{m: f"{m}_positive"for m in MODELS},
    "Negative":{m: f"{m}_negative"for m in MODELS},
}

# Choose F1 averaging:
# - For binary: "binary"
# - For multi-class: "macro" (harder) or "weighted" (common)
F1_AVG = "weighted"

N_BOOT = 2000
SEED = 42

# Vertical offsets so conditions don’t overlap per model row
OFFSETS = {"Human": -0.18, "Preserved": -0.06, "Positive": 0.06, "Negative": 0.18}

# Extra empty space (in classifier rows) above the first classifier, and the
# legend laid out in one row inside it, so the legend does not cover the top
# intervals. Datasets not listed keep the default layout.
TOP_PADDING = {"liar": 2.4}

# Fixed F1-axis range per dataset (min, max). Datasets not listed use matplotlib's automatic range.
X_RANGE = {"liar": (0.10, 0.35)}

# =========================
# 3) Bootstrap CI function
# =========================
def bootstrap_f1_ci(y_true, y_pred, average="weighted", n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    scores = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)  # sample with replacement
        scores[b] = f1_score(y_true[idx], y_pred[idx], average=average)

    mean = scores.mean()
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return mean, lo, hi

# =========================
# 4) Compute summary table
#    (drops NaNs per model/condition automatically)
# =========================
rows = []
for m in MODELS:
    for c in CONDS:
        col = PRED_COL[c][m]
        if col not in df.columns:
            print(f"WARNING: column '{col}' not found in combined CSV -- skipping {m}/{c}")
            continue
        sub = df[[TRUE_COL, col]].dropna()   # important: your csv has NaNs in many rows
        if len(sub) == 0:
            continue

        mean, lo, hi = bootstrap_f1_ci(
            sub[TRUE_COL].values,
            sub[col].values,
            average=F1_AVG,
            n_boot=N_BOOT,
            seed=SEED
        )
        rows.append({"Model": m, "Condition": c, "F1": mean, "Lo": lo, "Hi": hi, "N": len(sub)})

summary = pd.DataFrame(rows)

# =========================
# 5) Plot (dot + horizontal 95% CI)
# =========================

COND_COLORS = {
    "Human":    "#1f77b4",  # blue
    "Preserved":  "#2ca02c",  # green
    "Positive": "#d62728",  # red
    "Negative": "#9467bd",  # purple
}


fig, ax = plt.subplots(figsize=(12, 6))

y_base = {m: i for i, m in enumerate(MODELS)}

for cond in CONDS:
    subc = summary[summary["Condition"] == cond]
    for _, r in subc.iterrows():
        y = y_base[r["Model"]] + OFFSETS[cond]
        x = r["F1"]
        xerr = np.array([[x - r["Lo"]], [r["Hi"] - x]])

        ax.errorbar(
            x, y,
            xerr=xerr,
            fmt="o",
            capsize=5,
            elinewidth=2,
            color=COND_COLORS[cond],        # <-- FIX
            markeredgecolor="black",
            label=cond if r["Model"] == MODELS[0] else None
)
        # numeric label next to point (optional)
        # ax.text(x, y + 0.03, f"{x:.3f}", fontsize=9)

ax.set_yticks([y_base[m] for m in MODELS])
ax.set_yticklabels(MODELS)
ax.invert_yaxis()  # so MODELS[0] (CNN) is at the top, matching list order
top_padding = TOP_PADDING.get(variables.dataset_name, 0)
if top_padding:
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top - top_padding)
if variables.dataset_name in X_RANGE:
    ax.set_xlim(*X_RANGE[variables.dataset_name])
ax.set_xlabel("F1 score", fontsize=20)
ax.set_ylabel("Classifiers", fontsize=20)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.grid(True, axis="x", linestyle="--", alpha=0.35)
ax.grid(True, axis="y", linestyle="--", alpha=0.15)
if top_padding:
    ax.legend(title="Sentiment", loc="upper center", ncol=len(CONDS),
              fontsize=16, title_fontsize=18)
else:
    ax.legend(title="Sentiment", loc="upper left",
              fontsize=16, title_fontsize=18)

plt.tight_layout()

# Save or show
plt.savefig(f"results/confident_interval/{variables.dataset_name}_{variables.paraphraser}_f1_ci_by_model_and_sentiment.pdf", dpi=300, bbox_inches="tight")
plt.show()

print(summary.sort_values(["Model", "Condition"]))
