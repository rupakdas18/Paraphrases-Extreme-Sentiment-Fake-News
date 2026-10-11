'''
This script computes the factual entailment between original sentences and their paraphrased versions using a pre-trained NLI model. 
It processes the data in batches for efficiency and saves the entailment scores along with the original data for further analysis.
'''
# import pandas as pd
# import load_data
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm

# # =========================
# # DEVICE
# # =========================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # =========================
# # MODEL
# # =========================
# model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
# # model_name = "joeddav/xlm-roberta-large-xnli"
# # model_name = "facebook/bart-large-mnli"
# # model_name = "microsoft/deberta-v3-large-mnli" # Need HuggingFace login
# # model_name = "roberta-large-mnli"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
# model.eval()

# label_names = ["entailment", "contradiction"]

# # =========================
# # LOAD DATA
# # =========================

# dataset_name = 'covid-19' # 'TALLIP''liar_2','liar_6', 'kaggle', 'covid-19'
# project = "02-Paraphrasing"
# paraphraser = 'gpt' # 'human', 'gpt', 'llama', 'pegasus', 'gemini'
# sentiment = "_negative" # "positive", "negative", "neutral", "doesn't matter"
# num_class = 2
# df, train_df, test_df, valid_df = load_data.data_load(dataset_name,num_class,paraphraser,"English","doesn't matter",sentiment)

# #df = df.head(1000) # For testing purposes, remove this line for full dataset

# # =========================
# # BATCH INFERENCE FUNCTION
# # =========================
# def compute_entailment_batch(premises, hypotheses):
#     inputs = tokenizer(
#         premises,
#         hypotheses,
#         padding=True,
#         truncation=True,
#         return_tensors="pt"
#     ).to(device)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
#     return probs

# # =========================
# # PROCESS IN BATCHES
# # =========================
# batch_size = 16

# entailment_like_scores = []
# contradiction_scores = []
# labels = []

# for i in tqdm(range(0, len(df), batch_size)):
#     batch = df.iloc[i:i+batch_size]

#     premises = batch["human"].astype(str).tolist()
#     hypotheses = batch[paraphraser].astype(str).tolist()

#     probs = compute_entailment_batch(premises, hypotheses)

#     for p in probs:
#         entailment_like = float(p[0] + p[1])   # entailment + neutral
#         contradiction = float(p[2])
#         entailment_like_scores.append(entailment_like)
#         contradiction_scores.append(contradiction)

#         if entailment_like >= 0.5:
#             labels.append("non-contradiction")
#         else:
#             labels.append("contradiction")

# # =========================
# # ADD RESULTS
# # =========================
# df["entailment_neutral_score"] = entailment_like_scores
# df["contradiction_score"] = contradiction_scores
# df["binary_label"] = labels

# # Save results
# df.to_csv(f"linguistic/entailment/{dataset_name}_{paraphraser}{sentiment}_entailment_results.csv", index=False)

# print(f"Saved to linguistic/entailment/{dataset_name}_{paraphraser}{sentiment}_entailment_results.csv")

# # =========================
# # PLOT DISTRIBUTION
# # =========================
# plt.figure(figsize=(8,5))

# sns.histplot(df["entailment_neutral_score"], bins=30, kde=True)

# plt.title("Distribution of Factual Entailment (Original → Paraphrase)")
# plt.xlabel("Entailment Probability")
# plt.ylabel("Frequency")
# plt.savefig(f"linguistic/entailment/{dataset_name}_{paraphraser}{sentiment}_entailment_distribution.png")
# plt.tight_layout()
# plt.show()



# ---------------------------------- 3 label ------------------------------------------#
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import load_data
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Same colors as SentenceTransformer.py
colors = {
    "positive": "#ff7f0e",  # tab:orange
    "negative": "#040504",  # tab:green
    "neutral": "#d62728"    # tab:red
}


# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL
# =========================
# model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" # multilingual, use if language != English
# model_name = "joeddav/xlm-roberta-large-xnli"
# model_name = "facebook/bart-large-mnli"
# model_name = "microsoft/deberta-v3-large-mnli" # HuggingFace login required
# model_name = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

label_names = ["entailment", "neutral", "contradiction"]
# label_names = ["entailment", "contradiction"]

# =========================
# LOAD DATA
# =========================

dataset_name = 'election_2024' # 'TALLIP''liar_2','liar_6', 'kaggle', 'covid-19'
project = "02-Paraphrasing"
paraphraser = 'gpt' # 'human', 'gpt', 'llama', 'pegasus', 'gemini'
num_class = 2
sentiment_list = ['positive', 'negative', 'neutral']

# =========================
# BATCH INFERENCE FUNCTION
# =========================
def compute_entailment_batch(premises, hypotheses):
    inputs = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs

for sentiment in sentiment_list:
    print(f"Processing sentiment: {sentiment}")

    merged_df, train_df, test_df, valid_df = load_data.data_load(dataset_name,num_class,paraphraser,"en","supervised",sentiment)
    df = test_df
    # df = df.head(500) # For testing purposes, remove this line for full dataset

    # =========================
    # PROCESS IN BATCHES
    # =========================
    batch_size = 16

    entailment_scores = []
    neutral_scores = []
    contradiction_scores = []
    labels = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]

        premises = batch["human"].astype(str).tolist()
        hypotheses = batch[paraphraser].astype(str).tolist()

        probs = compute_entailment_batch(premises, hypotheses)

        for p in probs:
            entailment_scores.append(float(p[0]))
            neutral_scores.append(float(p[1]))
            contradiction_scores.append(float(p[2]))
            labels.append(label_names[p.argmax()])

    # =========================
    # ADD RESULTS
    # =========================
    # Combine entailment + neutral into a single "entailment-like" score, since
    # neutral means the paraphrase doesn't contradict the original either.
    entailment_like_scores = [e + n for e, n in zip(entailment_scores, neutral_scores)]
    binary_labels = ["entailment" if score >= 0.5 else "contradiction" for score in entailment_like_scores]

    df["entailment_score"] = entailment_scores
    df["neutral_score"] = neutral_scores
    df["contradiction_score"] = contradiction_scores
    df["entailment_label"] = labels
    df["entailment_like_score"] = entailment_like_scores
    df["binary_label"] = binary_labels

    # Save results
    df.to_csv(f"results/linguistic/entailment/{dataset_name}_{paraphraser}_{sentiment}_entailment_results_L3.csv", index=False)

    print(f"Saved to linguistic/entailment/{dataset_name}_{paraphraser}_{sentiment}_entailment_results_L3.csv")

    # =========================
    # PLOT 1: ENTAILMENT SCORE DISTRIBUTION
    # =========================
    # Same style as SentenceTransformer.py
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))

    data = df["entailment_like_score"].dropna()
    mean_val = np.mean(data)
    median_val = np.median(data)

    axes.hist(
        data,
        bins=30,
        color=colors[sentiment],
        edgecolor='black',
        alpha=0.7
    )

    # Mean line
    axes.axvline(
        mean_val,
        linestyle='--',
        linewidth=3,
        label=f"Mean = {mean_val:.3f}"
    )

    # Median line
    axes.axvline(
        median_val,
        linestyle=':',
        linewidth=3,
        label=f"Median = {median_val:.3f}"
    )

    #axes.set_title(f"Distribution of Entailment Score (Entailment + Neutral) - {sentiment}")
    axes.set_xlabel("Entailment Probability", fontsize=24)
    axes.set_ylabel("Frequency", fontsize=24)
    axes.tick_params(axis='both', labelsize=22)
    axes.grid(True)
    axes.legend(fontsize=22)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    fig.savefig(f"results/linguistic/entailment/{dataset_name}_{paraphraser}_{sentiment}_entailment_distribution_L3.pdf", format='pdf', bbox_inches='tight')
    time.sleep(5)
    plt.close()

    # =========================
    # PLOT 2: ENTAILMENT VS CONTRADICTION COMPARISON
    # =========================
    counts = df["binary_label"].value_counts().reindex(["entailment", "contradiction"], fill_value=0)
    percentages = counts / counts.sum() * 100

    # Both bars in the sentiment's color; contradiction is hatched to tell them apart
    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    bars = axes.bar(counts.index, counts.values, color=colors[sentiment],
                    edgecolor='black', alpha=0.7, hatch=["", "//"])
    for bar, pct in zip(bars, percentages):
        axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%",
                  ha="center", va="bottom", fontsize=22)

    axes.set_ylim(0, counts.max() * 1.15)  # room for the percentage labels
    axes.set_title(f"Entailment vs Contradiction - {sentiment}", fontsize=24)
    axes.set_xlabel("Label", fontsize=24)
    axes.set_ylabel("Count", fontsize=24)
    axes.tick_params(axis='both', labelsize=22)
    axes.grid(True, axis='y')
    axes.set_axisbelow(True)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    fig.savefig(f"results/linguistic/entailment/{dataset_name}_{paraphraser}_{sentiment}_entailment_vs_contradiction_L3.pdf", format='pdf', bbox_inches='tight')
    time.sleep(5)
    plt.close()

