import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import load_data
import re
import matplotlib.pyplot as plt
import numpy as np
import time

colors = {
    "positive":"#ff7f0e",            # tab:orange
    "negative": "#2ca02c",            # tab:green
    "neutral": "#d62728"  # tab:red
}



def compute_semantic_similarity(df, paraphraser,sentiment):
    """
    Computes cosine similarity between 'text' and 'paraphrased_text' columns of a DataFrame.
    Adds a new column 'semantic_similarity' with the results.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode all texts and paraphrases in batch (faster)
    embeddings_text = model.encode(df['human'].tolist(), convert_to_tensor=True)
    embeddings_paraphrased = model.encode(df[paraphraser].tolist(), convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(embeddings_text, embeddings_paraphrased).diagonal()
    df["semantic_similarity"] = similarities.cpu().numpy()

    average = df['semantic_similarity'].mean()
    minimum = df['semantic_similarity'].min()
    maximum = df['semantic_similarity'].max()
    value_range = maximum - minimum
    std_dev = df['semantic_similarity'].std()
    median = df['semantic_similarity'].median()

    # Print the results
    print(f"Average: {average}")
    print(f"Min: {minimum}")
    print(f"Max: {maximum}")
    print(f"Range: {value_range}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Median: {median}")

    fig, axes = plt.subplots(1, 1, figsize=(12, 4), sharex=True, sharey=True)

    data = df['semantic_similarity'].dropna()

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

    axes.set_ylabel('Frequency', fontsize=24)
    axes.tick_params(axis='both', labelsize=22)
    axes.grid(True)
    axes.legend(fontsize=22)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
# Adjust layout
    plt.tight_layout()
    fig.savefig(f"results/semantics/{dataset_choice}_{paraphraser}_{sentiment}_semantic_similarity.pdf", format='pdf', bbox_inches='tight')
    time.sleep(5)  # Ensure the file is saved before closing
    plt.close()
    


    return df


if __name__ == "__main__":
    # Example: Load from a CSV (replace with your file path)

    sentiment_list = ['positive', 'negative', 'neutral']

    for sentiment in sentiment_list:
        print(f"Processing sentiment: {sentiment}")
        paraphraser = 'gpt'
        dataset_choice = 'election_2024'
        num_classes = 2
        language = 'en'
        classifier = 'supervised'
        sentiment = sentiment  # 'positive', '_negative', '_neutral'


        full_df, train_df, test_df, valid_df =  load_data.data_load(dataset_choice,num_classes,paraphraser,language,classifier,sentiment)
        print(full_df.head())

        df=full_df.dropna()

        # Compute similarity and save
        output_path = f'results/semantics/{dataset_choice}_{paraphraser}_{sentiment}_semantic_similarity.csv'
        df = compute_semantic_similarity(df,paraphraser,sentiment)
        df.to_csv(output_path, index=False)
        
        print(f"Semantic similarities saved to: {output_path}")


    # paraphraser = 'llama'
    # dataset_choice = 'isot'
    # num_classes = 2
    # language = 'en'
    # classifier = 'supervised'
    # sentiment = 'positive'  # 'positive', '_negative', '_neutral'


    # full_df, train_df, test_df, valid_df =  load_data.data_load(dataset_choice,num_classes,paraphraser,language,classifier,sentiment)
    # print(full_df.head())

    # df=full_df.dropna()

    # # Compute similarity and save
    # output_path = f'results/semantics/{dataset_choice}_{paraphraser}_{sentiment}_semantic_similarity.csv'
    # df = compute_semantic_similarity(df,paraphraser,sentiment)
    # df.to_csv(output_path, index=False)
    
    # print(f"Semantic similarities saved to: {output_path}")
