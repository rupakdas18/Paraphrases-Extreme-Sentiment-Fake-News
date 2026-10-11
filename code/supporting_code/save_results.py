"""
This file saves the classification results in a csv file

"""


import pandas as pd
import os
import variables


def save_classification_results(accuracy, precision, recall, f1):

    metrics = {
    "accuracy": accuracy,
    "f1": f1,
    "precision": precision,
    "recall": recall
}

    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(f"results/{variables.classifier}/{variables.dataset_name}_{variables.paraphraser}{variables.sentiment}_{variables.classifier}_summary_results.csv", index=False)
