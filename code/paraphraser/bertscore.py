"""
Created on Sun Sep 29 23:25:13 2024

"""
'''
BERT_score
https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit?gid=0#gid=0
'''

import pandas as pd
import bert_score
import matplotlib as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import load_data


def calculate_bertscore(df):
    # Compute BERTScore for each pair of original and paraphrased sentences
    original_sentences = df['text'].tolist()
    #original_sentences = original_sentences[:500]
    paraphrased_sentences = df['paraphrased_text'].tolist() # Select the model
    #paraphrased_sentences = paraphrased_sentences[:500]

    P, R, F1 = bert_score.score(original_sentences, paraphrased_sentences, lang="en", model_type='microsoft/deberta-large', rescale_with_baseline=True, verbose=True)
    
    P_list = [round(num, 4) for num in P.tolist()]
    R_list = [round(num, 4) for num in R.tolist()]
    F1_list = [round(num, 4) for num in F1.tolist()]

    for i, val in enumerate(F1_list):
        df.loc[i, "bertscore_p"] = P_list[i]
        df.loc[i, "bertscore_r"] = R_list[i]
        df.loc[i, "bertscore_f1"] = F1_list[i]

    #df_selected = df[['id', 'text', 'paraphrased_text', 'bertscore_p', 'bertscore_r', 'bertscore_f1']]

    average = df['bertscore_f1'].mean()
    minimum = df['bertscore_f1'].min()
    maximum = df['bertscore_f1'].max()
    value_range = maximum - minimum
    std_dev = df['bertscore_f1'].std()
    median = df['bertscore_f1'].median()

    # Print the results
    print(f"Average: {average}")
    print(f"Min: {minimum}")
    print(f"Max: {maximum}")
    print(f"Range: {value_range}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Median: {median}")

    # Export the selected columns to a CSV file
    df.to_csv(f'Dataset/{dataset_choice}/{paraphraser}/{dataset_choice}_{paraphraser}_bertscore_2.csv', index=False)





"""## Main Code"""
if __name__ == "__main__":

    paraphraser = 'gpt'
    dataset_choice = 'covid-19'
    num_classes = 2
    language = 'en'
    classifier = 'supervised'

    full_df, train_df, test_df, valid_df =  load_data.data_load(dataset_choice,num_classes,paraphraser,language,classifier)
    print(full_df.head())

    df=full_df.dropna()

    # Compute BERTScore for each pair of original and paraphrased sentences
    calculate_bertscore(df)




    

    




    


