import pandas as pd
import time
import openai
import os
import sys
import re
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import load_data

openai.api_key = os.environ.get('openai_api_key')  # set openai_api_key in your environment

# Define the zero-shot prompt
def create_prompt_zero_shot(text,class_numbe):

  if class_numbe == 6:
    prompt = f"""
Classify the following text into one of these categories: 'half-true','false','mostly-true','barely-true','true','pants-fire'. Just provide the category without any additional text. all the labels are case insensitive.\n\n

Text: "{text}"
Category:
"""
  elif class_numbe == 2:
    prompt = f"""
Classify the following text into one of these two categories: 'fake' or 'true'. Just provide the category without any additional text. all the labels are case insensitive.\n\n

Text: "{text}"
Category:
"""

  #print(prompt)

  return prompt


def classify_text_zs(text,class_number):
    prompt = create_prompt_zero_shot(text,class_number)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4", "gpt-3.5-turbo-16k", etc.
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Please classify the below statement as Fake or True. Please just return the class no other text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5
    )
    category = response['choices'][0]['message']['content'].strip()
    return category.lower()

def classify_text(text, few_shot_examples,dataset_choice):
    prompt = create_prompt_few_shot(text, few_shot_examples, dataset_choice)
    #print(prompt)l
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4", "gpt-3.5-turbo-16k", etc.
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5
    )
    category = response['choices'][0]['message']['content'].strip()
    return category





def create_prompt_few_shot(text, few_shot_examples,dataset_choice):


  if dataset_choice == 'liar':
    prompt = "Classify the following text into one of these categories: 'half-true','false','mostly-true','barely-true','true','pants-fire'. Just provide the category without any additional text. all the labels are case insensitive.\n\n"
    for example_text, example_label in few_shot_examples:
        prompt += f"Example:\nText: \"{example_text[:500]}\"\nCategory: {example_label}\n\n"
    prompt += f"Now, classify this text:\nText: \"{text[:500]}\"\nCategory:"


  else:

    prompt = "Classify the following text into one of these two categories: 'FAKE' or 'TRUE'. Just provide the category without any additional text. all the labels are case insensitive.\n\n"
    for example_text, example_label in few_shot_examples:
        prompt += f"Example:\nText: \"{example_text[:500]}\"\nCategory: {example_label}\n\n"
    prompt += f"Now, classify this text:\nText: \"{text[:500]}\"\nCategory:"

    print(prompt)


  return prompt

def calculate_accuracy(resulted_df):


  from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

  # Load the dataframe (assuming df already exists with 'labels' and 'predicted_labels' columns)
  # df = pd.read_csv('classified_dataset.csv')  # Uncomment if you need to load the dataframe from CSV

  # Calculate metrics

  print(resulted_df)

  accuracy = accuracy_score(resulted_df['label'], resulted_df['predicted_label'])
  f1 = f1_score(resulted_df['label'], resulted_df['predicted_label'], average='weighted')
  precision = precision_score(resulted_df['label'], resulted_df['predicted_label'], average='weighted')
  recall = recall_score(resulted_df['label'], resulted_df['predicted_label'], average='weighted')
  report = classification_report(resulted_df['label'], resulted_df['predicted_label'])

  # Print metrics
  print(f"Accuracy: {accuracy:.3f}")
  print(f"F1 Score: {f1:.3f}")
  print(f"Precision: {precision:.3f}")
  print(f"Recall: {recall:.3f}")
  print("\nClassification Report:\n", report)



def classification(dataset_choice,paraphraser,filename,few_shot_value,train_df,test_df,num_classes):
   # test_df = test_df.head(10)

    # print(full_df.head())

    # if few_shot_value == 'zero_shot':
    # # Loop through each row in the dataset and classify the text
    #     predicted_labels = []
    #     for index, row in test_df.iterrows():
    #         text = row[paraphraser]
    #         predicted_label = classify_text_zs(text,num_classes)
    #         predicted_labels.append(predicted_label)

    #     # Assign the predicted labels to the dataframe
    #     test_df['predicted_label'] = predicted_labels

    #     # Save the results to a new CSV file
    #     test_df.to_csv(f'results/{dataset_choice}_{paraphraser}_zs_results.csv', index=False)

    #     print("Classification completed")
    #     print(test_df)

    # else:


    # # Generate few-shot examples
    #     few_shot_examples = []

    #     for label in train_df['label'].unique():
    #         examples = train_df[train_df['label'] == label].sample(n=few_shot_value, random_state=42)
    #         few_shot_examples.extend(zip(examples['human'], examples['label']))

    #     #print(few_shot_examples)

    #         # Use a for loop to classify text in test_df
    #         predicted_labels = []
    #         count = 0
    #         for text in test_df['human']:

    #             predicted_label = classify_text(text, few_shot_examples,dataset_choice)
    #             predicted_labels.append(predicted_label)
    #             time.sleep(1)  # Add a delay of 1 second between each classification
    #             count += 1
    #             print(count)


    #         test_df['predicted_label'] = predicted_labels

    BATCH_SIZE = 10
    RESULTS_PATH = filename

    # --- Handle few-shot examples if needed ---
    few_shot_examples = None
    if few_shot_value != 'zero_shot':
        few_shot_examples = []
        for label in train_df['label'].unique():
            examples = train_df[train_df['label'] == label].sample(n=int(few_shot_value), random_state=42)
            few_shot_examples.extend(zip(examples['human'], examples['label']))
        print(f"Generated {len(few_shot_examples)} few-shot examples.")

    # --- Load existing results (if any) ---
    if os.path.exists(RESULTS_PATH):
        print(f"Resuming from existing results: {RESULTS_PATH}")
        result_df = pd.read_csv(RESULTS_PATH)
    else:
        result_df = pd.DataFrame(columns=test_df.columns.tolist() + ['predicted_label'])

    # --- Determine which samples remain ---
    processed_indices = set(result_df.index)
    remaining_df = test_df.loc[~test_df.index.isin(processed_indices)]

    print(f"Already processed: {len(processed_indices)} samples")
    print(f"Remaining: {len(remaining_df)} samples")

    # --- Classification loop (unified for zero-shot and few-shot) ---
    predicted_batch = []

    for i, (index, row) in enumerate(remaining_df.iterrows(), start=1):
        if few_shot_value == 'zero_shot':
            text = row[paraphraser]
            predicted_label = classify_text_zs(text, num_classes)
        else:
            text = row['human']
            predicted_label = classify_text(text, few_shot_examples, dataset_choice)
            time.sleep(1)  # Optional delay for API rate limits

        predicted_batch.append((index, predicted_label))

        # --- Save every batch ---
        if i % BATCH_SIZE == 0 or i == len(remaining_df):
            print(f"Saving progress... processed {i}/{len(remaining_df)} new samples")

            # Update DataFrame and save
            for idx, label in predicted_batch:
                test_df.at[idx, 'predicted_label'] = label

            updated_df = pd.concat([result_df, test_df.loc[[idx for idx, _ in predicted_batch]]])
            updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
            updated_df.to_csv(RESULTS_PATH, index=False)

            # Reset batch & result cache
            result_df = updated_df
            predicted_batch = []

    print("✅ Classification completed successfully.")
    print(result_df)

    return result_df




def main():

    dataset_choice = 'election_2024' # kaggle, liar_2, liar_6, covid-19, TALLIP, election_2024
    language = 'English'
    paraphraser = 'gemini' #  'parrot' or 'bard', 'gpt','pegasus','llama'
    few_shot_value = 2 # 2 shot, 4 shot, 6 shot, 8 shot

    


    num_classes = 2
    text_length = 200 # The number of words to be used in the text. If the text is less than this number, it will be padded with 0. If it is more than this number, it will be truncated.
    classifier = 'gpt_n_shot'

    filename = f"results/{few_shot_value}_shot/{dataset_choice}_{paraphraser}_{few_shot_value}_shot_results.csv"

    full_df, train_df, test_df, valid_df =  load_data.data_load(dataset_choice,num_classes,paraphraser,language,classifier)

    test_df = test_df.head(5)

    resulted_df = classification(dataset_choice,paraphraser, filename, few_shot_value,train_df,test_df,num_classes)
    calculate_accuracy(resulted_df)

    

    


if __name__==main():
   main()



