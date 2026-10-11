import openai
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import load_data
import time
import pandas as pd


# Set up your OpenAI API key
openai.api_key = os.environ.get('openai_api_key')



def safe_generate(prompt, retries=3, delay=5):
    """Safely generate text from GPT with retries and exponential backoff."""
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that paraphrases text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7,
            )

            if 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0]['message']['content']
            else:
                print("Warning: No choices returned.")
                return ""

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(delay * (2 ** attempt))

    print("All retries failed.")
    return ""



def already_done_ids(df, output_path):
    """Ids (original dataset row numbers, the index of df) already in the output file."""
    if not os.path.exists(output_path):
        return set()
    existing_df = pd.read_csv(output_path)
    if 'id' not in existing_df.columns:
        # Older files were saved without an id: recover it from the human text and add it
        # to the file once, so that the rows appended below line up with its columns
        text_to_id = {}
        for idx, text in df['human'].items():
            text_to_id.setdefault(str(text), idx)
        existing_df.insert(0, 'id', existing_df['human'].astype(str).map(text_to_id).astype('Int64'))
        existing_df.to_csv(output_path, index=False)
        print(f"Added an 'id' column to {output_path} "
              f"({existing_df['id'].notna().sum()} of {len(existing_df)} rows matched by text).")
    done = set(existing_df['id'].dropna().astype(int))
    print(f"Found existing file with {len(existing_df)} rows ({len(done)} ids already done). Resuming.")
    return done


def save_rows(rows, output_path):
    """Append rows to the output CSV, writing the header only for a new file."""
    new_df = pd.DataFrame(rows)
    if os.path.exists(output_path):
        columns = pd.read_csv(output_path, nrows=0).columns  # keep the existing column order
        new_df.reindex(columns=columns).to_csv(output_path, mode='a', index=False, header=False)
    else:
        new_df.to_csv(output_path, index=False)


def paraphrase_text(df,output_path, sentiment):

    # Skip ids that are already in the output file (resuming an earlier run)
    already_done = already_done_ids(df, output_path)

    all_rows = []
    for idx, row in df.iterrows():
        if idx in already_done:
            print(f"Skipping id {idx}: already paraphrased")
            continue
        print(f"Processing id {idx}...")

        text = row['human']

        if sentiment == 'positive':

          prompt = (
                  f"""Paraphrase the text below with the provided constraints:
                      - Make the sentiment of the text as POSITIVE as possible.
                      - Keep the same semantic meaning as the original text.
                      - Keep the syntactic complexity (for example: flesch readability score, dependency tree depth, complex word ratio, etc.) similar to the original text. Do not make it more complex or simpler.
                      - Do not change the length of the text significantly.
                      - Do not change the factual information in the text. Keep all facts the same as the original text.
                      - Return only the paraphrased text, no explanations or formatting.
                      
                      Text: {text}
                      """
              )
          
        elif sentiment == 'negative':
            
          prompt = (
                  f"""Paraphrase the text below with the provided constraints:
                      - Make the sentiment of the text more NEGATIVE compared to the original.
                      - Do not change the factual information. Keep all facts the same and high entailment score with the original text.
                      - Keep the same semantic meaning as the original text.
                      - Keep the syntactic complexity (for example: flesch readability score, dependency tree depth, complex word ratio, etc.) similar to the original text. Do not make it more complex or simpler.
                      - Do not change the length of the text significantly.
                      - Return only the paraphrased text, no explanations or formatting.
                      
                      Text: {text}
                      """
              )
        elif sentiment == 'neutral':
            
          prompt = (
                  f"""Paraphrase the text below with the provided constraints:
                      - Keep the sentiment of the text LIKE THE ORIGINAL as possible.
                      - Keep the same semantic meaning as the original text.
                      - Keep the syntactic complexity (for example: flesch readability score, dependency tree depth, complex word ratio, etc.) similar to the original text. Do not make it more complex or simpler.
                      - Do not change the length of the text significantly.
                      - Do not change the factual information in the text. Keep all facts the same as the original text.
                      - Return only the paraphrased text, no explanations or formatting.
                      
                      Text: {text}
                      """
              )
          

        paraphrased = safe_generate(prompt)

        row_data = {'id': idx, **row.to_dict()}  # keep the original dataset row number
        row_data['gpt'] = paraphrased
        all_rows.append(row_data)

        # Append to CSV every 20 samples
        if len(all_rows) == 20:
            save_rows(all_rows, output_path)
            print(f"✅ Saved {len(all_rows)} new rows (up to id {idx})")
            all_rows = []

        #time.sleep(2)

    if all_rows:  # the last partial batch
        save_rows(all_rows, output_path)
        print(f"✅ Saved {len(all_rows)} new rows")

    print(f"\n✅ Paraphrasing complete. Final output: {output_path}")




dataset_choice = 'election_2024'
num_classes = 6
input_paraphraser = 'human'
target_paraphraser = 'gpt'
language = 'English'
classifier = 'n/a'
source_sentiment = ''
#target_sentiment = 'positive' # Change this to '_positive', '_negative', or '_neutral' as needed   

sentiment_list = ['neutral']
split_list = ['train']

full_df, train_df, test_df, valid_df = load_data.data_load(dataset_choice, num_classes, input_paraphraser, language, classifier, source_sentiment)
split_dfs = {'train': train_df, 'test': test_df, 'valid': valid_df}

for target_sentiment in sentiment_list:
    for split_name in split_list:
        work_df = split_dfs[split_name]
        print(f"\n=== sentiment={target_sentiment} | split={split_name} ===")
        print("Total number of samples in the dataset:", len(work_df))

        output_path = f'Dataset/{dataset_choice}/{target_paraphraser}/{dataset_choice}_{target_paraphraser}_{split_name}_{target_sentiment}.csv'
        paraphrase_text(work_df, output_path, target_sentiment)

