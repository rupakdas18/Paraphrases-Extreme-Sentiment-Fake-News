"""
Source: https://www.kaggle.com/code/makatronium/gemini-ai-paraphrasing-capabilities
"""




import os
import sys
import pandas as pd
import google.generativeai as genai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import load_data
import re
import time
import print_dataset



def truncate_to_500_tokens(text):
    """
    Truncates the input text to approximately the first 500 tokens (words).
    """
    tokens = re.findall(r'\b\w+\b', text)
    truncated_text = ' '.join(tokens[:500])
    return truncated_text


def load_gemini_api_key():
    """
    Loads the Gemini API key from environment variables and configures the generative AI client.
    """
    api_key = os.getenv('gemini_api_key')
    if api_key is None:
        raise ValueError("GEMINI_API_KEY environment variable not found.")
    else:
        print("GEMINI_API_KEY environment variable found. Proceeding with Gemini API configuration.")

    genai.configure(api_key=api_key)

def safe_generate(model, prompt, retries=3, delay=5):
    """Safely generate text from Gemini with retries and exponential backoff."""
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            if response.candidates:
                return response.text.strip()
            else:
                print("Warning: No candidates returned.")
                return ""
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(delay * (2 ** attempt))  # Exponential backoff
    print("All retries failed.")
    return ""


# def paraphrase_texts(df):
#     """
#     Paraphrase the 'text' column of a DataFrame using Gemini and return the result.
#     """
#     model = genai.GenerativeModel('gemini-1.5-flash')

#     original_texts = df['text'].tolist()
#     paraphrased_texts = []

#     for idx, text in enumerate(original_texts):
#         truncated_text = truncate_to_500_tokens(text)
#         prompt = (
#             "Paraphrase the following text and make it more syntactically complex and more lexically diverse "
#             "(using less frequent words) while keeping the original meaning (semantics) and the overall length of the text same. "
#             "Just provide the paraphrased version without any additional explanation.\n\n"
#             f"{truncated_text}"
#         )

#         paraphrased = safe_generate(model, prompt)
#         paraphrased_texts.append(paraphrased)

#         if idx % 10 == 0:
#             print(f"Processed {idx + 1} / {len(original_texts)} samples.")
#         time.sleep(1)

#     result_df = df.copy()
#     result_df['paraphrased_text'] = paraphrased_texts
#     return result_df

def paraphrase_texts(df, output_path):
    #model = genai.GenerativeModel('gemini-1.5-flash')
    model = genai.GenerativeModel('gemma-3-27b-it')

    # Load existing output if resuming
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        print(f"Found existing file with {len(existing_df)} samples. Resuming from last checkpoint.")
        already_done_indices = set(existing_df.index)
    else:
        existing_df = pd.DataFrame()
        already_done_indices = set()

    all_rows = []
    for idx, row in df.iterrows():
        if idx in already_done_indices:
            continue  # Already processed

        text = row['text']
        truncated_text = truncate_to_500_tokens(text)
        prompt = (
            "Paraphrase the following text and make it more syntactically complex and more lexically diverse "
            "(using less frequent words) while keeping the original meaning (semantics) and the overall length of the text same. "
            "Just provide the paraphrased version without any additional explanation.\n\n"
            f"{truncated_text}"
        )

        paraphrased = safe_generate(model, prompt)
        row_data = row.to_dict()
        row_data['paraphrased_text'] = paraphrased
        all_rows.append(row_data)

        # Append to CSV every 10 samples
        if len(all_rows) % 10 == 0 or idx == df.index[-1]:
            temp_df = pd.DataFrame(all_rows)
            mode = 'a' if os.path.exists(output_path) else 'w'
            header = not os.path.exists(output_path)
            temp_df.to_csv(output_path, mode=mode, index=False, header=header)
            print(f"✅ Saved {len(temp_df)} new rows (up to index {idx})")
            all_rows = []

        time.sleep(1)

    print(f"\n✅ Paraphrasing complete. Final output: {output_path}")
    

    



if __name__ == "__main__":
    dataset_choice = 'election_2024'
    num_classes = 2
    paraphraser = 'gemini'
    language = 'English'
    classifier = 'n/a'

    full_df, train_df, test_df, valid_df = load_data.data_load(
        dataset_choice, num_classes, paraphraser, language, classifier
    )

    # Slice dataset
    work_df = train_df
    print("Total number of samples in the dataset:", len(work_df))
    output_path = f'Dataset/{dataset_choice}/gemini/{dataset_choice}_gemini.csv'
l
    # Load Gemini and run
    load_gemini_api_key()
    paraphrase_texts(work_df, output_path)
    import sound

