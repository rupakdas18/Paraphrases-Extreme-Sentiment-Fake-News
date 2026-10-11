"""
Universal Zero-Shot Text Classification for Decoder-Only Instruct Models
------------------------------------------------------------------------
Supports models such as:
- Meta Llama 3 / 3.1 / 3.2 Instruct
- Microsoft Phi-3 / Phi-3.5 Instruct
- Falcon 3 Instruct
- Mistral Instruct
- Qwen Instruct
- Gemma Instruct / IT
- Other decoder-only chat/instruct models

Key idea:
- Use a standard message structure: system + user
- Use tokenizer.apply_chat_template(...) when available
- Fall back to a plain text prompt otherwise
"""

import os
import re
import random
import warnings
import difflib
from datetime import datetime
from typing import List, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import variables
import load_data
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")
HF_TOKEN = os.environ.get("HF_TOKEN")  # set HF_TOKEN in your environment for gated models

# =========================================================
# CONFIGURATION
# =========================================================

variables.dataset_name = 'election_2024' # kaggle, liar_2, liar_6, covid-19
variables.paraphraser = 'gpt' #  'parrot' or 'bard', 'gpt','pegasus','llama'
language = 'English'
#variables.sentiment = '_neutral' # '_positive' or '_negative'

if variables.dataset_name == 'liar':
    num_labels = 6
else:   
    num_labels = 2
project = "02_Paraphrasing"
testing_program = False

sentiment_list = ['positive', 'negative', 'neutral']  # Add an empty string for no sentiment filtering
#sentiment_list = ['']
#sentiment_list = ['_negative_neutralized_v2']
# model_list = ['falcon3','mistral','qwen','llama3','gemma','phi35']
# model_list = ['phi35']
model_list = ["phi35",'mistral','qwen']



    

# Choose any model key below
#SELECTED_MODEL = "mistral"

RESULTS_SUMMARY_PATH = "results/zero_shot/all_zero_shot_results.csv"

MODEL_OPTIONS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "gemma": "google/gemma-2-2b-it",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama31": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama32": "meta-llama/Llama-3.2-3B-Instruct",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "phi35": "microsoft/Phi-3.5-mini-instruct",
    "falcon3": "tiiuae/Falcon3-7B-Instruct",
}

label_col = "label"


 

MAX_INPUT_TOKENS = 3072
MAX_NEW_TOKENS = 8
USE_4BIT = True
RANDOM_SEED = 42
SAVE_EVERY_N = 100

PRINT_RENDERED_PROMPT = False
PRINT_RAW_OUTPUT = False


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# MODEL LOADING
# =========================================================

def load_model_and_tokenizer(model_name: str):
    """
    Load tokenizer and causal LM for inference.
    """
    print(f"\nLoading model: {model_name}")

    quant_config = None
    if USE_4BIT and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=HF_TOKEN,
    )

    model.eval()
    return tokenizer, model




# =========================================================
# PROMPT BUILDING
# =========================================================

def build_task_messages(text: str, labels: List[str]):
    """
    Standard cross-model message structure.
    Let the tokenizer render model-specific control tokens.
    """
    label_string = ", ".join(labels)

    system_content = (
        "You are a text classification system for misinformation detection. Your job is to identify wheater a text is FAKE or REAL."
        f"Choose exactly one label from: {label_string}. "
        "Return only the label. "
        "Do not explain your answer. "
        "Do not output any extra words, punctuation, or sentences. "
        "Do not create new labels."
    )

    user_content = (
        "Classify the following text.\n\n"
        f"Text: {text}\n\n"
        "Label:"
    )

    if "gemma" in model_list:
        return [
            {
                "role": "user",
                "content": system_content + "\n\n" + user_content,
            }
    ]

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_plain_fallback_prompt(text: str, labels: List[str]) -> str:
    """
    Fallback prompt for models without a chat template.
    """
    label_string = ", ".join(labels)

    prompt = f"""
You are a text classification system for misinformation detection.

Choose exactly one label from: {label_string}
Return only the label.
Do not explain your answer.
Do not output any extra words, punctuation, or sentences.
Do not create new labels.

Classify the following text.

Text: {text}

Label:
""".strip()

    return prompt


def prepare_model_inputs(tokenizer, text: str, labels: List[str]):
    """
    Robustly prepare inputs across model families.

    We use:
    1. tokenizer.apply_chat_template(..., tokenize=False) if available
    2. tokenizer(...) on the rendered string
    3. plain prompt fallback if no chat template exists

    This avoids the Tensor vs BatchEncoding inconsistency that can happen
    with tokenize=True across some models/tokenizers/processors.
    """
    has_chat_template = (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None) is not None
    )

    if has_chat_template:
        messages = build_task_messages(text, labels)

        rendered_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if PRINT_RENDERED_PROMPT:
            print("\n===== RENDERED CHAT PROMPT =====")
            print(rendered_text)
            print("================================\n")

        tokenized = tokenizer(
            rendered_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            add_special_tokens=False,  # important after apply_chat_template(tokenize=False)
        )
        return tokenized

    prompt = build_plain_fallback_prompt(text, labels)

    if PRINT_RENDERED_PROMPT:
        print("\n===== PLAIN PROMPT =====")
        print(prompt)
        print("========================\n")

    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        add_special_tokens=True,
    )
    return tokenized

def save_metrics_to_csv(
    dataset: str,
    paraphraser: str,
    sentiment: str,
    selected_model: str,
    shots: str,
    accuracy: float,
    f1: float,
    precision: float,
    recall: float,
    output_csv_path: str = RESULTS_SUMMARY_PATH,
):
    """
    Append one experiment's overall metrics as a row in the summary CSV.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    new_result = pd.DataFrame([{
        "dataset": dataset,
        "paraphraser": paraphraser,
        "sentiment": sentiment,
        "model": selected_model,
        "shots": shots,
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "written_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }])

    write_header = not os.path.exists(output_csv_path)
    new_result.to_csv(output_csv_path, mode="a", header=write_header, index=False)
    print(f"Summary metrics appended to: {output_csv_path}")


# =========================================================
# OUTPUT CLEANING / LABEL MATCHING
# =========================================================

def clean_text(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-zA-Z0-9\s_-]", "", value)
    return value.strip()


def extract_label_from_generation(generated_text: str, labels: List[str]) -> Tuple[str, str]:
    """
    Map raw generation into one of the allowed labels only.
    Returns (predicted_label, strategy_used).
    """
    cleaned_prediction = clean_text(generated_text)
    cleaned_labels = [clean_text(label) for label in labels]

    # remove common wrappers
    prefixes = [
        "label:",
        "answer:",
        "prediction:",
        "the label is",
        "the correct label is",
    ]
    for prefix in prefixes:
        if cleaned_prediction.startswith(prefix):
            cleaned_prediction = cleaned_prediction[len(prefix):].strip()

    cleaned_prediction = cleaned_prediction.split("\n")[0].strip()
    cleaned_prediction = cleaned_prediction.split(".")[0].strip()

    # 1. exact match
    for original, cleaned in zip(labels, cleaned_labels):
        if cleaned_prediction == cleaned:
            return original, "exact"

    # 2. substring match
    for original, cleaned in zip(labels, cleaned_labels):
        if cleaned in cleaned_prediction or cleaned_prediction in cleaned:
            return original, "substring"

    # 3. fuzzy match with safer cutoff
    closest = difflib.get_close_matches(
        cleaned_prediction,
        cleaned_labels,
        n=1,
        cutoff=0.4,
    )
    if closest:
        idx = cleaned_labels.index(closest[0])
        return labels[idx], "fuzzy"

    # 4. final fallback
    return labels[0], "fallback"


# =========================================================
# PREDICTION
# =========================================================

def predict_label(text: str, labels: List[str], tokenizer, model) -> str:
    """
    Predict one label for one text sample.
    """
    model_inputs = prepare_model_inputs(tokenizer, text, labels)

    if torch.cuda.is_available():
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    input_len = model_inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[:, input_len:]
    generated_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    predicted_label, strategy = extract_label_from_generation(generated_text, labels)

    if PRINT_RAW_OUTPUT:
        print(f"TEXT: {text[:120]}...")
        print(f"RAW OUTPUT: {generated_text}")
        print(f"PREDICTED LABEL: {predicted_label} | STRATEGY: {strategy}")
        print("-" * 80)

    return predicted_label


# =========================================================
# EVALUATION
# =========================================================

def evaluate_predictions(true_labels: List[str], predicted_labels: List[str], SELECTED_MODEL: str, sentiment: str):
    """
    Evaluate classification performance.
    """
    print("\n===== EVALUATION METRICS =====")

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )
    recall = recall_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )
    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(true_labels, predicted_labels, zero_division=0))

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(true_labels, predicted_labels))


    save_metrics_to_csv(
        dataset = variables.dataset_name,
        paraphraser = variables.paraphraser,
        sentiment = sentiment,
        selected_model=SELECTED_MODEL,
        shots='zero-shot',
        accuracy=accuracy,
        f1=f1,
        precision=precision,
        recall=recall,
    )


# =========================================================
# SAVE / RESUME
# =========================================================

def get_resume_offset(output_path: str) -> int:
    """
    Number of rows already predicted and saved at output_path, so a run
    can resume from where a previous, interrupted run left off instead
    of starting over.
    """
    if not os.path.exists(output_path):
        return 0
    return len(pd.read_csv(output_path))


def append_predictions(df_chunk: pd.DataFrame, predictions_chunk: List[str], output_path: str):
    """
    Append one chunk of predictions to output_path, writing the header
    only if the file doesn't exist yet.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chunk = df_chunk.copy()
    chunk["predicted_label"] = predictions_chunk
    write_header = not os.path.exists(output_path)
    chunk.to_csv(output_path, mode="a", header=write_header, index=False)
    print(f"Saved {len(chunk)} predictions to: {output_path}")


# =========================================================
# MAIN
# =========================================================

def main():

    for sentiment in sentiment_list:
        for SELECTED_MODEL in model_list:

            try:

                set_seed(RANDOM_SEED)

                full_df, train_df, test_df, valid_df = load_data.data_load(variables.dataset_name,num_labels,variables.paraphraser,language,'zero-shot',sentiment)
                
                df = test_df.copy()

                if testing_program:
                    df = test_df.head()
                text_col = variables.paraphraser

                OUTPUT_FILE_PATH = f"results/zero_shot/{variables.dataset_name}_{variables.paraphraser}_{sentiment}_{SELECTED_MODEL}_zs_results.csv"


                if SELECTED_MODEL not in MODEL_OPTIONS:
                    raise ValueError(
                        f"Invalid model choice: {SELECTED_MODEL}. "
                        f"Available: {list(MODEL_OPTIONS.keys())}"
                    )

                model_name = MODEL_OPTIONS[SELECTED_MODEL]

                # load dataset

                df = df.dropna(subset=[text_col, label_col]).copy()
                df = df.reset_index(drop=True)

                print("Dataset size:", df.shape)

                # candidate labels from dataset
                labels = sorted(df[label_col].astype(str).unique().tolist())
                print(f"\nDetected Labels: {labels}")

                print("\nDataset examples:")
                print(df[[text_col, label_col]].head())

                # resume from the last save point instead of starting over
                resume_offset = get_resume_offset(OUTPUT_FILE_PATH)
                if resume_offset >= len(df):
                    print(f"Already completed: {OUTPUT_FILE_PATH} ({resume_offset}/{len(df)} rows). Skipping.")
                    final_df = pd.read_csv(OUTPUT_FILE_PATH)
                    evaluate_predictions(
                        true_labels=final_df[label_col].astype(str).tolist(),
                        predicted_labels=final_df["predicted_label"].astype(str).tolist(),
                        SELECTED_MODEL=SELECTED_MODEL, sentiment=sentiment
                    )
                    continue
                if resume_offset > 0:
                    print(f"Resuming from row {resume_offset}/{len(df)} (found existing partial results at {OUTPUT_FILE_PATH})")

                remaining_df = df.iloc[resume_offset:].reset_index(drop=True)

                # load model
                print("\nLoading model...")
                tokenizer, model = load_model_and_tokenizer(model_name)
                print("Model loading done.")

                rows_buffer = []
                predictions_buffer = []

                for pos, (_, row) in enumerate(remaining_df.iterrows(), start=1):
                    text = str(row[text_col]).strip()

                    pred = predict_label(
                        text=text,
                        labels=labels,
                        tokenizer=tokenizer,
                        model=model,
                    )

                    rows_buffer.append(row)
                    predictions_buffer.append(pred)

                    total_done = resume_offset + pos
                    if (total_done % 10) == 0:
                        print(f"Processed {total_done}/{len(df)} samples")

                    if len(predictions_buffer) >= SAVE_EVERY_N:
                        append_predictions(pd.DataFrame(rows_buffer), predictions_buffer, OUTPUT_FILE_PATH)
                        rows_buffer = []
                        predictions_buffer = []

                # flush any remaining rows that didn't fill a full chunk
                if predictions_buffer:
                    append_predictions(pd.DataFrame(rows_buffer), predictions_buffer, OUTPUT_FILE_PATH)

                # evaluate on the full saved file (covers this run + any earlier resumed chunks)
                final_df = pd.read_csv(OUTPUT_FILE_PATH)
                evaluate_predictions(
                    true_labels=final_df[label_col].astype(str).tolist(),
                    predicted_labels=final_df["predicted_label"].astype(str).tolist(),
                    SELECTED_MODEL=SELECTED_MODEL, sentiment=sentiment
                )
            except Exception as e:
                print(f"{SELECTED_MODEL} is not working: {e}")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()