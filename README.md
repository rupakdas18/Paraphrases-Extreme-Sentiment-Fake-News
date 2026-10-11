# Paraphrases that Exhibit Extreme Sentiment Polarity Reveal Vulnerability in Fake News Detection

Code and data for studying how the sentiment of LLM-generated paraphrases affects fake news
detectors. Every news article is paraphrased by an LLM (Gemini or GPT) in three ways:

| Variant | File suffix | Instruction to the paraphraser |
|---|---|---|
| Positive | `_positive` | shift the sentiment to be as positive as possible |
| Negative | `_negative` | shift the sentiment to be more negative |
| Preserved | `_neutral` | keep the sentiment of the original text |

All variants keep the meaning, facts, length and syntactic complexity of the original. Detectors
are trained and tested on each version, and their predictions on the human-written text are
compared with their predictions on each paraphrase.

## Repository structure

```
dataset/
  {dataset}/human/{dataset}_human_{split}.csv
  {dataset}/{paraphraser}/{dataset}_{paraphraser}_{split}_{variant}.csv
code/
  load_data.py                      loads the train/test/valid splits of a dataset version
  deep_learning.py                  CNN and LSTM classifiers
  bert_classifier.py                BERT classifier
  gpt/                              GPT-2 fine-tuning and GPT few-shot classification
  llama/                            Llama fine-tuning and prediction
  zero_shot_classification_v2.py    zero-shot classification with instruction-tuned LLMs
  paraphraser/                      paraphrase generation (GPT, Gemini) and BERTScore
  analysis/                         sentiment, linguistic, semantic and entailment analyses,
                                    significance tests and effect sizes
  supporting_code/                  helpers
```

Datasets: `covid-19`, `election_2024`, `isot`. Paraphrasers: `gemini`, `gpt`.
Splits: `train`, `test`, `valid`. Variants: `positive`, `negative`, `neutral` (preserved sentiment).

### Data files

| Column | Meaning |
|---|---|
| `id` | row id of the article in the original dataset; links the human text and its paraphrases |
| `human` | original human-written article |
| `gemini` / `gpt` | paraphrase written by that LLM |
| `label` | gold label (0 = fake, 1 = real) |
| `title`, `subject`, `date` | ISOT metadata |
| `id_match` | (election_2024) how `id` was recovered: `split`, `repeat`, `other` or `unmatched` |

`election_2024` comes from the Hugging Face dataset
[`newsmediabias/fake_news_elections_labelled_data`](https://huggingface.co/datasets/newsmediabias/fake_news_elections_labelled_data)
(gated; accept its terms before use). Use of the source datasets is subject to their original licenses.

## Setup

Python 3.11+ with `pandas`, `numpy`, `scikit-learn`, `scipy`, `nltk`, `spacy` (`en_core_web_sm`),
`torch`, `transformers`, `tensorflow`/`keras` (CNN, LSTM), `datasets`, `vaderSentiment`,
`textstat`, `sentence-transformers`, `matplotlib` and `seaborn`.

API keys and tokens are read from environment variables; none are stored in the code:

| Variable | Used by |
|---|---|
| `openai_api_key` | GPT paraphrasing and few-shot classification |
| `gemini_api_key` | Gemini paraphrasing |
| `HF_TOKEN` | gated Hugging Face models (e.g. Llama) |

Scripts are run from the repository root and expect the data under `Dataset/` and write to
`results/`; rename `dataset/` to `Dataset/` (or adjust the paths in `code/load_data.py`) before running.
