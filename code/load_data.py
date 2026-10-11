#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import os
import matplotlib as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import spacy
#import torch.nn.functional as F
import nltk
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from datasets import load_dataset
from sklearn.model_selection import train_test_split



def remove_isot_dateline(text):
    """
    Remove ISOT-style news portal datelines and agency markers from an article:
    - the leading dateline, e.g. "WASHINGTON (Reuters) - ", "BEIJING (Reuters) - ",
      "ERBIL, Iraq (Reuters) - ", "WASHINGTON - ", "WASHINGTON Reuters - ";
    - any other standalone all-caps proper noun appearing later in the body,
      e.g. "RICHMOND", "LONDON", "MADRID";
    - the word "Reuters" wherever else it occurs, with its wrapping parentheses if present.
    """
    if not isinstance(text, str):
        return text

    # Leading dateline
    text = re.sub(
        r'^\s*[A-Z]{2,}(?:\s+[A-Z]{2,})*(?:,\s*[A-Za-z\.]+)?(?:\s*\(?(?:Reuters|AP|AFP)\)?)?\s*-\s*',
        '',
        text,
    )

    # "Reuters" anywhere else in the text, with its wrapping parentheses if present
    text = re.sub(r'\(?\bReuters\b\)?', '', text)

    # Any remaining standalone all-caps word (2+ letters), e.g. mid-article place names
    text = re.sub(r'\b[A-Z]{2,}\b', '', text)

    # Tidy up whitespace/punctuation left behind by the removals above
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.;:])', r'\1', text)
    text = text.strip(' -,')

    return text


def data_load(dataset_name,class_number,paraphraser,language,classifier,suffix):


    if dataset_name == 'election_2024' and paraphraser == 'human':
        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset("newsmediabias/fake_news_elections_labelled_data")
        print(ds)
        print(type(ds))
        df = pd.DataFrame(ds['train'])
        df = df.rename(columns={"text": "human"})
        # # First split: train (70%) and temp (30%)
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        # Second split: test (20%) and valid (10%) from temp (which is 30%)
        # test_size=2/3 means test=20%, valid=10% of the original dataset
        test_df, valid_df = train_test_split(temp_df, test_size=1/3, random_state=42)

    elif dataset_name == 'election_2024' and paraphraser != 'human':
        # df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}.csv")
        # train_df, temp_llldf = train_test_split(df, test_size=0.3, random_state=42)
        # test_df, valid_df = train_test_split(temp_df, test_size=1/3, random_state=42)
        train_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_train_{suffix}.csv")
        test_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_test_{suffix}.csv")
        valid_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_valid_{suffix}.csv")

    elif dataset_name == 'isot' and paraphraser =='human':

        # if suffix == '_positive_neutralized' or suffix == '_negative_neutralized':
            train_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_train.csv")
            test_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_test.csv")
            valid_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_valid.csv")
        # else: 
        #     fake_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_fake_{suffix}.csv")
        #     true_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_true_{suffix}.csv")
        #     fake_df['label'] = 0
        #     true_df['label'] = 1

        #     all_df = pd.concat([fake_df, true_df], axis=0)
        #     all_df = all_df.sample(frac=1, random_state=42)

        #     train_df, temp_df = train_test_split(all_df, test_size=0.3, random_state=42)
        #     test_df, valid_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    else:
        
        if paraphraser == 'human':
            train_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_train.csv")
            test_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_test.csv", encoding='latin1')
            valid_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_valid.csv")
        else:
            train_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_train_{suffix}.csv")
            test_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_test_{suffix}.csv", encoding='latin1')
            valid_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_valid_{suffix}.csv")

    print("Shape before removing NAN values")
    print(train_df.shape)
    print(test_df.shape)
    print(valid_df.shape)


    train_df = train_df.dropna(how='any')
    test_df = test_df.dropna(how='any')
    valid_df = valid_df.dropna(how='any')

    print("Shape after removing NAN values")

    print(train_df.shape)
    print(test_df.shape)
    print(valid_df.shape)

    merged_df = pd.concat([train_df, test_df, valid_df], axis =0 )

    print("Merged dataset before any processing")
    print(merged_df.head(10))

    if dataset_name == 'isot':
        text_column = 'human' if paraphraser == 'human' else paraphraser
        train_df[text_column] = train_df[text_column].apply(remove_isot_dateline)
        test_df[text_column] = test_df[text_column].apply(remove_isot_dateline)
        valid_df[text_column] = valid_df[text_column].apply(remove_isot_dateline)
        merged_df[text_column] = merged_df[text_column].apply(remove_isot_dateline)

    if dataset_name == 'kaggle':
        
        merged_df.reset_index(inplace = True)
        # drop the index column
        merged_df.drop(["index"], axis = 1, inplace = True)
        # Check the number of columns containing "null"

        
        merged_df['label'] = merged_df['label'].astype(int)
        train_df['label'] = train_df['label'].astype(int)
        test_df['label'] = test_df['label'].astype(int)
        valid_df['label'] = valid_df['label'].astype(int)

        label_mapping = {
        0: 0,
        1: 1,
        }

    elif dataset_name == 'election_2024':

        if paraphraser == 'human':
            label_mapping = {
            'FAKE': 0,
            'REAL': 1,
            }
        else:
            label_mapping = {
            0: 0,
            1: 1,
            }

      
    
    ################################################################################################################################
    elif dataset_name == 'TALLIP':

        headers = ['domain', 'topic', 'text', 'label']
        train_df.columns = headers
        valid_df.columns = headers
        test_df.columns = headers

        train_df['label'] = train_df['label'].map(label_mapping)
        test_df['label'] = test_df['label'].map(label_mapping)
        valid_df['label'] = valid_df['label'].map(label_mapping)

    elif dataset_name == 'covid-19' or dataset_name == 'isot':
         label_mapping = {
        0: 0,
        1: 1,
        }

    
    elif dataset_name == 'liar_2' or dataset_name == 'liar_6'or dataset_name == 'liar':

        if class_number == 6:
                    
            # if paraphraser == 'human':
            #         label_mapping = {
            #         'pants-fire': 0,
            #         'false': 1,
            #         'barely-true': 2,
            #         'half-true': 3,
            #         'mostly-true': 4,
            #         'true': 5}
            # else:
                label_mapping = {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 4,
                5: 5
                    }
                    
        elif class_number == 2:
             
            label_mapping = {
        0: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1
        }
            


        
    if (classifier != "gpt_n_shot") and (classifier != "zero-shot") and (classifier != "few-shot"):
        train_df['label'] = train_df['label'].map(label_mapping)
        test_df['label'] = test_df['label'].map(label_mapping)
        valid_df['label'] = valid_df['label'].map(label_mapping)
        merged_df['label'] = merged_df['label'].map(label_mapping)
                    
    label_counts = merged_df['label'].value_counts()
    print(label_counts)


    
              
    return merged_df, train_df, test_df, valid_df
    
    

# def clean_text(text,word_count):
#     """Process text function.
#     Input:
#         tweet: a string containing a tweet
#     Output:
#         tweets_clean: a list of words containing the processed tweet
#     """

#     try:
#         lemmatizer = WordNetLemmatizer()
#         stopwords_english = stopwords.words('english')
#         text= re.sub('\[[^]]*\]', '', text)
#         # remove stock market tickers like $GE
#         text = re.sub(r'\$\w*', '', text)
#         #removal of html tags
#         review =re.sub(r'<.*?>',' ',text)
#         # remove old style retweet text "RT"
#         text = re.sub(r'^RT[\s]+', '', text)
#         # remove hyperlinks
#         text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
#         # remove hashtags
#         # only removing the hash # sign from the word
#         text = re.sub(r'#', '', text)
#         text = re.sub("["
#                             u"\U0001F600-\U0001F64F"  # removal of emoticons
#                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                             u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                             u"\U00002702-\U000027B0"
#                             u"\U000024C2-\U0001F251"
#                             "]+",' ',text)
#         text = re.sub('[^a-zA-Z]',' ',text)
#         text = text.lower()
#         text_tokens =word_tokenize(text)

#         text_clean = []
#         for word in  text_tokens:
#             if (word not in stopwords_english and  # remove stopwords
#                     word not in string.punctuation):  # remove punctuation
#                 lem_word =lemmatizer.lemmatize(word)  # lemmitiging word
#                 text_clean.append(lem_word)
#         text_mod=[i for i in text_clean if len(i)>2]
#         text_clean=' '.join(text_mod)

#         words = text_clean.split()
#         first_n_words = words[:word_count]
#         trunced_clean_text = ' '.join(first_n_words)

#         return  trunced_clean_text

#     except:
#         pass
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')


def clean_text(text,word_count):
    """Process text function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    """

    lemmatizer = WordNetLemmatizer()
    stopwords_english = stopwords.words('english')
    text= re.sub('\[[^]]*\]', '', text)
    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    #removal of html tags
    review =re.sub(r'<.*?>',' ',text)
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    text = re.sub("["
                        u"\U0001F600-\U0001F64F"  # removal of emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+",' ',text)
    text = re.sub('[^a-zA-Z]',' ',text)
    text = text.lower()
    text_tokens =word_tokenize(text)

    text_clean = []
    for word in  text_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            lem_word =lemmatizer.lemmatize(word)  # lemmitiging word
            text_clean.append(lem_word)
    text_mod=[i for i in text_clean if len(i)>2]
    text_clean=' '.join(text_mod)

    words = text_clean.split()
    first_n_words = words[:word_count]
    trunced_clean_text = ' '.join(first_n_words)

    return  trunced_clean_text


nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def remove_proper_nouns(text, max_len=None):
    """
    Remove all proper nouns (PROPN) from text using POS tagging.

    Args:
        text (str): input news article
        max_len (int, optional): truncate to max_len tokens AFTER cleaning

    Returns:
        str: cleaned text
    """
    if not isinstance(text, str):
        return ""

    doc = nlp(text)

    tokens = [
        token.text
        for token in doc
        if token.pos_ != "PROPN"
    ]

    if max_len is not None:
        tokens = tokens[:max_len]

    cleaned = " ".join(tokens)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned





    
