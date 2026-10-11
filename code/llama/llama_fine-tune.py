import time

import warnings
from transformers import DataCollatorWithPadding
warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import torch.nn.functional as F
import nltk
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
import data_processing
import prediction
import train
import fine_tune_model
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import load_data
import print_dataset
import variables


if __name__ == "__main__":
    
  start = time.time()
  variables.dataset_name = 'election_2024' # 'liar', 'kaggle', covid-19, election_2024
  variables.num_classes = 2 # 2, 6
  variables.paraphraser = 'gpt' # 'no', 'bard', 'parrot','gpt', 'pegasus', 'llama''
  language = 'English' # 'english', 'german', 'french',
  variables.classifier = 'llama'
  variables.sentiment = 'positive' # '_positive', '_negative',

  test_case = False

  full_df, train_df, test_df, valid_df =  load_data.data_load(variables.dataset_name,variables.num_classes,variables.paraphraser,variables.language,variables.classifier,variables.sentiment)

  if test_case:
    train_df = train_df[0:100]
    test_df = test_df[0:20]
    valid_df = valid_df[0:20]

  text_length = 4500
  train_df['clean_text']=train_df[variables.paraphraser].apply(lambda x: load_data.clean_text(x,text_length))
  test_df['clean_text']=test_df[variables.paraphraser].apply(lambda x: load_data.clean_text(x,text_length))
  valid_df['clean_text']=valid_df[variables.paraphraser].apply(lambda x: load_data.clean_text(x,text_length))


  train_df = data_processing.label_convert(train_df)
  test_df = data_processing.label_convert(test_df)
  valid_df = data_processing.label_convert(valid_df)

  category_map = {code: category for code, category in enumerate(train_df['label'].cat.categories)}
  print(category_map)
  print(train_df.head())

  dataset = data_processing.process_data(train_df,test_df,valid_df)
  print(dataset)

  class_weights = prediction.weight_calculate(train_df)

  # # model_name = "meta-llama/Llama-2-7b-chat-hf"
  # login(token='your api code')
  model_name = "meta-llama/Meta-Llama-3-8B"
  model, tokenizer = train.define_model(model_name,variables.num_classes)

  test_df = prediction.make_predictions(model,test_df,category_map,tokenizer)
  prediction.get_performance_metrics(test_df)
  
  mid = time.time()
  print("Time required for classification without fine-tuning: ", (mid-start)/3600)

  MAX_LEN = 512
  # col_to_delete = ['id', 'text']
  col_to_delete = []

  # tokenized_datasets = dataset.map(data_processing.llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
  tokenized_datasets = dataset.map(
    data_processing.llama_preprocessing_function,
    fn_kwargs={"tokenizer": tokenizer, "MAX_LEN": MAX_LEN},
    batched=True,
    remove_columns=col_to_delete
)
  tokenized_datasets = tokenized_datasets.rename_column("target", "label")
  tokenized_datasets.set_format("torch")

  output_dir="trained_weigths_2"
  trainer = fine_tune_model.fine_tune(model, tokenized_datasets['train'], tokenized_datasets['val'], tokenizer,class_weights, 5,output_dir)
  train_result = trainer.train()

  prediction.make_predictions(model,test_df,category_map,tokenizer)
  prediction.get_performance_metrics(test_df)
  
  end = time.time()
  print("Time required for classification with fine-tuning: ", (end-start)/3600)

  import sound