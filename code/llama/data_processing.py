from datasets import Dataset, DatasetDict

def process_data(train_df,test_df,valid_df):

  # X_train = pd.DataFrame(X_train)
  # X_test = pd.DataFrame(X_test)
  # X_eval = pd.DataFrame(X_eval)

  # train_data = Dataset.from_pandas(X_train)
  # test_data = Dataset.from_pandas(X_test)
  # eval_data = Dataset.from_pandas(X_eval)

  # Converting pandas DataFrames into Hugging Face Dataset objects:
  dataset_train = Dataset.from_pandas(train_df.drop('label',axis=1))
  dataset_val = Dataset.from_pandas(valid_df.drop('label',axis=1))
  dataset_test = Dataset.from_pandas(test_df.drop('label',axis=1))
# Shuffle the training dataset
  dataset_train_shuffled = dataset_train.shuffle(seed=42)  # Using a seed for reproducibility

# Combine them into a single DatasetDict
  dataset = DatasetDict({
      'train': dataset_train_shuffled,
      'val': dataset_val,
      'test': dataset_test
  })
  
  print(dataset)

  return dataset


def label_convert(df):
  df['label']=df['label'].astype('category')
  df['target']=df['label'].cat.codes

  return df


def llama_preprocessing_function(examples,tokenizer,MAX_LEN):
    return tokenizer(examples['clean_text'], truncation=True, max_length=MAX_LEN)