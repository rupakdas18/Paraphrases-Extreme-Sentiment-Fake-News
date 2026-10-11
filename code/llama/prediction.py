

import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score
from huggingface_hub import login, hf_hub_download
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import variables
import supporting_code.save_results as save_results
 

def make_predictions(model,df_test,category_map,tokenizer):

  # Convert summaries to a list
  sentences = df_test.clean_text.tolist()

  # Define the batch size
  batch_size = 32  # You can adjust this based on your system's memory capacity

  # Initialize an empty list to store the model outputs
  all_outputs = []

  # Process the sentences in batches
  for i in range(0, len(sentences), batch_size):
      # Get the batch of sentences
      batch_sentences = sentences[i:i + batch_size]

      # Tokenize the batch
      inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)

      # Move tensors to the device where the model is (e.g., GPU or CPU)
      inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

      # Perform inference and store the logits
      with torch.no_grad():
          outputs = model(**inputs)
          all_outputs.append(outputs['logits'])


  final_outputs = torch.cat(all_outputs, dim=0)
  df_test['Llama_pred']=final_outputs.argmax(axis=1).cpu().numpy()
  df_test['Llama_pred']=df_test['Llama_pred'].apply(lambda l:category_map[l])

  return df_test


def weight_calculate(df):

  df.target.value_counts(normalize=True)

  class_weights=(1/df.target.value_counts(normalize=True).sort_index()).tolist()
  class_weights=torch.tensor(class_weights)
  class_weights=class_weights/class_weights.sum()

  return class_weights


def get_performance_metrics(df_test):
  y_test = df_test.label
  y_pred = df_test.Llama_pred

  print("Confusion Matrix:")
  print(confusion_matrix(y_test, y_pred))

  print("\nClassification Report:")
  print(classification_report(y_test, y_pred))

  print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))


  
  accuracy = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred, average='weighted')
  precision = precision_score(y_test, y_pred, average='weighted')
  recall = recall_score(y_test, y_pred, average='weighted')

  print("Accuracy Score:", accuracy)
  print(f"F1 Score: {f1:.3f}")
  print(f"Precision: {precision:.3f}")
  print(f"Recall: {recall:.3f}")

  df_test.to_csv('llama_test_pred.csv', index=False)
  df_test.to_excel(f"results/{variables.classifier}/{variables.dataset_name}_{variables.paraphraser}_{variables.sentiment}_Llama_results.xlsx")

  save_results.save_classification_results(accuracy, precision, recall, f1)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'balanced_accuracy' : balanced_accuracy_score(predictions, labels),'accuracy':accuracy_score(predictions,labels)}

