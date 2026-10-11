import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import load_data
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
try:
    from transformers import get_scheduler
except ImportError:
    from transformers.optimization import get_scheduler

from torch.optim import AdamW



from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm.auto import tqdm
import variables
import supporting_code.save_results as save_results

RESULTS_SUMMARY_PATH = "results/gpt_2/all_gpt2_results.csv"


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
    Date: str,
    output_csv_path: str = RESULTS_SUMMARY_PATH,
):
    """
    Append one experiment's overall metrics as a row in the summary CSV,
    so every run of this script adds a row instead of overwriting.
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
        "Date": Date,
    }])

    write_header = not os.path.exists(output_csv_path)
    new_result.to_csv(output_csv_path, mode="a", header=write_header, index=True)
    print(f"Summary metrics appended to: {output_csv_path}")


    
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        
       # self.texts = dataframe[variables.paraphraser].tolist()
        self.dataframe = dataframe
        self.texts = dataframe['clean_text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        # Pass the row's position, not its ``id`` label: ids can be strings,
        # which torch.tensor cannot hold.  The ids stay in dataframe.index.
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label),
            'index': torch.tensor(idx),
        }

    

# def evaluate(model, data_loader):
#     model.eval()
#     predictions, true_labels = [], []

#     with torch.no_grad():
#         for batch in data_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(**batch)
#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=-1)
#             predictions.extend(preds.cpu().numpy())
#             true_labels.extend(batch['labels'].cpu().numpy())

#     precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
#     acc = accuracy_score(true_labels, predictions)
#     conf_mat = confusion_matrix(true_labels, predictions)
#     class_report = classification_report(true_labels, predictions)
#     return acc, precision, recall, f1, conf_mat, class_report

def get_predictions(model, data_loader, dataframe):
    model.eval()
    predictions, true_labels, texts = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            # Dataset ``index`` holds row positions, so retrieve rows with iloc.
            positions = batch['index'].tolist()
            texts.extend(dataframe['clean_text'].iloc[positions].tolist())
            batch = {k: v.to(device) for k, v in batch.items() if k != 'index'}  # Pass only relevant data to the model
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    acc = accuracy_score(true_labels, predictions)
    conf_mat = confusion_matrix(true_labels, predictions)
    class_report = classification_report(true_labels, predictions)

    print(f'Accuracy: {acc}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}')
    print('Confusion Matrix:\n', conf_mat)
    print('Classification Report:\n', class_report)

    save_results.save_classification_results(acc, precision, recall, f1)

    save_metrics_to_csv(
        dataset=variables.dataset_name,
        paraphraser=variables.paraphraser,
        sentiment=variables.sentiment,
        selected_model=variables.classifier,
        shots="full-fine-tuning",
        accuracy=acc,
        f1=f1,
        precision=precision,
        recall=recall,
        Date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    return texts, predictions, true_labels




if __name__ == "__main__":




    variables.dataset_name = 'election_2024' # 'liar', 'kaggle', covid-19, election_2024

    if variables.dataset_name == 'liar':
        variables.num_classes = 6 # 2, 6
    else:
        variables.num_classes = 2

    variables.paraphraser = 'gpt' # 'gpt', 'pegasus', 'llama''
    language = 'English' # 'english', 'german', 'french',
    variables.classifier = 'gpt_2'
    variables.sentiment = 'neutral' # 'postive', 'negative',

    test_case = False



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    


    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load configuration and modify it to acknowledge the new padding token
    config = GPT2Config.from_pretrained('gpt2', pad_token_id=tokenizer.pad_token_id, num_labels=variables.num_classes)

    # Load the model with updated configuration
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', config=config)

    # Update model's tokenizer to include the padding token
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)


    
    
        

    print(variables.sentiment)
    full_df, train_df, test_df, valid_df = load_data.data_load(variables.dataset_name,variables.num_classes,variables.paraphraser,language,variables.classifier,variables.sentiment)

    # The source CSVs store their original row identifier in ``id``.  Pandas
    # otherwise creates a new 0-based index when reading each split.
    for split_name, split_df in (("train", train_df), ("test", test_df), ("valid", valid_df)):
        if 'id' not in split_df.columns:
            raise KeyError(f"The {split_name} dataset has no 'id' column to preserve as its index.")
        split_df.set_index('id', drop=False, inplace=True)

    #list_available_gpus()
    #train_df, test_df, valid_df,num_labels = load_liar2()
    #train_df, test_df, valid_df,num_labels = load_data()

    '''
    This part is just to use a small set of data. Remove that when the code is completed and working

    '''

    if test_case:
        train_df = train_df.head(100)
        test_df = test_df.head(20)
        valid_df = valid_df.head(20)
    
    print("Shape before removing NAN values")
    print(train_df.shape)
    print(test_df.shape)
    print(valid_df.shape)


    train_df = train_df.dropna(how='any')
    test_df = test_df.dropna(how='any')
    valid_df = valid_df.dropna(how='any')

    
    text_length = 512

    train_df['clean_text']=train_df[variables.paraphraser].apply(lambda x: load_data.clean_text(x,text_length))
    test_df['clean_text']=test_df[variables.paraphraser].apply(lambda x: load_data.clean_text(x,text_length))
    valid_df['clean_text']=valid_df[variables.paraphraser].apply(lambda x: load_data.clean_text(x,text_length))

    train_df = train_df[
    train_df['clean_text'].notna() &
    train_df['clean_text'].astype(str).str.strip().ne('')
]

    valid_df = valid_df[
    valid_df['clean_text'].notna() &
    valid_df['clean_text'].astype(str).str.strip().ne('')
]

    test_df = test_df[
    test_df['clean_text'].notna() &
    test_df['clean_text'].astype(str).str.strip().ne('')
]

    print("Shape after removing NAN values")

    print(train_df.shape)
    print(test_df.shape)
    print(valid_df.shape)



    print(train_df.head())

    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_df, tokenizer, max_length=512)
    valid_dataset = TextDataset(valid_df, tokenizer, max_length=512)
    test_dataset = TextDataset(test_df, tokenizer, max_length=512)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    print("Data loaders created: ", len(train_loader))


    # Setup optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 5
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


   # Training loop
    model.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc="Epoch {:1d}".format(epoch+1))
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items() if k != 'index'}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({'loss': loss.item()})


    # Evaluate the model
    # accuracy, precision, recall, f1_score, conf_matrix, class_rep = evaluate(model, test_loader)
    # print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1_score}')
    # print('Confusion Matrix:\n', conf_matrix)
    # print('Classification Report:\n', class_rep)

    texts, predicted_labels, actual_labels = get_predictions(model, test_loader, test_df)
    results_df = pd.DataFrame({
        'clean_text': texts,
        'Text': test_df[variables.paraphraser].tolist(),
        'Actual Label': actual_labels,
        'GPT2_pred': predicted_labels,
    }, index=test_df.index)

    # Save the original ISOT row ID as the first CSV column.
    results_df.to_csv(
        f"results/gpt_2/{variables.dataset_name}_{variables.paraphraser}_{variables.sentiment}_gpt2_results.csv",
        index=True,
        index_label='id',
    )
