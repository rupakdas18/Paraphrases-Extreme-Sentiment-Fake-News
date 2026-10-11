from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from transformers import TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer


import prediction
import train
import torch





def fine_tune(model, train_data, eval_data, tokenizer,class_weights, epoch,output_dir):

  collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

  peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

  training_arguments = TrainingArguments(
    output_dir=output_dir,                    # directory to save and repository id
    num_train_epochs=epoch,                       # number of training epochs
    per_device_train_batch_size=1,            # batch size per device during training
    gradient_accumulation_steps=8,            # number of steps before performing a backward/update pass
    gradient_checkpointing=True,              # use gradient checkpointing to save memory
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,                         # log every 10 steps
    learning_rate=2e-4,                       # learning rate, based on QLoRA paper
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
    max_steps=-1,
    warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
    # group_by_length=True,
    lr_scheduler_type="cosine",               # use cosine learning rate scheduler
    report_to="tensorboard",                  # report metrics to tensorboard
    eval_strategy="epoch"               # save checkpoint every epoch
)

  trainer = train.CustomTrainer(
      model=model,
      args=training_arguments,
      train_dataset=train_data,
      eval_dataset=eval_data,
      # tokenizer=tokenizer,
      data_collator=collate_fn,
      compute_metrics=prediction.compute_metrics,
      class_weights=class_weights
  )


#   training_args = TrainingArguments(
#     output_dir='fake_news_classification',
#     learning_rate=1e-4,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=epoch,
#     weight_decay=0.01,
#     eval_strategy='epoch',
#     save_strategy='epoch',
#     load_best_model_at_end=True
# )
  

#   trainer = train.CustomTrainer(
#     model = model,
#     args = training_args,
#     train_dataset = train_data,
#     eval_dataset = eval_data,
#     tokenizer = tokenizer,
#     data_collator = collate_fn,
#     compute_metrics = prediction.compute_metrics,
#     class_weights=class_weights,
#)

  return trainer