
import torch
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from trl import setup_chat_format
from accelerate import PartialState, Accelerator
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch.nn.functional as F



from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure label_weights is a tensor
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    # def compute_loss(self, model, inputs, num_items_in_batch=None):
    #     # Custom loss computation logic here
    #     outputs = model(**inputs)
    #     loss = outputs.loss
    #     return loss

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        # Extract labels and convert them to long type for cross_entropy
        labels = inputs.pop("labels").long()

        # Forward pass
        outputs = model(**inputs)

        # Extract logits assuming they are directly outputted by the model
        logits = outputs.get('logits')

        # Compute custom loss with class weights for imbalanced data handling
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss
    


def define_model(model_name,num_labels):


#   bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float32,
#     bnb_4bit_use_double_quant=True
# )

#   model = AutoModelForCausalLM.from_pretrained(
#       model_name,
#       torch_dtype=torch.float32,
#       quantization_config=bnb_config,
#       device_map = torch.cuda.set_device(desired_device)
#   )

  # model.config.use_cache = False
  # model.config.pretraining_tp = 1
  # tokenizer = AutoTokenizer.from_pretrained(model_name,
  #                                         trust_remote_code=True,
  #                                        )
  # tokenizer.pad_token = tokenizer.eos_token
  # tokenizer.padding_side = "right"
  # model, tokenizer = setup_chat_format(model, tokenizer)

  quantization_config = BitsAndBytesConfig(
      load_in_4bit = True, # enable 4-bit quantization
      bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
      bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
      bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
  )

  lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
  )

  model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels= num_labels
)

  model = prepare_model_for_kbit_training(model)
  model = get_peft_model(model, lora_config)

  tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

  tokenizer.pad_token_id = tokenizer.eos_token_id
  tokenizer.pad_token = tokenizer.eos_token
  model.config.pad_token_id = tokenizer.pad_token_id
  model.config.use_cache = False
  model.config.pretraining_tp = 1


  return model,tokenizer


