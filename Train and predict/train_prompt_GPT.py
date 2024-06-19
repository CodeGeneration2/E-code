# coding = UTF-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from numpy import mean
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from Dataset_prompt_GPT import MyDataset

# ######################################################################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_path = 'codegen-350M-mono_GPT'
# model_path = r'F:\PythonPureData\LargeLanguageModelParams\GPT-NEO 125M'

trained_name = f'./retrained_{model_path}_def_prompt'

# ======================================================== Load training data 2048 =============================================#
training_data = MyDataset(
    dataset_path="../../GEC/train",
    max_token_count=2048,
    model_path=model_path,
    train_or_predict="train"
)

# ############################################################ Main ####################################################
# ------------------------------------ Tokenization Vocabulary ------------------------------------#
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ------------------------------------ Vocabulary ------------------------------------#
if "codegen" in model_path or "PolyCoder" in model_path or "GPT-NEO" in model_path:
    tokenizer.pad_token = tokenizer.eos_token
elif "incoder" in model_path:
    tokenizer.pad_token = "<pad>"

# ---------------------------------------- Model ---------------------------------------#
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from transformers import Trainer, TrainingArguments
# ############################################################## Training ##################################################
training_args = TrainingArguments(
    output_dir=trained_name,          # output directory
    save_strategy="epoch",                 # save model method
    evaluation_strategy="no",            # prediction method
    num_train_epochs=15,                   # total epochs
    per_device_train_batch_size=1,        # training batch_size
    gradient_accumulation_steps=32,       # gradient accumulation
    per_device_eval_batch_size=1,         # prediction batch_size
    eval_accumulation_steps=1,            # number of evaluation steps to accumulate output tensors before moving them to the CPU
    warmup_steps=500,                     # warm up step count
    weight_decay=0.01,                    # strength of weight decay
    logging_dir='./logs',                  # log directory
    logging_strategy="epoch",             # log saving strategy
    logging_first_step=True,              # save log for the first step
    save_total_limit=1,                     # at most save 2 models
    overwrite_output_dir=True,              # overwrite
    dataloader_drop_last=True,              # drop the last
    dataloader_pin_memory=False,            # whether you want to pin memory in the dataloader. Defaults to True
    dataloader_num_workers=0,               # data loading
    prediction_loss_only=True,              # evaluate only the loss

    fp16=True,
)

# =============================================================================================================
"""Required. Otherwise, it will throw an error: ValueError: Expected input batch_size (978) to match target batch_size (404)"""
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,                           # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=training_data,                # training dataset
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
print('\033[0:34m===========================================Training finished!!!===================================\033[m')
print('\033[0:34m===========================================Training finished!!!===================================\033[m')
print('\033[0:34m===========================================Training finished!!!===================================\033[m')

# trainer.evaluate()
# print('\033[0:34m===========================================Prediction finished!!!===================================\033[m')

# trainer.save_model(f'model_params_{model_path}_prompt')
