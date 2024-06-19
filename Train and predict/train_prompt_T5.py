# coding = UTF-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from numpy import mean
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq

from Dataset_prompt_T5 import MyDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_path = 'codet5-small'

# ======================== Import training data 1792 =============================================#
training_data = MyDataset(
    dataset_path="../../GEC/train",
    max_token_count=2048,
    model_path=model_path,
    training_or_predicting="training"
)

# ================================ Tokenization Dictionary ================================#
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ================================ Model ================================#
model = T5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from transformers import Trainer, TrainingArguments

# ================================= Training =================================#
training_args = Seq2SeqTrainingArguments(
    output_dir=f'./trained_model_{model_path}_prompt',      # output directory
    save_strategy="epoch",                                  # saving model method
    evaluation_strategy="no",                               # evaluation method
    num_train_epochs=15,                                    # total epochs
    per_device_train_batch_size=1,                          # batch size for training
    gradient_accumulation_steps=32,                         # gradient accumulation
    per_device_eval_batch_size=1,                           # batch size for evaluation
    eval_accumulation_steps=1,                              # number of evaluation steps before results are accumulated to CPU
    warmup_steps=500,                                       # warm up steps
    weight_decay=0.01,                                      # strength of weight decay
    logging_dir='./logs',                                   # logging directory
    logging_strategy="epoch",                               # logging strategy
    logging_first_step=True,                                # log the first step
    save_total_limit=1,                                     # at most save 2 models
    overwrite_output_dir=True,                              # overwrite
    dataloader_drop_last=True,                              # discard the last
    dataloader_pin_memory=False,                            # whether to pin memory in the data loader. Defaults to True
    dataloader_num_workers=0,                               # data loading
    prediction_loss_only=True,                              # only evaluate loss

    fp16=True,

    # deepspeed=command_line_args.deepspeed,
)

# =========================================================================================== #
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

trainer = Seq2SeqTrainer(
    model=model,                                            # the instantiated 🤗 Transformers model to be trained
    args=training_args,                                     # training arguments, defined above
    train_dataset=training_data,                            # training dataset
    data_collator=data_collator,
    tokenizer=tokenizer,
)


trainer.train()
print('\033[0:34m=========================================== Training Finished ===================================\033[m')
print('\033[0:34m=========================================== Training Finished ===================================\033[m')
print('\033[0:34m=========================================== Training Finished ===================================\033[m')
