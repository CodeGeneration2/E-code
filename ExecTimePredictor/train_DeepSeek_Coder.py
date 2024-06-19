# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from peft import PeftConfig, PeftModel


import pprint
import json

import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch.multiprocessing

import warnings
warnings.filterwarnings("ignore")  # Ignore all types of warning messages

import os
os.environ["WANDB_DISABLED"] = "true"


def main(args):
    # -------------------------------------
    args_dict = vars(args)
    print(pprint.pformat(args_dict))
    os.makedirs(args.save_dir, exist_ok=True)
    # Save args dictionary
    json.dump(args_dict, open(os.path.join(args.save_dir, "args.json"), 'w'))

    total_dataset_preprocessed = get_dataset(args)

    # Load and train model; save model checkpoints
    lora_model = get_model(args)
    run_training(args, total_dataset_preprocessed, lora_model)


def get_dataset(args):

    # Load dataset
    total_data_table = pd.read_csv(args.train_path)
    total_data_table = convert_decimal_to_integer(total_data_table, 'Time')
    total_data_table['Time'] = total_data_table['Time'].astype(str)

    # ------------------------------------------------
    if args.debug:
        total_data_table = total_data_table[:1500]

    # Convert pandas DataFrame to datasets.Dataset
    train_data = Dataset.from_pandas(total_data_table)

    # Combine datasets.Dataset objects into DatasetDict
    total_dataset_DatasetDict = DatasetDict({
        "train": train_data,
        # "valid": dataset_valid
    })

    total_dataset_preprocessed = total_dataset_DatasetDict.map(
        preprocess_function,
        batched=True,
        remove_columns=total_dataset_DatasetDict["train"].column_names  # All these columns will be deleted after the mapping operation.
    )

    # ------------------------------------------------
    if args.debug:
        # Suppose you want to see the first 3 samples
        for i in range(2):
            sample = total_dataset_preprocessed['train'][i]
            print(f"Sample {i}:", sample)

    return total_dataset_preprocessed


def convert_decimal_to_integer(df, column_name):
    # Check if column name is in DataFrame
    if column_name in df.columns:
        # Multiply specified column values by 100 and convert to integer
        df[column_name] = df[column_name].apply(lambda x: int(x * 100))
        return df
    else:
        print(f"Column '{column_name}' not found in DataFrame, column name error.")
        return df


def preprocess_function(example):
    feature_dict = tokenizer(
        example['Prompt_input'],
        truncation=True,
        max_length=384,
        padding='max_length',
        return_length=True,
        return_tensors="pt"
    )

    label_dict = tokenizer(
        example["Time"],
        truncation=True,
        max_length=6,
        padding='max_length',
        return_length=True,
        return_tensors="pt"
    )

    # print(example["Time"])

    negative_100 = [[-100 for _ in row] for row in feature_dict['attention_mask']]
    negative_100_label_tensor = [[-100 if value == 100001 else value for value in row] for row in label_dict['input_ids']]

    feature_dict['input_ids'] = np.concatenate((feature_dict['input_ids'], label_dict['input_ids']), axis=1)
    feature_dict['attention_mask'] = np.concatenate((feature_dict['attention_mask'], label_dict['attention_mask']), axis=1)

    feature_dict['labels'] = np.concatenate((negative_100, negative_100_label_tensor), axis=1)

    assert len(feature_dict['input_ids'][0]) == len(feature_dict['labels'][0]) == 390

    return feature_dict



def get_model(args):

    """ Define specific LoRA parameters: large, (8,32) (256,128) ()"""
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",  # Sequential to sequential language modeling (TaskType.SEQ_2_SEQ_LM). Classification (TaskType.SEQ_CLS).
        inference_mode=False,  # Whether you are using the model for inference
        r=args.lora_r,  # Dimension (rank) of update matrix. Lower rank leads to smaller update matrices and fewer trainable parameters. 64,256
        lora_alpha=args.lora_alpha,  # LoRA scale factor. Scaling factor for low-rank matrix 128
        lora_dropout=0.01,  # Dropout probability for LoRA layer. Dropout rate
        use_rslora=True,  # Use rank stable LoRA, set adapter scaling factor to lora_alpha/math.sqrt(r), it works better. Default value lora_alpha/r.
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Modules to apply LoRA update matrices to. If not specified, modules will be chosen based on the model architecture.
    )


    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )


    model_path = args.model_path if args.model_path is not None else '{}'.format(args.model)
    print("Loading model from {}...".format(model_path))

    # ---------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 quantization_config=q_config,
                                                 torch_dtype=torch.bfloat16,
                                                 pad_token_id=tokenizer.pad_token_id,
                                                 trust_remote_code=True
                                                 )

    if args.use_lora_model:
        for i in args.lora_model_path:
            model = PeftModel.from_pretrained(model, i)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ---------------------------------------------------------------------
    lora_model = get_peft_model(model, peft_config)
    lora_model.print_trainable_parameters()  # Output the proportion of trainable parameters

    print(f'Finished loading model {args.model}, lora: {args.use_lora_model}')

    return lora_model



def run_training(args, total_dataset_preprocessed, lora_model):

    print(f"Starting main loop...")

    # =================================================================
    training_args = TrainingArguments(
        output_dir=args.save_dir,  # Save
        overwrite_output_dir=True,  # Do not overwrite content in output directory

        do_train=True,
        do_eval=False,
        do_predict=False,

        save_strategy='epoch',  # Strategy
        evaluation_strategy='no',  # Strategy
        # eval_steps=341,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.01,
        max_grad_norm=0.3,  # Maximum gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # Warmup ratio according to QLoRA paper
        # lr_scheduler_type='linear',

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_strategy='epoch',

        save_total_limit=args.save_total_limit,
        seed=args.seed,

        dataloader_drop_last=True,
        dataloader_num_workers=0,
        local_rank=args.local_rank,

        deepspeed=args.deepspeed,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        group_by_length=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=total_dataset_preprocessed['train'],

        data_collator=data_collator,
        tokenizer=tokenizer,

    )

    trainer.train()




# ######################################################################################################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training a model for code generation")

    parser.add_argument('--model', default="DeepSeek_coder_7b_instruct_v1_5", type=str,
                        help='type of transformers model as model backbone')
    parser.add_argument('--model_path', default=r'DeepSeek_coder_7b_instruct_v1_5', type=str,
                        help='path to model backbone pretrained weights')
    parser.add_argument('--use_lora_model', default=False, type=bool,
                        help='')
    parser.add_argument('--lora_model_path', default=['Model_DeepSeek_Coder_Pre_time_RandDel/checkpoint-9044',
                                                   'Model_DeepSeek_Coder_Pre_time_Loop&Rec/checkpoint-9306',
                                                   'Model_DeepSeek_Coder_Pre_time_Ori/checkpoint-8188',
                                                   'Model_DeepSeek_Coder_Pre_time_Uni/checkpoint-7906',], type=str,
                        help='')
    parser.add_argument('--save_dir', default='Model_DeepSeek_Coder_Pre_time_RandDel', type=str,
                        help='path to save trained model checkpoints')


    # Dataloading
    parser.add_argument('--train_path', default="CodeExecTimeDB-RandDel/train.csv", type=str,
                        help='training data path')

    # lora,（8,32）(32,32)(128,64)（256,128）
    parser.add_argument('--lora_r', default=256, type=int, help='')
    parser.add_argument('--lora_alpha', default=128, type=int, help='')

    # Training
    parser.add_argument('--epochs', default=2, type=int, help='total number of training epochs')
    parser.add_argument('--lr', default=2e-4, type=float, help='training learning rate')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size per GPU')
    parser.add_argument('--grad_acc_steps', default=8, type=int,
                        help='number of training steps before each gradient update')

    # Acceleration
    parser.add_argument('--deepspeed', default=None, type=str,
                        help='path to deepspeed configuration file; set None if not using deepspeed')
    parser.add_argument('--fp16', default=False, action='store_true',
                        help='set 16-bit training to reduce memory usage')
    parser.add_argument('--bf16', default=True, action='store_true',
                        help='set 16-bit training to reduce memory usage')

    parser.add_argument('--local_rank', default=-1, type=int)

    parser.add_argument('--debug', default=False, action='store_true',
                        help='set to turn on debug mode i.e. using dummy small data split and only 1 data worker')

    parser.add_argument('--seed', default=42, type=int)

    # Logging
    parser.add_argument('--save_total_limit', default=6, type=int,
                        help='total number of checkpoints to keep; 1 keeps only the latest checkpoint')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    main(args)
