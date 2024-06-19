# coding=utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from numpy import mean


import transformers
import torch

import argparse

from torch import tensor

import logging
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain

from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split

from Model import E_code_Model
from No_expert_Model import No_expert_Model
from GPT_Neo_Model import GPT_Neo_Model
from No_expert_train_Dataset import No_expert_Train_Dataset
from No_expert_test_Dataset import No_expert_Test_Dataset
from train_Dataset import Train_Dataset
from GPT_Neo_train_Dataset import GPT_Neo_Train_Dataset
from No_expert_test_Dataset import No_expert_Test_Dataset
from test_Dataset import Test_Dataset
from GPT_Neo_test_Dataset import GPT_Neo_Test_Dataset
from Prediction_generation_code import main as prediction_generation_code_main

Logger = None

GPT_tokenizer = transformers.GPT2Tokenizer.from_pretrained("GPT_Tokenizer/", pad_token="[PAD]", cls_token="[CLS]")
Bert_tokenizer = BertTokenizer.from_pretrained("Bert_Tokenizer/", pad_token="[PAD]")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def parse_command_line_args():
    # ----------------------------------- Command line parsing module -----------#
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='0', type=str, required=False)
    parser.add_argument('--cuda', default=1, required=False, help='0:NO,1:Yes')
    parser.add_argument('--task', default=0, type=int, required=False, help='0: E-code, 1: No-expert group, 2: GPT-Neo')
    parser.add_argument('--GPT_arch', default="GPT-NEO-125M")
    parser.add_argument('--heads', default=48, type=int)
    parser.add_argument('--RELU', default=0, type=int, required=False, help='0: NO, 1: Yes')
    parser.add_argument('--log_path', default='Log/train_log.txt', type=str, required=False)
    parser.add_argument('--epochs', default=15, type=int, required=False, )
    parser.add_argument('--batch_size', default=1, type=int, required=False)
    parser.add_argument('--gradient_accumulation', default=32, type=int, required=False)
    parser.add_argument('--lr', default=2e-5, type=float, required=False)
    # parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up steps')
    parser.add_argument('--log_step', default=1000, type=int, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--Generated_models', default='Generated_models', type=str, required=False)
    parser.add_argument('--use_trained_local_models', default=0, type=int, required=False, help='0: NO, 1: Yes')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--num_workers', type=int, default=0)

    # --------------------------- Parse arguments ------------------#
    args = parser.parse_args()

    return args


def create_log_file(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # ----------------------- Formatting ---------------------#
    time_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # ------------------- Create a handler to write to log file ---------------#
    file_handler = logging.FileHandler(filename=args.log_path)
    file_handler.setFormatter(time_format)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # ------------------- Create a handler to log to the console -----------#
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(time_format)
    logger.addHandler(console)

    return logger


def E_code_padding(batch_data):
    titles = []
    problem_description = []
    input_description = []
    output_description = []
    io_example_note = []
    slow_code = []
    target_code = []

    # ------------------------------- Calculate the maximum input length in this batch --------------------------#
    for data in batch_data:
        titles.append(data[0])
        problem_description.append(data[1])
        input_description.append(data[2])
        output_description.append(data[3])
        io_example_note.append(data[4])
        slow_code.append(data[5])
        target_code.append(data[6])

    # ---------------------------------------------------------- Convert to tensor dictionary --------------------------------#
    title_tensors = Bert_tokenizer(titles, max_length=512, truncation=True, padding=True, return_tensors="pt")
    problem_description_tensors = Bert_tokenizer(problem_description, max_length=512, truncation=True, padding=True,
                                                 return_tensors="pt")
    input_description_tensors = Bert_tokenizer(input_description, max_length=512, truncation=True, padding=True,
                                               return_tensors="pt")
    output_description_tensors = Bert_tokenizer(output_description, max_length=512, truncation=True, padding=True,
                                                return_tensors="pt")
    io_example_note_tensors = Bert_tokenizer(io_example_note, max_length=512, truncation=True, padding=True,
                                             return_tensors="pt")
    slow_code_tensors = GPT_tokenizer(slow_code, max_length=768, truncation=True, padding=True, return_tensors="pt")
    target_code_tensors = GPT_tokenizer(target_code, max_length=766, truncation=True, padding=True, return_tensors="pt")

    # =================================== Add start and end tokens to target_code_tensors =====================================#
    batch_size = len(target_code_tensors["input_ids"])

    # ----------------------------- First part: input_ids -----------------------------#
    start_token_list = [[102]] * batch_size
    start_token = torch.tensor(start_token_list)

    end_token_list = [[50256]] * batch_size
    end_token = torch.tensor(end_token_list)

    target_code_tensors["input_ids"] = torch.cat((start_token, target_code_tensors["input_ids"], end_token), dim=1)

    # ----------------------------- Second part: attention_mask ------------------------------------#
    attention_token_list = [[1]] * batch_size
    attention_token = torch.tensor(attention_token_list)

    target_code_tensors["attention_mask"] = torch.cat(
        (attention_token, target_code_tensors["attention_mask"], attention_token), dim=1)

    return [title_tensors, problem_description_tensors, input_description_tensors, output_description_tensors,
            io_example_note_tensors, slow_code_tensors, target_code_tensors]


def No_expert_padding(batch_data):
    total_question_text = []
    slow_code = []
    target_code = []

    # ------------------------------- Calculate the maximum input length in this batch --------------------------#
    for data in batch_data:
        total_question_text.append(data[0])
        slow_code.append(data[1])
        target_code.append(data[2])

    # ---------------------------------------------------------- Convert to tensor dictionary --------------------------------#
    total_question_text_tensors = Bert_tokenizer(total_question_text, max_length=2048, truncation=True, padding=True,
                                                 return_tensors="pt")
    slow_code_tensors = GPT_tokenizer(slow_code, max_length=768, truncation=True, padding=True, return_tensors="pt")
    target_code_tensors = GPT_tokenizer(target_code, max_length=766, truncation=True, padding=True, return_tensors="pt")

    # =================================== Add start and end tokens to target_code_tensors =====================================#
    batch_size = len(target_code_tensors["input_ids"])

    # ----------------------------- First part: input_ids -----------------------------#
    start_token_list = [[102]] * batch_size
    start_token = torch.tensor(start_token_list)

    end_token_list = [[50256]] * batch_size
    end_token = torch.tensor(end_token_list)

    target_code_tensors["input_ids"] = torch.cat((start_token, target_code_tensors["input_ids"], end_token), dim=1)

    # ----------------------------- Second part: attention_mask ------------------------------------#
    attention_token_list = [[1]] * batch_size
    attention_token = torch.tensor(attention_token_list)

    target_code_tensors["attention_mask"] = torch.cat(
        (attention_token, target_code_tensors["attention_mask"], attention_token), dim=1)

    return [total_question_text_tensors, slow_code_tensors, target_code_tensors]


# ####################################### Training Function #######################################
def train(model, device, train_data, test_data, multiple_gpus, cmd_args, padding_fn):

    # ---------------- Convert train data to train data loader --------------------#
    train_loader = DataLoader(dataset=train_data,
                              batch_size=cmd_args.batch_size,
                              shuffle=True,
                              num_workers=cmd_args.num_workers,
                              collate_fn=padding_fn)

    # ---------------- Switch to train mode --------------------#
    model.train()

    # --------------- Calculate total number of optimization steps --------------------#
    total_steps = int(train_loader.__len__() * cmd_args.epochs / cmd_args.batch_size / cmd_args.gradient_accumulation)

    Logger.info(f'Total training steps: {total_steps}')

    optimizer = transformers.AdamW(model.parameters(), lr=cmd_args.lr, correct_bias=True)

    Logger.info('Starting training')

    accumulated_loss = 0

    for epoch in range(cmd_args.epochs):
        Logger.info(f'Training, epoch: {epoch+1}')

        out_of_memory_occurrences = 0
        train_loss_list = []

        for batch_idx, feature_list in enumerate(train_loader):
            for idx, tensor_dict in enumerate(feature_list):
                feature_list[idx]["input_ids"] = tensor_dict["input_ids"].to(device)
                feature_list[idx]["attention_mask"] = tensor_dict["attention_mask"].to(device)
                try:
                    feature_list[idx]["token_type_ids"] = tensor_dict["token_type_ids"].to(device)
                except:
                    pass

            try:
                model_output = model(feature_list)
                loss = model_output.loss

                train_loss_list.append(loss.item())

                if batch_idx % cmd_args.log_step == 0:
                    Logger.info(f'Train, epoch: {epoch + 1}, Batch: {batch_idx + 1}/{len(train_loader)}, Loss: {loss:.3f}')

                if multiple_gpus:
                    loss = loss.mean()

                if cmd_args.gradient_accumulation > 1:
                    loss = loss / cmd_args.gradient_accumulation

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), cmd_args.max_grad_norm)

                if (batch_idx + 1) % cmd_args.gradient_accumulation == 0:
                    accumulated_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    out_of_memory_occurrences += 1
                    Logger.info(f'WARNING: Out of memory, occurrences: {out_of_memory_occurrences}')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    Logger.info(str(e))
                    raise e

        if cmd_args.task == 0:
            model.TitleLayer.save_pretrained(f"{cmd_args.generated_models}/TitleLayer/")

        Logger.info(f'Train, epoch: {epoch + 1}, Loss: {mean(train_loss_list)}')

    Logger.info('End of training')


# ######################################## Prediction Function #######################################
def predict(model, device, test_data, multiple_gpus, cmd_args, padding_fn):
    Logger.info("Starting prediction")

    model.eval()

    test_loader = DataLoader(test_data,
                             batch_size=cmd_args.batch_size,
                             shuffle=True,
                             num_workers=cmd_args.num_workers,
                             collate_fn=padding_fn)

    with torch.no_grad():
        loss_list = []
        for batch_idx, feature_list in enumerate(test_loader):
            for idx, tensor_dict in enumerate(feature_list):
                feature_list[idx]["input_ids"] = tensor_dict["input_ids"].to(device)
                feature_list[idx]["attention_mask"] = tensor_dict["attention_mask"].to(device)
                try:
                    feature_list[idx]["token_type_ids"] = tensor_dict["token_type_ids"].to(device)
                except:
                    pass

            try:
                model_output = model(feature_list)
            except:
                continue
            loss = model_output.loss

            loss_list.append(loss.item())

        Logger.info(f'Evaluate test set loss: {mean(loss_list):.3f}')

        return mean(loss_list)


# ######################################## Main Function ########################################
def main():
    cmd_args = get_cmd_args()

    global Logger
    Logger = create_log_file(cmd_args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if not os.path.exists(cmd_args.log_path):
        os.mkdir(cmd_args.log_path)
    if not os.path.exists(cmd_args.generated_models):
        os.mkdir(cmd_args.generated_models)

    if cmd_args.task == 0:
        model = ECodeModel(cmd_args)
    elif cmd_args.task == 1:
        model = NoExpertModel(cmd_args)
    elif cmd_args.task == 2:
        model = GPTNeoModel(cmd_args)

    model.to(device)

    multiple_gpus = False

    total_params = sum(p.numel() for p in model.parameters())

    Logger.info(f'Total model parameters: {total_params}')

    Logger.info("Loading training data")

    if cmd_args.task == 0:
        train_data = TrainDataset()
        train_function(model, device, train_data, train_data, multiple_gpus, cmd_args, ECodePadding)
    elif cmd_args.task == 1:
        train_data = NoExpertTrainDataset()
        test_data = NoExpertTestDataset()
        train_function(model, device, train_data, test_data, multiple_gpus, cmd_args, NoExpertPadding)
    elif cmd_args.task == 2:
        train_data = GPTNeoTrainDataset()
        test_data = GPTNeoTestDataset()
        train_function(model, device, train_data, test_data, multiple_gpus, cmd_args, GPTPadding)


if __name__ == '__main__':
    main()
