# coding=utf-8

import os
from numpy import mean

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import json
import random
import numpy as np
import argparse

from torch import tensor
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain

from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from Model import E_code_Model
from No_expert_Model import No_expert_Model
from GPT_Neo_Model import GPT_Neo_Model
from train_Dataset import Train_Dataset
from GPT_Neo_train_Dataset import GPT_Neo_Train_Dataset
from test_Dataset import Test_Dataset
from GPT_Neo_test_Dataset import GPT_Neo_Test_Dataset

from No_expert_train_Dataset import No_expert_Train_Dataset

from No_expert_test_Dataset import No_expert_Test_Dataset


from CodeBleu import _bleu
from sacrebleu.metrics import BLEU, CHRF, TER
from IO_testing_of_generated_code import main as IO_testing_of_generated_code_main



projection_log = None

GPT_tokenizer = GPT2Tokenizer.from_pretrained("GPT_Tokenizer/", pad_token="[PAD]", cls_token="[CLS]")
Bert_tokenizer = BertTokenizer.from_pretrained("Bert_Tokenizer/", pad_token="[PAD]")


# ##################################################################### Command line arguments ######################################
def parse_command_line_args():

    # ----------------------------------- Command line parser -----------#
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='0', type=str, required=False)
    parser.add_argument('--task', default=0, type=int, required=False, help='0: E-code, 1: No-expert group, 2: GPT-Neo')
    parser.add_argument('--GPT_arch', default="GPT-NEO-125M")
    parser.add_argument('--heads', default=48, type=int)
    parser.add_argument('--RELU', default=0, type=int, required=False, help='0: NO, 1: Yes')
    parser.add_argument('--Maximum_length_pattern_of_generated_code', default=0, type=int, required=False)
    parser.add_argument('--Maximum_length_of_generated_code', default=748, type=int, required=False)
    parser.add_argument('--topk', default=3, type=int, required=False)
    parser.add_argument('--topp', default=0.7, type=float, required=False)
    parser.add_argument('--Temperature', default=0.25, type=float, required=False)
    parser.add_argument('--Train_set_interval', default=500, type=int, required=False)
    parser.add_argument('--Test_set_interval', default=1, type=int, required=False)
    parser.add_argument('--cuda', default=1, required=False)
    parser.add_argument('--log_path', default='Log/Projection_log.txt', type=str, required=False)
    parser.add_argument('--batch_size', default=1, type=int, required=False)
    parser.add_argument('--log_step', default=100, type=int, required=False)
    parser.add_argument('--Generated_models', default='Generated_models', type=str, required=False)
    parser.add_argument('--Whether_to_use_trained_local_models', default=1, type=int, required=False, help='0: NO, 1: Yes')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--num_workers', type=int, default=5)

    # --------------------------- Parse arguments ------------------#
    args = parser.parse_args()

    return args


# ##################################################################### Logging ######################################
def create_log_file(args):

    prediction_log = logging.getLogger(__name__)
    prediction_log.setLevel(logging.INFO)

    # ----------------------- Format ---------------------#
    time_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # ------------------- Create a handler for writing to log files ---------------#
    file_writer = logging.FileHandler(filename=args.log_path)
    file_writer.setFormatter(time_format)
    file_writer.setLevel(logging.INFO)
    prediction_log.addHandler(file_writer)

    # ------------------- Create a handler for outputting logs to console -----------#
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(time_format)
    prediction_log.addHandler(console)

    return prediction_log


def filter_with_TopK_and_Nucleus_method(predicted_token_probs, TopK=0, TopP=0.0, negative_infinity=-float('Inf')):

    # ----------------- Now, set batch as 1 --- This can be updated, but the code will be less clear -------------#
    assert predicted_token_probs.dim() == 1

    # -------------------------------------- TopK Method -------------------------------#
    TopK = min(TopK, predicted_token_probs.size(-1))
    if TopK > 0:
        topk_values = torch.topk(predicted_token_probs, TopK)
        last_token_prob_in_topk = topk_values[0][..., -1, None]
        removed_indices_bool_mask = predicted_token_probs < last_token_prob_in_topk
        predicted_token_probs[removed_indices_bool_mask] = negative_infinity

    # -------------------------------------- TopP (Nucleus) Method -------------------------------#
    if TopP > 0.0:
        sorted_probs, sorted_indices = torch.sort(predicted_token_probs, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_probs, dim=-1), dim=-1)
        removed_indices_post_sort = cumulative_probs > TopP
        removed_indices_post_sort[..., 1:] = removed_indices_post_sort[..., :-1].clone()
        removed_indices_post_sort[..., 0] = 0
        removed_id_mask = sorted_indices[removed_indices_post_sort]
        predicted_token_probs[removed_id_mask] = negative_infinity

    return predicted_token_probs


def E_code_padding(batch_data):

    title_tensor_dict = Bert_tokenizer(batch_data[0][0], max_length=512, truncation=True, padding=True, return_tensors="pt")
    body_tensor_dict = Bert_tokenizer(batch_data[0][1], max_length=512, truncation=True, padding=True, return_tensors="pt")
    input_desc_tensor_dict = Bert_tokenizer(batch_data[0][2], max_length=512, truncation=True, padding=True, return_tensors="pt")
    output_desc_tensor_dict = Bert_tokenizer(batch_data[0][3], max_length=512, truncation=True, padding=True, return_tensors="pt")
    io_example_and_note_tensor_dict = Bert_tokenizer(batch_data[0][4], max_length=512, truncation=True, padding=True, return_tensors="pt")
    slow_code_tensor_dict = GPT_tokenizer(batch_data[0][5], max_length=768, truncation=True, padding=True, return_tensors="pt")

    return [title_tensor_dict, body_tensor_dict, input_desc_tensor_dict, output_desc_tensor_dict, io_example_and_note_tensor_dict, slow_code_tensor_dict], batch_data[0][6], batch_data[0][7], batch_data[0][8]


def No_expert_padding(batch_data):

    total_question_tensor_dict = Bert_tokenizer(batch_data[0][0], max_length=2048, truncation=True, padding=True, return_tensors="pt")
    slow_code_tensor_dict = GPT_tokenizer(batch_data[0][1], max_length=768, truncation=True, padding=True, return_tensors="pt")

    return [total_question_tensor_dict, slow_code_tensor_dict], batch_data[0][2], batch_data[0][3], batch_data[0][4]


def GPT_NEO_padding(batch_data):

    total_question_tensor_dict = GPT_tokenizer(batch_data[0][0], max_length=2048, truncation=True, padding=True, return_tensors="pt")

    return [total_question_tensor_dict], batch_data[0][1], batch_data[0][2], batch_data[0][3]




# ########################################## Training Function #####################################
def prediction_generation_function(model, device, dataset, multi_GPU, command_line_args, train_or_test, epoch, padding_function, Task):
    # ---------------------------- Convert training data into training data loader --------------------#
    data_loader = DataLoader(dataset=dataset,  # Dataset being used
                             batch_size=command_line_args.batch_size,  # Batch sample size
                             shuffle=False,  # Do not shuffle data before each iteration
                             num_workers=command_line_args.num_workers,  # Using one process
                             collate_fn=padding_function
                             )

    # -------- Switch to prediction mode: do not compute gradients --------#
    model.eval()

    projection_log.info('##################### starting Prediction_generation_code ###########')

    # ########################################## Start ########################################## #
    # =========================== No Gradient ======================#
    with torch.no_grad():
        # ---------------------------------- Compute Bleu ----------------------------------------#
        BLEU_score_list = []
        CodeBLEU_score_list = []
        compile_rate_list = []
        IO_test_pass_rate_list = []

        # ------------------- Iterate over the data loader -------------#
        # ---------- Feature list: title, question body, Input description, Output description, I/O test cases and Note description, some slow code, label code -------#
        for batch_index, (feature_list, label_code, input_output_case_path, original_data_path) in enumerate(data_loader):
            if train_or_test == "Train" and batch_index % command_line_args.Train_set_interval != 0:
                continue
            elif train_or_test == "Test" and batch_index % command_line_args.Test_set_interval != 0:
                continue
            else:
                # ----------------------------- Move to GPU -------------------------------#
                for index, tensor_dict in enumerate(feature_list):
                    feature_list[index]["input_ids"] = tensor_dict["input_ids"].to(device)
                    feature_list[index]["attention_mask"] = tensor_dict["attention_mask"].to(device)
                    try:
                        feature_list[index]["token_type_ids"] = tensor_dict["token_type_ids"].to(device)
                    except:
                        pass

                # ----------------------------- Move to GPU -------------------------------#
                generated_code = tensor([[102]])
                generated_code_attention_matrix = tensor([[1]])

                feature_list.append({"input_ids": generated_code.to(device), "attention_mask": generated_code_attention_matrix.to(device)})

                # ---------------------------------------------
                generated_list = []

                # ####################################### Generate a maximum of max_len tokens #################################
                if command_line_args.Maximum_length_pattern_of_generated_code == 0:
                    Maximum_length = command_line_args.Maximum_length_of_generated_code
                else:
                    Maximum_length = len(feature_list[5]["input_ids"][0])
                for _ in range(Maximum_length):

                    # -------------------------------- Feed into GPT-2 model ----------------#
                    model_output = model(feature_list).logits

                    # ----------------------- The model's predicted sequence -------------------------#
                    # ----------------------- The predicted probability of the next Token by the model -----------#
                    next_token_probability = model_output[0, -1, :]

                    # ------------- Add a repetition penalty for each token in the generated result, reducing its generation probability -------#
                    for id in set(generated_list):  # Meaning of the set
                        next_token_probability[id] /= 1

                    # ------------------------------------ Temperature: Process predicted probability -----------------------------#
                    next_token_probability = next_token_probability / command_line_args.Temperature

                    # ----------- The probability for [UNK] is set to negative infinity, i.e., the model's predicted result cannot be the [UNK] token -----------#
                    next_token_probability[102] = -float('Inf')

                    # --------------------------------------------- Filter ----------------------------------------------#
                    filtered_next_token_probability = TopK_and_nucleus_method_filter_function(next_token_probability, TopK=command_line_args.topk, TopP=command_line_args.topp)

                    # torch.multinomial represents drawing num_samples elements without replacement from the candidate set. The higher the weight, the higher the chance of being drawn, returning the index of the element
                    predicted_token = torch.multinomial(F.softmax(filtered_next_token_probability, dim=-1), num_samples=1)

                    # ------------------------------ If encountering [SEP], the response generation ends --------------#
                    if predicted_token == 50256:
                        break

                    # ------------------------------- Add to the generated list ----------------#
                    generated_list.append(predicted_token.item())

                    # -------------------------------- Concatenate the part of the code generated by the model with the original input text ------#
                    if Task == 0:
                        feature_list[-1]["input_ids"] = torch.cat((feature_list[-1]["input_ids"], tensor([[predicted_token]]).to(device)), dim=1)
                        feature_list[-1]["attention_mask"] = torch.cat((feature_list[-1]["attention_mask"], tensor([[1]]).to(device)), dim=1)
                    elif Task == 0:
                        feature_list[-1]["input_ids"] = torch.cat((feature_list[-1]["input_ids"], tensor([[predicted_token]]).to(device)), dim=1)
                        feature_list[-1]["attention_mask"] = torch.cat((feature_list[-1]["attention_mask"], tensor([[1]]).to(device)),dim=1)
                    elif Task == 2:
                        feature_list[0]["input_ids"] = torch.cat((feature_list[0]["input_ids"], tensor([[predicted_token]]).to(device)), dim=1)
                        feature_list[0]["attention_mask"] = torch.cat((feature_list[0]["attention_mask"], tensor([[1]]).to(device)), dim=1)
                        feature_list[-1]["input_ids"] = torch.cat((feature_list[-1]["input_ids"], tensor([[predicted_token]]).to(device)), dim=1)
                        feature_list[-1]["attention_mask"] = torch.cat((feature_list[-1]["attention_mask"], tensor([[1]]).to(device)),
                                                               dim=1)

                # ------------------------------ Convert to output text -------------------------#
                output_text = TokenizerGPT_vocab.batch_decode(feature_list[-1]["input_ids"])[0].replace("[CLS]", "")

                # ===================================================== Compute Bleu score ===================================#
                reference_list = [
                    [label_code],
                ]
                model_generated_list = [output_text]
                bleu = BLEU()
                bleu_score = bleu.corpus_score(model_generated_list, reference_list).score
                BLEU_score_list.append(bleu_score)

                # ===================================================== Compute CodeBleu score ===============================#
                try:
                    Codebleu_score = round(_bleu(label_code, output_text), 2)
                except:
                    Codebleu_score = 0

                CodeBLEU_score_list.append(Codebleu_score)

                # ----------------------------- Save record file -----------------Generated_code/Round_{epoch}_prediction_code/Train------------------------------------------#
                question_index = original_data_path.split("/")[0]
                inefficient_code_name = original_data_path.split("/")[-1]

                os.makedirs(f"Generated_code/{question_index}", exist_ok=True)
                with open(f"Generated_code/{question_index}/{inefficient_code_name}", 'w', encoding='UTF-8') as f:
                    f.write(output_text.strip())

                # ------------------------------------------ Update log information ----------------------------------------------#
                projection_log.info(
                    f'Prediction_generation_code, {train_or_test}, {batch_index}(Total:{len(data_loader)}), CodeBLEU:{Codebleu_score:.3f}, BLEU:{bleu_score:.3f}, compile rate:{-1:.3f}, IO_test_pass_rate:{-1:.3f}')
                projection_log.info(f'{train_or_test}, {batch_index}(Total:{len(data_loader)})')
                projection_log.info(f'Prediction code: {output_text.strip()}')
                projection_log.info(f'Gold code: {label_code.strip()}')

    # -------------------------- Write information to the log file ----------------------- #
    if train_or_test == "Train":
        log_information = f"Train_prediction_generation_code, average_Bleu:{sum(BLEU_score_list) / len(BLEU_score_list):.3f}, average_CodeBleu:{sum(CodeBLEU_score_list) / len(CodeBLEU_score_list):.3f}, average_compile_rate:{-1:.3f}, average_IO_test_pass_rate:{-1:.3f}"
    else:
        log_information = f"Test_prediction_generation_code, average_Bleu:{sum(BLEU_score_list) / len(BLEU_score_list):.3f}, average_CodeBleu:{sum(CodeBLEU_score_list) / len(CodeBLEU_score_list):.3f}, average_compile_rate:{-1:.3f}, average_IO_test_pass_rate:{-1:.3f}"
    projection_log.info(log_information)

    return





# ####################################################################### Main Function #######################################
def main(round_num, task):
    # -------------------------- Initialize command line arguments -----#
    cmd_args = get_command_line_args()

    # -------------------------------- Create log object -----#
    # ----------- Logging to both file and console -------#
    global projection_log
    projection_log = create_log_file(cmd_args)

    # ================================================================ Use GPU if available ===================#
    os.environ['CUDA_VISIBLE_DEVICES'] = cmd_args.device
    device = 'cuda' if cmd_args.cuda else 'cpu'
    projection_log.info('using:{}'.format(device))

    # ======================================= Create output directory for generated code ======================#
    if not os.path.exists(f"Generated_code/Round_{round_num}_prediction_code/Train"):
        os.makedirs(f"Generated_code/Round_{round_num}_prediction_code/Train")
    if not os.path.exists(f"Generated_code/Round_{round_num}_prediction_code/Test"):
        os.makedirs(f"Generated_code/Round_{round_num}_prediction_code/Test")

    # --------------------- Load GPT model ------------------------#
    if task == -1:
        if cmd_args.task == 0:
            model = E_code_Model(cmd_args)
        elif cmd_args.task == 1:
            model = No_expert_Model(cmd_args)
        elif cmd_args.task == 2:
            model = GPT_Neo_Model(cmd_args)
    else:
        if task == 0:
            model = E_code_Model(cmd_args)
        elif task == 1:
            model = No_expert_Model(cmd_args)
        elif task == 2:
            model = GPT_Neo_Model(cmd_args)
    # ------------- Move to GPU ----#
    model.to(device)

    # -------------------------- Use multiple GPUs if available ------------------------#
    use_multi_gpus = False

    # ------------------------------------- Record total model parameters --------------------------------------------#
    total_params = 0
    model_params = model.parameters()
    for param in model_params:
        total_params += param.numel()
    projection_log.info(f'================When tested Model Total number of model parameters : {total_params} ====================')

    # ========================================================================= Begin code prediction ===========================#
    if task == -1:
        if cmd_args.task == 0:
            # train_data = Train_Dataset()
            test_data = Test_Dataset()
            # train_code_bleu_score, train_bleu_score = prediction_generation_function(model, device, train_data, use_multi_gpus, cmd_args, "Train", round_num, E_code_padding_function, cmd_args.task)
            test_code_bleu_score, test_bleu_score = prediction_generation_function(model, device, test_data, use_multi_gpus, cmd_args, "Test", round_num, E_code_padding_function, cmd_args.task)
        elif cmd_args.task == 1:
            train_data = No_expert_Train_Dataset()
            test_data = No_expert_Test_Dataset()
            train_code_bleu_score, train_bleu_score = prediction_generation_function(model, device, train_data, use_multi_gpus, cmd_args, "Train", round_num, No_expert_padding_function, cmd_args.task)
            test_code_bleu_score, test_bleu_score = prediction_generation_function(model, device, test_data, use_multi_gpus, cmd_args, "Test", round_num, No_expert_padding_function, cmd_args.task)
        elif cmd_args.task == 2:
            train_data = GPT_Neo_Train_Dataset()
            test_data = GPT_Neo_Test_Dataset()
            train_code_bleu_score, train_bleu_score = prediction_generation_function(model, device, train_data, use_multi_gpus, cmd_args, "Train", round_num, GPT_padding_function, cmd_args.task)
            test_code_bleu_score, test_bleu_score = prediction_generation_function(model, device, test_data, use_multi_gpus, cmd_args, "Test", round_num, GPT_padding_function, cmd_args.task)
    else:
        if task == 0:
            train_data = Train_Dataset()
            test_data = Test_Dataset()
            train_code_bleu_score, train_bleu_score = prediction_generation_function(model, device, train_data, use_multi_gpus, cmd_args, "Train", round_num, E_code_padding_function, task)
            test_code_bleu_score, test_bleu_score = prediction_generation_function(model, device, test_data, use_multi_gpus, cmd_args, "Test", round_num, E_code_padding_function, task)
        elif task == 1:
            train_data = No_expert_Train_Dataset()
            test_data = No_expert_Test_Dataset()
            train_code_bleu_score, train_bleu_score = prediction_generation_function(model, device, train_data, use_multi_gpus, cmd_args, "Train", round_num, No_expert_padding_function, cmd_args.task)
            test_code_bleu_score, test_bleu_score = prediction_generation_function(model, device, test_data, use_multi_gpus, cmd_args, "Test", round_num, No_expert_padding_function, cmd_args.task)
        elif task == 2:
            train_data = GPT_Neo_Train_Dataset()
            test_data = GPT_Neo_Test_Dataset()
            train_code_bleu_score, train_bleu_score = prediction_generation_function(model, device, train_data, use_multi_gpus, cmd_args, "Train", round_num, GPT_padding_function, task)
            test_code_bleu_score, test_bleu_score = prediction_generation_function(model, device, test_data, use_multi_gpus, cmd_args, "Test", round_num, GPT_padding_function, task)

    # ============================================================ Final output ============================================#
    projection_log.info(f'================= Prediction_generation_code: epochs {round_num} : Train_CodeBLEU: {train_code_bleu_score}  ,  Train_BLEU: {train_bleu_score} ==')
    projection_log.info(f'================= Prediction_generation_code: epochs {round_num} : Test_CodeBLEU: {test_code_bleu_score}   ,   Test_BLEU: {test_bleu_score} ==')

    projection_log.info('######################################## End Prediction_generation_code #####################')

    IO_testing_of_generated_code_main()

# ############################################################ Entry Point ####################################################
if __name__ == '__main__':
    # ===================================================== Load data ==================================================#
    main("0", -1)
