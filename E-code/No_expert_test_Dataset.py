# coding=utf-8
"""
Dataset to be used for APPS Training
"""

import torch
import glob
import logging
import random
import fnmatch

from multiprocessing import Manager
# from multiprocessing.shared_memory import ShareableList

import dataset_lm.util as dsutil
import numpy as np
import gc
import os
import io

import transformers

from dataset_lm.reindent import run as run_reindent
from tqdm import tqdm


import json

# ############################################################# My Dataset Function #####################################
class NoExpertTestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root_path="ECG"):


        self.Tokenized_GPT = transformers.GPT2Tokenizer.from_pretrained("GPT_Tokenizer/", pad_token="[PAD]", cls_token="[CLS]")

        self.Total_Data_List = []  # Should be set in initialize_function()

        # ================================= Initialize function (import data from local) ==================#
        self.initialize(f"{dataset_root_path}/test")

    # =========================================== Initialize function (import data from local) =========================================#
    def initialize(self, dataset_root_path):
        # """ Import data from local
        # Returns:
        #     self.Total_Data_List = Total_Data_List
        # """

        total_data_list = []

        training_set_list = os.listdir(f"{dataset_root_path}")
        # ----------------------------------------- Import data ------------------------------------------------------------#
        for item in tqdm(range(len(training_set_list))):
            # ----------------------------------------------- Labels --------------------------------#
            with open(f"{dataset_root_path}/{item}/Accepted.txt", 'r', encoding='UTF-8') as f:
                label_code = f.read()

            # -------------------------------- Filtering -----------------------------#
            label_code_tensor_dict = self.Tokenized_GPT(label_code, return_tensors="pt")
            if len(label_code_tensor_dict["input_ids"][0]) > 766:
                continue

            # ------------------------------- Indent code ----------------------#
            label_code = reindent_code_function(label_code)

            # ------------------------------------------- Title ---------------------------------#
            with open(f"{dataset_root_path}/{item}/Question.txt", 'r', encoding='UTF-8') as f:
                total_question_text = f.read()

            # ============================================================= Features ====================================#
            slow_code_list = os.listdir(f"{dataset_root_path}/{item}/Acc_tle_solutions")

            # ============================================== Import item feature set ===================================#
            for some_code in slow_code_list:
                with open(f"{dataset_root_path}/{item}/Acc_tle_solutions/{some_code}", 'r', encoding='UTF-8') as f:
                    specific_slow_code = f.read()

                # ------------------------------- Indent code ----------------------#
                specific_slow_code = reindent_code_function(specific_slow_code)
                specific_slow_code_tensor_dict = self.Tokenized_GPT(specific_slow_code, return_tensors="pt")
                if len(specific_slow_code_tensor_dict["input_ids"][0]) > 768:
                    continue

                # ---------------------------------------- Input output case path ----------------------------------------#
                input_output_case_path = f"{dataset_root_path}/{item}/IO Case Test Dictionary.txt"

                # ---------------------------------------- Original data path ----------------------------------------#
                original_data_path = f"{dataset_root_path}/{item}/Acc_tle_solutions/{some_code}"

                # ---------------------------------------- Minimal unit ----------------------------------------#
                item_data_tuple = (total_question_text, specific_slow_code, label_code, input_output_case_path, original_data_path)

                # -------------------------- Add to total data list -----------#
                total_data_list.append(item_data_tuple)

        self.Total_Data_List = total_data_list


    def __len__(self):
        return len(self.Total_Data_List)

    # ========================================= Iterative traversal function =========================================#
    def __getitem__(self, index):

        # ----------(Title, question description body, Input description, Output description, Input output example test, Note description, specific slow code, label code, original data path)--------#
        sample_list = self.Total_Data_List[index]

        return sample_list


def reindent_code_function(code_string):
    # """
    # Given code string, reindent it in the same way that the Github dataset was indented
    # """
    # --------------------------------- Mutable string_io.stringIO operations ---------------------------------#
    code_string = io.StringIO(code_string)
    reindented_code_string = io.StringIO()

    run_reindent(
        code_string,
        reindented_code_string,
        config={
            "dry-run": False,
            "help": False,
            "to": 4,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 4,
            "all-tabs": False
        }
    )

    # ------------------- Get object value ---------------#
    return reindented_code_string.getvalue()
