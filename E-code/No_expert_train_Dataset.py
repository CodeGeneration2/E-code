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
class NonExpertTrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root_path="ECG"):

        self.Tokenized_GPT = transformers.GPT2Tokenizer.from_pretrained("GPT_Tokenizer/", pad_token="[PAD]", cls_token="[CLS]")

        self.Total_Data_List = []  # Should be set in initialize_function()

        # Initialize function (Import data from local)
        self.Initialize(f"{dataset_root_path}/train")

    # Initialization function (Import data from local)
    def Initialize(self, dataset_root_path):

        total_data_list = []

        training_set_list = os.listdir(f"{dataset_root_path}")
        for item in tqdm(range(len(training_set_list))):
            # Load label data
            with open(f"{dataset_root_path}/{item}/Accepted.txt", 'r', encoding='UTF-8') as f:
                label_code = f.read()

            # Filter out
            label_code_tensor_dict = self.Tokenized_GPT(label_code, return_tensors="pt")
            if len(label_code_tensor_dict["input_ids"][0]) > 766:
                continue

            # Reindent code
            label_code = reindent_code_function(label_code)

            # Load title data
            with open(f"{dataset_root_path}/{item}/Question.txt", 'r', encoding='UTF-8') as f:
                total_question_text = f.read()

            # Load features
            slow_code_list = os.listdir(f"{dataset_root_path}/{item}/Acc_tle_solutions")

            # Import each feature
            for code in slow_code_list:
                with open(f"{dataset_root_path}/{item}/Acc_tle_solutions/{code}", 'r', encoding='UTF-8') as f:
                    specific_slow_code = f.read()

                # Reindent code
                specific_slow_code = reindent_code_function(specific_slow_code)
                specific_slow_code_tensor_dict = self.Tokenized_GPT(specific_slow_code, return_tensors="pt")
                if len(specific_slow_code_tensor_dict["input_ids"][0]) > 768:
                    continue

                # I/O test case path
                io_case_test_path = f"{dataset_root_path}/{item}/IO Case Test Dictionary.txt"

                # Original data path
                original_data_path = f"{dataset_root_path}/{item}/Acc_tle_solutions/{code}"

                # Minimum data unit
                data_tuple = (total_question_text, specific_slow_code, label_code, io_case_test_path, original_data_path)

                # Add to total data list
                total_data_list.append(data_tuple)

        self.Total_Data_List = total_data_list

    def __len__(self):
        return len(self.Total_Data_List)

    # Iteration function
    def __getitem__(self, index):

        # (Title, Question Body, Input Description, Output Description, I/O Sample Test, Note Description, Specific Slow Code, Label Code, Original Data Path)
        sample_list = self.Total_Data_List[index]

        return sample_list

def reindent_code_function(code_string):
    """
    Given the code string, reindent it in the same way that the Github dataset was indented.
    """
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

    # Retrieve the value of the object
    return reindented_code_string.getvalue()
