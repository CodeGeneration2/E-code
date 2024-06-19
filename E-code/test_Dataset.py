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
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root_path="../../GEC"):
        self.Tokenized_GPT = transformers.GPT2Tokenizer.from_pretrained("GPT_Tokenizer/", pad_token="[PAD]",
                                                                        cls_token="[CLS]")
        self.Total_Data_List = []  # Should be set in initialization_function()
        self.initialization_function(f"{dataset_root_path}/test")

    def initialization_function(self, dataset_root_path):
        total_data_list = []
        training_data_list = os.listdir(f"{dataset_root_path}")

        for entry in tqdm(training_data_list):
            with open(f"{dataset_root_path}/{entry}/Accepted.json", 'r', encoding='UTF-8') as f:
                label_code = f.read()
            label_code = label_code.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

            label_code_tensor_dict = self.Tokenized_GPT(label_code, return_tensors="pt")
            if len(label_code_tensor_dict["input_ids"][0]) > 766:
                continue

            label_code = reindent_code_function(label_code)
            with open(f"{dataset_root_path}/{entry}/metadata.json", 'r', encoding='UTF-8') as f:
                title_dict = eval(f.read())
            title = f"Title:{title_dict['title']}\nDifficulty:{title_dict['difficulty']}"

            with open(f"{dataset_root_path}/{entry}/question dictionary.txt", 'r', encoding='UTF-8') as f:
                question_dict = eval(f.read())

            question_description_body = question_dict["NL"]
            input_description = f"Input:{question_dict['input']}"
            output_description = f"Output:{question_dict['output']}"
            io_test_samples_and_note_description = f"I/O test:{question_dict['IO test samples']}\nNote:{question_dict['note']}"

            slow_code_set_list = os.listdir(f"{dataset_root_path}/{entry}/Acc_tle_solutions")
            for some_code in slow_code_set_list:
                with open(f"{dataset_root_path}/{entry}/Acc_tle_solutions/{some_code}", 'r', encoding='UTF-8') as f:
                    slow_code = f.read()
                slow_code = slow_code.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

                slow_code = reindent_code_function(slow_code)
                slow_code_tensor_dict = self.Tokenized_GPT(slow_code, return_tensors="pt")
                if len(slow_code_tensor_dict["input_ids"][0]) > 768:
                    continue

                io_case_test_path = f"{dataset_root_path}/{entry}/IO Case Test Dictionary.txt"
                original_data_path = f"{entry}/{some_code}"
                data_tuple = (title.strip(),
                              question_description_body.strip(),
                              input_description.strip(),
                              output_description.strip(),
                              io_test_samples_and_note_description.strip(),
                              slow_code.strip(),
                              label_code.strip(),
                              io_case_test_path,
                              original_data_path)
                total_data_list.append(data_tuple)

        self.Total_Data_List = total_data_list

    def __len__(self):
        return len(self.Total_Data_List)

    def __getitem__(self, index):
        sample_list = self.Total_Data_List[index]
        return sample_list


def reindent_code_function(code_string):
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
    return reindented_code_string.getvalue()
