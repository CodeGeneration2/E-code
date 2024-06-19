# coding=utf-8
"""
Dataset to be used for APPS Training
Note: Success with GPT
"""

import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import json

# ################################################################# MyDataset #####################################
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, max_token_count=512, model_path=r'F:\PythonPureData\LargeLanguageModelParams\codet5-small', train_or_predict=""):
        self.dataset_path = dataset_path
        self.max_token_count = max_token_count
        self.model_path = model_path
        self.train_or_predict = train_or_predict

        # ------------------------------------ Tokenization ------------------------------------#
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if "codegen" in model_path or "PolyCoder" in model_path or "GPT-NEO" in model_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif "incoder" in model_path:
            self.tokenizer.pad_token = "<pad>"

        self.total_data_list = []  # Should be set in initialize()
        self.total_data_dict = {}  # Should be set in initialize()

        # ========================= Initialization (loading data from local) ===============#
        self.initialize()

    # =========================================== Initialization (loading data from local) =========================================#
    def initialize(self):
        """
        Load data from local
        Returns:
            self.total_data_list = total_data_list
            self.total_data_dict = total_data_dict

        Assume self.dataset_path is set to folderName/data
        """

        total_data_list = []

        dataset_list = os.listdir(f"{self.dataset_path}")
        for question in tqdm(dataset_list):
            # ------------------------------------------- Title ---------------------------------#
            with open(f"{self.dataset_path}/{question}/Question.txt", 'r', encoding='UTF-8') as f:
                total_question_text = f.read()

            # ----------------------------------------- Label Code ----------------------------#
            with open(f"{self.dataset_path}/{question}/Accepted.json", 'r', encoding='UTF-8') as f:
                accepted_code = f.read()
            accepted_code = accepted_code.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

            # ============================================================= Features ====================================#
            inefficient_code_list = os.listdir(f"{self.dataset_path}/{question}/Acc_tle_solutions")
            # ------------------------------------------- Importing particular feature set ---------------------------------#
            for inefficient_code_name in inefficient_code_list:
                with open(f"{self.dataset_path}/{question}/Acc_tle_solutions/{inefficient_code_name}", 'r', encoding='UTF-8') as f:
                    inefficient_code = f.read()
                inefficient_code = inefficient_code.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

                # ==================================================================================================#
                total_input_feature = f"Natural Language Description:\n{total_question_text}\nInefficient code:\n{inefficient_code}\nPlease provide an efficient version:"
                single_data_tuple = (total_input_feature, accepted_code, f"{self.dataset_path}/{question}/Acc_tle_solutions/{inefficient_code_name}")
                total_data_list.append(single_data_tuple)

        self.total_data_list = total_data_list

        print(f'\033[0:36m========================== {len(total_data_list)} training data loaded ==================\033[m')

    def __len__(self):
        return len(self.total_data_list)

    # ================================================ Iteration function ==================================#
    def __getitem__(self, index):
        input_feature, label_code, code_path = self.total_data_list[index]

        # ---------------------------- Always kept. Postponed bug fix. --------------#
        input_feature = input_feature[:150000]
        label_code = label_code[:150000]

        # ------------------------------------- Encoding ---------------------------------------------#
        feature_encoded_dict = self.tokenizer(input_feature, truncation=True, max_length=self.max_token_count, return_tensors="pt")
        label_encoded_dict = self.tokenizer(label_code, truncation=True, max_length=self.max_token_count, return_tensors="pt")

        if self.train_or_predict == "train":
            # ------------------------------------- Encoding ---------------------------------------------#
            feature_encoded_dict["input_ids"] = feature_encoded_dict['input_ids'].squeeze()
            feature_encoded_dict["attention_mask"] = feature_encoded_dict['attention_mask'].squeeze()
            feature_encoded_dict["labels"] = label_encoded_dict['input_ids'].squeeze()

        elif self.train_or_predict == "predict":
            # ------------------------------------- Encoding ---------------------------------------------#
            feature_encoded_dict["input_feature"] = input_feature
            feature_encoded_dict["label_code"] = label_code
            feature_encoded_dict["code_path"] = code_path

        else:
            print("Error!! self.train_or_predict!!")

        if "incoder" in self.model_path:
            feature_encoded_dict.pop("token_type_ids", None)

        return feature_encoded_dict
