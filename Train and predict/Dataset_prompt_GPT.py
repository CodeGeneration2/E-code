# coding=utf-8
"""
Dataset to be used for APPS Training
Note: Successful GPT
"""

import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import json

# ######################################################################################################################
# total_input_feature = f"Inefficient code:\n{slow_code}\nNatural Language Description:\n{total_question_text}\nPlease provide an efficient version:"


# ################################################################# MyDataset #####################################
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, max_tokens=2048, model_path=r'F:\PythonPureData\LargeLanguageModelParams\codet5-small',
                 train_or_predict=""):

        self.dataset_path = dataset_path
        self.max_tokens = max_tokens
        self.model_path = model_path
        self.train_or_predict = train_or_predict

        # ------------------------------------ Tokenization ------------------------------------#
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if "codegen" in model_path or "PolyCoder" in model_path or "GPT-NEO" in model_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif "incoder" in model_path:
            self.tokenizer.pad_token = "<pad>"

        self.total_data_list = []  # Should be set in initialization()
        self.total_data_dict = {}  # Should be set in initialization()

        # ========================= Initialization (Load data from local) ===============#
        self.initialization()

    # =========================================== Initialization (Load data from local) =========================================#
    def initialization(self):
        """
        Load data from local
        Assume self.dataset_root_path is set to folderName/data
        """

        total_data_list = []

        dataset_list = os.listdir(f"{self.dataset_path}")
        for question in tqdm(dataset_list):

            # ------------------------------------------- Title ---------------------------------#
            with open(f"{self.dataset_path}/{question}/Question.txt", 'r', encoding='UTF-8') as f:
                total_question_text = f.read()

            # ----------------------------------------- Label code ----------------------------#
            with open(f"{self.dataset_path}/{question}/Accepted.json", 'r', encoding='UTF-8') as f:
                accepted_code = f.read()
            accepted_code = accepted_code.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

            # ============================================================= Features ====================================#
            slow_code_list = os.listdir(f"{self.dataset_path}/{question}/Acc_tle_solutions")
            # ------------------------------------------- Load specific feature set ---------------------------------#
            for slow_code_name in slow_code_list:
                with open(f"{self.dataset_path}/{question}/Acc_tle_solutions/{slow_code_name}", 'r', encoding='UTF-8') as f:
                    slow_code = f.read()
                slow_code = slow_code.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

                total_input_feature = f"Natural Language Description:\n{total_question_text}\nInefficient code:\n{slow_code}\nPlease provide an efficient version:\n"

                # ---------------------------------------- Minimal unit ----------------------------------------#
                data_tuple = (total_input_feature, accepted_code, f"{self.dataset_path}/{question}/Acc_tle_solutions/{slow_code_name}")
                # -------------------------- Add to total data list -----------#
                total_data_list.append(data_tuple)

        self.total_data_list = total_data_list

        print(f'\033[0:36m========================== Loaded {len(total_data_list)} training set data ==================\033[m')

    def __len__(self):
        return len(self.total_data_list)

    # ================================================ Iteration function ==================================#
    def __getitem__(self, index):
        input_feature, label_code, code_path = self.total_data_list[index]

        # ---------------------------- Will never be deleted. Delay bug fixes. --------------#
        input_feature = input_feature[:150000]
        label_code = label_code[:150000]

        if self.train_or_predict == "train":
            label_encoding_dict = self.tokenizer(label_code, truncation=True, max_length=self.max_tokens, return_tensors="pt")

            feature_encoding_dict = self.tokenizer(input_feature,
                                                   truncation=True,
                                                   max_length=(2048 - len(label_encoding_dict["input_ids"][0])),
                                                   return_tensors="pt")

            zero_tensor = torch.full(feature_encoding_dict['input_ids'][0].shape, -100)  # Set all to -100 initially
            feature_encoding_dict['input_ids'] = torch.cat((feature_encoding_dict['input_ids'][0], label_encoding_dict['input_ids'][0]), dim=0)
            feature_encoding_dict['attention_mask'] = torch.cat((feature_encoding_dict['attention_mask'][0], label_encoding_dict['attention_mask'][0]), dim=0)
            feature_encoding_dict['labels'] = torch.cat((zero_tensor, label_encoding_dict['input_ids'][0]), dim=0)

        elif self.train_or_predict == "predict":
            feature_encoding_dict = self.tokenizer(input_feature, truncation=True, max_length=self.max_tokens, return_tensors="pt")
            feature_encoding_dict["input_feature"] = input_feature
            feature_encoding_dict["label_code"] = label_code
            feature_encoding_dict["code_path"] = code_path

        else:
            print("Error in self.train_or_predict!!!")

        if "incoder" in self.model_path:
            feature_encoding_dict.pop("token_type_ids", None)

        return feature_encoding_dict
