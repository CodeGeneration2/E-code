# coding=utf-8

from transformers import BertTokenizer, BertModel
import transformers
from transformers import BertConfig, BertModel
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

from torch.utils.data import Dataset
import torch

# ######################################################## Building Fully Connected Neural Network for Regression ##############################
class NoExpertModel(nn.Module):
    # =========================================== Initialization =======================#
    def __init__(self, cmdline_args):
        super(NoExpertModel, self).__init__()

        # ============================ If there are already trained local models, use them ===========================#
        if cmdline_args.whether_to_use_trained_local_models:
            # ------------------------------- Opened Expert Group Layer ------------------------------------#
            self.opened_expert_group_layer = BertModel.from_pretrained(f"{cmdline_args.generated_models}/Opened_expert_group_layer/")

            # ------------------------------- Expert Group Integration Layer ------------------------------------#
            self.expert_group_integration_layer = BertModel.from_pretrained(f"{cmdline_args.generated_models}/Expert_Group_Integration_Layer/")

            # ---------------------------------------- MLP Enlarge Layer -----------------------#
            self.mlp_enlarge_layer = torch.load(f"{cmdline_args.generated_models}/MLP_enlarge_layer.pkl")

            # ---------------------------------------- IC Layer ----------------------------------------
            self.ic_layer = transformers.GPTNeoModel.from_pretrained(f"{cmdline_args.generated_models}/IC_layer/")
            self.ec_layer = transformers.GPTNeoModel.from_pretrained(f"{cmdline_args.generated_models}/EC_layer/")

            # ---------------------------------------- Multi-Headed Attention Mechanism ----------------------------------------#
            self.multi_headed_attention_mechanism = torch.load(f"{cmdline_args.generated_models}/Multi_headed_attention_mechanism.pkl")

            # ---------------------------------------- Final Output Layer ----------------------------------------#
            self.final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained(f"{cmdline_args.generated_models}/Final_output_layer/")

        # ================================ If no pre-trained models exist, use initial models ================================#
        else:
            # --------------------- Define Expert Group Parameters --------------------------------#
            bert_params = BertConfig.from_json_file("Bert_tiny_Opened_expert_group_layer_Weights/config.json")
            self.opened_expert_group_layer = BertModel(config=bert_params)

            # ------------------------------- Expert Group Integration Layer ------------------------------------#
            bert_params = BertConfig.from_json_file("Weights_Expert_Group_Integration_Layer/config.json")
            self.expert_group_integration_layer = BertModel(config=bert_params)

            # ---------------------------------------- MLP Enlarge Layer -----------------------#
            if cmdline_args.RELU:
                self.mlp_enlarge_layer = nn.Sequential(
                    nn.Linear(in_features=128, out_features=768, bias=True),
                    nn.ReLU()
                )
            else:
                self.mlp_enlarge_layer = nn.Linear(in_features=128, out_features=768, bias=True)

            # ---------------------------------------- IC Layer ----------------------------------------
            self.ic_layer = transformers.GPTNeoModel.from_pretrained(cmdline_args.GPT_arch)
            self.ec_layer = transformers.GPTNeoModel.from_pretrained(cmdline_args.GPT_arch)

            # ---------------------------------------- Multi-Headed Attention Mechanism ----------------------------------------#
            self.multi_headed_attention_mechanism = nn.MultiheadAttention(embed_dim=768, num_heads=cmdline_args.heads, batch_first=True)

            # ---------------------------------------- Final Output Layer ----------------------------------------#
            self.final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained(cmdline_args.GPT_arch)

    # ################################################## Define Network Forward Propagation #########################################
    def forward(self, feature_list):
        # ------------------------------- Decompose ------------------------------#
        total_question_text, some_slow_code, label_code = feature_list

        # ------------------------------- Define 5 Expert Groups ------------------------------#
        opened_expert_group_output = self.opened_expert_group_layer(**total_question_text).last_hidden_state
        expert_group_output = self.expert_group_integration_layer(inputs_embeds=opened_expert_group_output).last_hidden_state

        # ---------------------------------------- MLP Enlarge Layer -------------#
        text_layer_output = self.mlp_enlarge_layer(expert_group_output)

        # ---------------------------------------- IC Layer -------------#
        inefficient_code_layer_output = self.ic_layer(**some_slow_code).last_hidden_state

        # ---------------------------------------- Encoder Output -------------#
        encoder_output = torch.cat((text_layer_output, inefficient_code_layer_output), dim=1)

        # -------------------------------------------- EC Layer ----------------------------------------------------#
        efficient_code_layer_output = self.ec_layer(**label_code).last_hidden_state

        # -------------------------------------------- Multi-Headed Attention Mechanism -----------------------------------------#
        encoder_decoder_attention_output, attn_output_weights = self.multi_headed_attention_mechanism(efficient_code_layer_output, encoder_output, encoder_output)

        # ---------------------------------------- Label Code Processing ----------------------------------------#
        label_tensor = label_code["input_ids"].clone().detach()
        for i in range(len(label_tensor)):
            for j in range(len(label_tensor[i])):
                if label_tensor[i, j] == 0:
                    label_tensor[i, j] = -100

        # ---------------------------------------- Final Output Layer ----------------------------------------#
        final_output = self.final_output_layer(inputs_embeds=encoder_decoder_attention_output, labels=label_tensor)

        # ----------------------------------- Output -------------------
        return final_output
