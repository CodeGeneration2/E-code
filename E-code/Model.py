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

# ######################################################## Build Fully connected neural network for regression #################################
class ECodeModel(nn.Module):
    # =========================================== Initialization =======================#
    def __init__(self, cmdline_args):
        super(ECodeModel, self).__init__()

        # ============================ If there are already trained local models, use the trained local models ===========================#
        if cmdline_args.use_trained_local_models:
            # Define Expert Group: Bert output matrix [Batch size, Sentence length, Word embedding]
            self.TitleLayer = BertModel.from_pretrained(f"{cmdline_args.generated_models}/TitleLayer/")
            self.ProblemDescriptionBodyLayer = BertModel.from_pretrained(f"{cmdline_args.generated_models}/ProblemDescriptionBodyLayer/")
            self.InputDescriptionLayer = BertModel.from_pretrained(f"{cmdline_args.generated_models}/InputDescriptionLayer/")
            self.OutputDescriptionLayer = BertModel.from_pretrained(f"{cmdline_args.generated_models}/OutputDescriptionLayer/")
            self.IOSampleTestingAndNoteDescriptionLayer = BertModel.from_pretrained(f"{cmdline_args.generated_models}/IOSampleTestingAndNoteDescriptionLayer/")

            # Expert Group Integration Layer
            self.ExpertGroupIntegrationLayer = BertModel.from_pretrained(f"{cmdline_args.generated_models}/ExpertGroupIntegrationLayer/")

            # MLP Enlarge Layer
            self.MLPEnlargeLayer = torch.load(f"{cmdline_args.generated_models}/MLPEnlargeLayer.pkl")

            # IC Layer
            self.ICLayer = transformers.GPTNeoModel.from_pretrained(f"{cmdline_args.generated_models}/ICLayer/")
            self.ECLayer = transformers.GPTNeoModel.from_pretrained(f"{cmdline_args.generated_models}/ECLayer/")

            # Multi-headed attention mechanism
            self.MultiHeadedAttentionMechanism = torch.load(f"{cmdline_args.generated_models}/MultiHeadedAttentionMechanism.pkl")

            # Final output layer
            self.FinalOutputLayer = transformers.GPTNeoForCausalLM.from_pretrained(f"{cmdline_args.generated_models}/FinalOutputLayer/")

        # ================================ If there are no trained models, use the initial models ================================#
        else:
            # Define Expert Group parameters
            bert_params = BertConfig.from_json_file("BertTinyWeights/config.json")

            # Define Expert Group: Bert output matrix [Batch size, Sentence length, Word embedding]
            self.TitleLayer = BertModel.from_pretrained("BertTinyWeights/", config=bert_params)
            self.ProblemDescriptionBodyLayer = BertModel.from_pretrained("BertTinyWeights/", config=bert_params)
            self.InputDescriptionLayer = BertModel.from_pretrained("BertTinyWeights/", config=bert_params)
            self.OutputDescriptionLayer = BertModel.from_pretrained("BertTinyWeights/", config=bert_params)
            self.IOSampleTestingAndNoteDescriptionLayer = BertModel.from_pretrained("BertTinyWeights/", config=bert_params)

            # Expert Group Integration Layer
            bert_params = BertConfig.from_json_file("WeightsExpertGroupIntegrationLayer/config.json")
            self.ExpertGroupIntegrationLayer = BertModel(config=bert_params)

            # MLP Enlarge Layer
            if cmdline_args.RELU:
                self.MLPEnlargeLayer = nn.Sequential(
                    nn.Linear(in_features=128, out_features=768, bias=True),
                    nn.ReLU()
                )
            else:
                self.MLPEnlargeLayer = nn.Linear(in_features=128, out_features=768, bias=True)

            # IC Layer
            self.ICLayer = transformers.GPTNeoModel.from_pretrained(cmdline_args.GPT_arch)
            self.ECLayer = transformers.GPTNeoModel.from_pretrained(cmdline_args.GPT_arch)

            # Multi-headed attention mechanism
            self.MultiHeadedAttentionMechanism = nn.MultiheadAttention(embed_dim=768, num_heads=cmdline_args.heads, batch_first=True)

            # Final output layer
            self.FinalOutputLayer = transformers.GPTNeoForCausalLM.from_pretrained(cmdline_args.GPT_arch)

    # ################################################## Define the forward propagation path of the network #########################################
    def forward(self, feature_list):
        # Decomposition
        title, problem_description_body, input_description, output_description, io_sample_testing_and_note_description, inefficient_code, label_code = feature_list

        # Define 5 Expert Groups
        title_layer_output = self.TitleLayer(**title).last_hidden_state
        problem_description_body_layer_output = self.ProblemDescriptionBodyLayer(**problem_description_body).last_hidden_state
        input_description_layer_output = self.InputDescriptionLayer(**input_description).last_hidden_state
        output_description_layer_output = self.OutputDescriptionLayer(**output_description).last_hidden_state
        io_sample_testing_and_note_description_layer_output = self.IOSampleTestingAndNoteDescriptionLayer(**io_sample_testing_and_note_description).last_hidden_state

        # Expert Group Integration Layer
        expert_output_concat = torch.cat((title_layer_output, problem_description_body_layer_output, input_description_layer_output, output_description_layer_output, io_sample_testing_and_note_description_layer_output), dim=1)
        if len(expert_output_concat[0]) > 2048:
            expert_output_concat = expert_output_concat[:,:2048,:]

        # Expert Group Integration Layer
        expert_group_output = self.ExpertGroupIntegrationLayer(inputs_embeds=expert_output_concat).last_hidden_state

        # MLP Enlarge Layer
        text_layer_output = self.MLPEnlargeLayer(expert_group_output)

        # IC Layer
        inefficient_code_layer_output = self.ICLayer(**inefficient_code).last_hidden_state

        # Encoder output
        encoder_output = torch.cat((text_layer_output, inefficient_code_layer_output), dim=1)

        # EC Layer
        efficient_code_layer_output = self.ECLayer(**label_code).last_hidden_state

        # Multi-headed attention mechanism
        encoder_decoder_self_attention_layer_output, attn_output_weights = self.MultiHeadedAttentionMechanism(efficient_code_layer_output, encoder_output, encoder_output)

        # Process label code
        label_tensor = label_code["input_ids"].clone().detach()
        for i in range(len(label_tensor)):
            for j in range(len(label_tensor[i])):
                if label_tensor[i, j] == 0:
                    label_tensor[i, j] = -100

        # Final output layer
        final_output = self.FinalOutputLayer(inputs_embeds=encoder_decoder_self_attention_layer_output, labels=label_tensor)

        # Output
        return final_output.loss, final_output.logits
