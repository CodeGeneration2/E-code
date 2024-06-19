# -*- coding: utf-8 -*-

import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# from numpy import mean
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM

from Dataset_prompt_T5 import MyDataset

# model_path = "TrainedModelParameters_PolyCoder_400M_prompt/checkpoint-11865"
# model_path = "TrainedModelParameters_GPT-NEO 125M_prompt/checkpoint-11865"
# model_name = "TrainedModelParameters_GPT-NEO 125M_prompt"
# model_path = "GPT-NEO 125M"
# model_name = "UntrainedModel_GPT-NEO 125M"

model_path = "TrainedModel_codet5-base_prompt/checkpoint-11865"
model_name = "TrainedModel_codet5-base_prompt"

# beam_sample_num = "UseBeamSearch"
beam_sample_num = "UseSamplingMethod"

# model_type = "DecoderModel"
model_type = "Encoder-DecoderModel"

# ======================================================== Load Training Data =============================================#
test_data = MyDataset(
    dataset_path="../../GEC/test",
    max_token_num=1792,
    model_path=model_path,
    train_or_predict="predict"
)

# ------------------------------------ Tokenizer ------------------------------------#
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ------------------------------------ Vocab ------------------------------------#
if "codegen" in model_path or "PolyCoder" in model_path or "GPT-NEO" in model_path:
    tokenizer.pad_token = tokenizer.eos_token
elif "incoder" in model_path:
    tokenizer.pad_token = "<pad>"

# ---------------------------------------- Model ---------------------------------------#
if "codet5" in model_path:
    model = T5ForConditionalGeneration.from_pretrained(model_path)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path)

# ---------------------------------------- Model ---------------------------------------#
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# ---------------------------------- Calculate Bleu ----------------------------------------#
BLEU_score_list = []
CodeBLEU_score_list = []

# ######################################################################################################################
for index, test_tensor in tqdm(enumerate(test_data)):

    # #################################################################################################################
    partial_data_path = test_tensor["code_path"].split("test")[-1][:-5].replace("Acc_tle_solutions/", "")
    generated_code_path = f"./{model_name}_GeneratedCode/{partial_data_path}"
    if not os.path.exists(generated_code_path):
        os.makedirs(generated_code_path)
    else:
        check_list = os.listdir(f"{generated_code_path}")
        if len(check_list) != 0:
            continue

    if model_type == "Encoder-DecoderModel":
        max_output_length = 512
        min_output_length = 8
    elif model_type == "DecoderModel":
        input_length = len(test_tensor['input_ids'][0])
        max_output_length = min(input_length + 512, 2048)
        min_output_length = input_length + 8

    # =================================================== Simply generate a single sequence ===========================
    with torch.no_grad():
        if beam_sample_num == "UseSamplingMethod":
            generated_token_list = model.generate(test_tensor['input_ids'].to(device),
                                                  attention_mask=test_tensor["attention_mask"].to(device),
                                                  max_length=max_output_length,
                                                  min_length=min_output_length,
                                                  num_return_sequences=5,
                                                  no_repeat_ngram_size=2,
                                                  early_stopping=True,
                                                  temperature=0.25,
                                                  do_sample=True,
                                                  top_k=3,
                                                  top_p=0.7,
                                                  pad_token_id=tokenizer.eos_token_id,
                                                  )
        elif beam_sample_num == "UseBeamSearch":
            generated_token_list = model.generate(test_tensor['input_ids'].to(device),
                                                  max_length=2048,
                                                  min_length=input_length + 8,
                                                  num_beams=beam_sample_num,
                                                  num_return_sequences=beam_sample_num,
                                                  no_repeat_ngram_size=2,
                                                  early_stopping=True,
                                                  )

    # ======================================= Mistake: GPT's output includes the input !!! =========================#
    for code_index, generated_token in enumerate(generated_token_list):

        if model_type == "DecoderModel":
            generated_token = generated_token[input_length:]

        predicted_code = tokenizer.decode(generated_token, skip_special_tokens=True)
        predicted_code = predicted_code.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
        predicted_code = predicted_code.strip()  # Remove leading and trailing spaces

        # if model_type == "DecoderModel":
        #     predicted_code = "def function1(" + predicted_code

        # ---------------------------- Save to file -----------------------------------------------------------#
        with open(f"{generated_code_path}/{code_index}.txt", 'w', encoding='UTF-8') as f:
            f.write(predicted_code)
