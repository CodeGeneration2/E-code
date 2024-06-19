import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallel processing
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import PeftConfig, PeftModel, AutoPeftModelForCausalLM

import torch
from transformers import BitsAndBytesConfig
import torch.multiprocessing

os.environ["WANDB_DISABLED"] = "true"

model = "DeepSeek_coder_7b_instruct_v1_5"
model_path = r'DeepSeek_coder_7b_instruct_v1_5'
use_Lora_plugin_model = True
Lora_model_plugin_paths = ['Model_DeepSeek_Coder_Pre_time_RandDel/checkpoint-9044',
                           'Model_DeepSeek_Coder_Pre_time_Loop&Rec/checkpoint-9306',
                           'Model_DeepSeek_Coder_Pre_time_Ori/checkpoint-8188',
                           'Model_DeepSeek_Coder_Pre_time_Uni/checkpoint-7906'
                           ]

q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Word embedding dimensions: [102400, 4096]. trust_remote_code=True: Trust and execute code loaded remotely
tokenizer_vocab = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

model_path = model_path if model_path is not None else '{}'.format(model)

# ---------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             quantization_config=q_config,
                                             torch_dtype=torch.bfloat16,
                                             pad_token_id=tokenizer_vocab.pad_token_id,
                                             eos_token_id=tokenizer_vocab.eos_token_id,
                                             trust_remote_code=True
                                             )
if use_Lora_plugin_model:
    for path in Lora_model_plugin_paths:
        model = PeftModel.from_pretrained(model, path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def predict_code_runtime(predict_code):
    feature_dict = tokenizer_vocab(predict_code, return_tensors="pt")

    with torch.no_grad():
        generated_tokens = model.generate(feature_dict['input_ids'].to(device),
                                          attention_mask=feature_dict["attention_mask"].to(device),
                                          max_new_tokens=4,  # Error: max_length refers to the total length of input + output
                                          do_sample=True,
                                          top_k=1,  # 50
                                          top_p=0.95,
                                          num_return_sequences=1,
                                          eos_token_id=tokenizer_vocab.eos_token_id,
                                          pad_token_id=tokenizer_vocab.pad_token_id,
                                          early_stopping=True,  # Optional parameter to stop searching early for efficiency
                                          )

    input_length = len(feature_dict['input_ids'][0])
    some_generated_token = generated_tokens[0]
    some_generated_token = some_generated_token[input_length:]
    predicted_time = tokenizer_vocab.decode(some_generated_token, skip_special_tokens=True)

    predicted_time = int(predicted_time) / 100

    return predicted_time
