import os
import argparse
import pprint
import json
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, DataCollatorWithPadding, DataCollatorForSeq2Seq, BitsAndBytesConfig

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallel processing
os.environ["WANDB_DISABLED"] = "true"

def get_hyperparameters():
    parser = argparse.ArgumentParser(description="Training a model for code generation")
    parser.add_argument('--model', default="DeepSeek_coder_7b_instruct_v1_5", type=str, help='Type of transformers model as model backbone')
    parser.add_argument('--model_path', default=r'DeepSeek_coder_7b_instruct_v1_5', type=str, help='Path to model backbone pretrained weights')
    parser.add_argument('--use_Lora_plugin', default=True, type=bool, help='Whether to use the Lora plugin model')
    parser.add_argument('--Lora_plugin_paths', default=['Model_DeepSeek_Coder_Pre_time_RandDel/checkpoint-9044',
                                                       'Model_DeepSeek_Coder_Pre_time_Loop&Rec/checkpoint-9306',
                                                       'Model_DeepSeek_Coder_Pre_time_Ori/checkpoint-8188',
                                                       'Model_DeepSeek_Coder_Pre_time_Uni/checkpoint-7906',], type=str, help='Paths for Lora model plugins')
    parser.add_argument('--Lora_model_path', default=r'Finally_Model_DeepSeek_Coder_Pre_time', type=str, help='Path to the overall Lora model')
    parser.add_argument('--save_dir', default='DeepSeek—test_generate_Time_4model_Uni', type=str, help='Path to save trained model checkpoints')
    parser.add_argument('--test_path', default="CodeExecTimeDB_Uni/test.csv", type=str, help='Path to test data')
    parser.add_argument('--debug', default=False, action='store_true', help='Set to turn on debug mode, i.e., using a small data split and only 1 data worker')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    return args

def main(args):
    args_dict = vars(args)
    print(pprint.pformat(args_dict))
    os.makedirs(args.save_dir, exist_ok=True)
    # Save args dictionary
    json.dump(args_dict, open(os.path.join(args.save_dir, "args.json"), 'w'))
    full_dataset = get_dataset(args)
    # Load
    run_generate(args, full_dataset)

def get_dataset(args):
    # Load dataset
    full_dataframe = pd.read_csv(args.test_path)
    full_dataframe = convert_decimal_to_integer(full_dataframe, 'Time')
    full_dataframe['Time'] = full_dataframe['Time'].astype(str)
    if args.debug:
        full_dataframe = full_dataframe[:10]
    # Convert pandas DataFrame to datasets.Dataset
    test_data = Dataset.from_pandas(full_dataframe)
    # Combine datasets.Dataset objects into a DatasetDict
    full_dataset = DatasetDict({
        "test": test_data,
    })
    return full_dataset

def convert_decimal_to_integer(df, column_name):
    if column_name in df.columns:
        # Multiply the specified column values by 100 and convert to integer
        df[column_name] = df[column_name].apply(lambda x: int(x * 100))
        return df
    else:
        print(f"Column '{column_name}' not found in DataFrame, incorrect column name.")
        return df

q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

def run_generate(args, full_dataset):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model_path = args.model_path if args.model_path is not None else '{}'.format(args.model)
    print("Loading model from {}...".format(model_path))
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 quantization_config=q_config,
                                                 torch_dtype=torch.bfloat16,
                                                 pad_token_id=tokenizer.pad_token_id,
                                                 trust_remote_code=True
                                                 )
    if args.use_Lora_plugin:
        for path in args.Lora_plugin_paths:
            model = PeftModel.from_pretrained(model, path)

    print(f'Finished loading model, {args.use_Lora_plugin}, {args.Lora_plugin_paths}')
    print(f"Starting main loop...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    result_dict = {'prediction_list': [], 'answer_list': [], 'error_list': []}
    for i, sample in tqdm(enumerate(full_dataset['test'])):
        feature_dict = tokenizer(sample['Prompt_input'], return_tensors="pt")
        with torch.no_grad():
            generated_token_list = model.generate(feature_dict['input_ids'].to(device),
                                                  attention_mask=feature_dict["attention_mask"].to(device),
                                                  max_new_tokens=4,
                                                  do_sample=True,
                                                  top_k=1,
                                                  top_p=0.95,
                                                  num_return_sequences=1,
                                                  eos_token_id=tokenizer.eos_token_id,
                                                  pad_token_id=tokenizer.pad_token_id,
                                                  early_stopping=True
                                                 )
        input_length = len(feature_dict['input_ids'][0])
        for code_index, generated_token in enumerate(generated_token_list):
            generated_token = generated_token[input_length:]
            prediction = tokenizer.decode(generated_token, skip_special_tokens=True)
            result_dict['prediction_list'].append(prediction)
            result_dict['answer_list'].append(sample['Time'])

    assert len(result_dict['prediction_list']) == len(result_dict['answer_list'])
    for i, predicted_value in enumerate(result_dict['prediction_list']):
        predicted_value = float(predicted_value)/100
        error_value = abs(predicted_value - (float(result_dict['answer_list'][i])/100))
        result_dict['error_list'].append(error_value)
    result_dict["final_error_value"] = mean(result_dict['error_list'])
    print(f"Final error value: {result_dict['final_error_value']}")
    with open(f"{args.save_dir}/predictions_dict_error_{result_dict['final_error_value']}.txt", 'w', encoding='UTF-8') as f:
        f.write(str(result_dict))

if __name__ == "__main__":
    args = get_hyperparameters()
    main(args)
