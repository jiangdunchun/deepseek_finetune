from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
import os
from transformers import AutoTokenizer
import shutil
import argparse

parser = argparse.ArgumentParser(description="Merge Model")
parser.add_argument("--pretrained", type=str, required=True,
                    help="Path to the pretrained model directory")
parser.add_argument("--finetuned", type=str, required=True,
                    help="Path to the lora out directory")
parser.add_argument("--merged", type=str, required=True,
                    help="Path to the merged model directory")
args = parser.parse_args()

#Copies files(except for the weight files) from directory A to directory B if they exist in A but not in B.
def copy_files_not_in_B(A_path, B_path):
    if not os.path.exists(A_path):
        raise FileNotFoundError(f"The directory {A_path} does not exist.")
    if not os.path.exists(B_path):
        os.makedirs(B_path)

    files_in_A = os.listdir(A_path)
    files_in_A = set([file for file in files_in_A if not (".bin" in file or "safetensors" in file)])

    files_in_B = set(os.listdir(B_path))

    files_to_copy = files_in_A - files_in_B

    for file in files_to_copy:
        src_path = os.path.join(A_path, file)
        dst_path = os.path.join(B_path, file)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

def merge_lora_to_base_model():
    model_path = args.pretrained
    adapter_path = args.finetuned
    save_path = args.merged

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(model, adapter_path, device_map="auto",trust_remote_code=True)
    merged_model = model.merge_and_unload()
    tokenizer.save_pretrained(save_path)
    merged_model.save_pretrained(save_path, safe_serialization=False)

    copy_files_not_in_B(model_path, save_path)

    print("****************************Merged success!!!****************************")

if __name__ == '__main__':
    merge_lora_to_base_model()
