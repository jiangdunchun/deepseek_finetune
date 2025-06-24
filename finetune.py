import argparse
from os.path import join
import pandas as pd
from datasets import Dataset
#from loguru import logger
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
)
import torch
from peft import LoraConfig, get_peft_model, TaskType
#from swanlab.integration.transformers import SwanLabCallback
import bitsandbytes as bnb
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import json

def configuration_parameter():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for deepseek model")

    parser.add_argument("--model_name_or_path", type=str, default="./model",
                        help="Path to the model directory downloaded locally")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save the fine-tuned model and checkpoints")

    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to the training data file in JSONL format")

    parser.add_argument("--num_train_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length for the input")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Number of steps between logging metrics")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Number of steps between saving checkpoints")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Alpha parameter for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout rate for LoRA")


    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)),
                        help="Local rank for distributed training")
    parser.add_argument("--use_cuda", type=bool, default=True,
                        help="Enable distributed training")


    parser.add_argument("--gradient_checkpointing", type=bool, default=True,
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="Optimizer to use during training")
    parser.add_argument("--train_mode", type=str, default="lora",
                        help="lora or qlora")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--fp16", type=bool, default=True,
                        help="Use mixed precision (FP16) training")
    parser.add_argument("--report_to", type=str, default=None,
                        help="Reporting tool for logging (e.g., tensorboard)")
    parser.add_argument("--remove_unused_columns", type=bool, default=True,
                        help="Remove unused columns from the dataset")

    args = parser.parse_args()

    if args.use_cuda:
        args.use_cuda = torch.cuda.is_available()
    
    if args.local_rank != -1:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    
    os.makedirs(args.output_dir, exist_ok=True)

    return args

def setup_distributed(backend="nccl"):
    """Initialize distributed training environment"""
    if "LOCAL_RANK" in os.environ:
        # For torch.distributed.launch or torchrun
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
    else:
        # For single node multi-gpu without distributed launcher
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        print(f"Distributed training initialized. Rank {rank}/{world_size}")
    return local_rank, world_size, rank

def get_device(local_rank, use_cuda=True):
    if use_cuda:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device

def find_all_linear_names(model, train_mode):
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    #logger.info('LoRA target module names: {lora_module_names}')
    return lora_module_names

def load_model(args, train_dataset, data_collator):
    local_rank, world_size, rank = setup_distributed()
    device = get_device(local_rank, args.use_cuda)

    model_kwargs = {
        #"trust_remote_code": True,
        "torch_dtype": torch.float16 if args.fp16 else torch.bfloat16,
        "use_cache": False if args.gradient_checkpointing else True,
        #"device_map": None,
    }
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.enable_input_require_grads()

    # model = model.to(device)
    # if world_size > 1:
    #     model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    target_modules = find_all_linear_names(model.module if isinstance(model, DDP) else model, args.train_mode)

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False

    )
    use_bfloat16 = torch.cuda.is_bf16_supported()

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        seed=args.seed,
        optim=args.optim,
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        fp16=args.fp16,
        bf16=not args.fp16 and use_bfloat16,
        remove_unused_columns=False
    )

    model = get_peft_model(model.module if isinstance(model, DDP) else model, config)
    print("model:", model)
    model.print_trainable_parameters()

    # swanlab_config = {
    #     "lora_rank": args.lora_rank,
    #     "lora_alpha": args.lora_alpha,
    #     "lora_dropout": args.lora_dropout,
    #     "dataset": args.train_data
    # }
    # swanlab_callback = SwanLabCallback(
    #     project="deepseek-finetune",
    #     experiment_name="deepseek-finetune",
    #     description="deepseek-finetune",
    #     workspace=None,
    #     config=swanlab_config,
    # )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        #callbacks=[swanlab_callback],
        callbacks=[]
    )
    return trainer


#{"conversation_id": 1, "conversation": [{"user": "", "assistant": ""}]}
def process_data(data: dict, tokenizer, max_seq_length):
    conversation = data["conversation"]
    pre_text = ""
    if data["prompt"] != "":
        pre_text = "System:" + data["prompt"] + "\n\n"
    samples = []
    for i, conv in enumerate(conversation):
        human_text = conv["user"].strip()
        assistant_text = conv["assistant"].strip()

        input_text = pre_text + "User:" + human_text + "\n\nAssistant:"
        pre_text = input_text + assistant_text + "\n\n"

        #print("input>>>>>>>>>>>>>:", input_text)
        #print("output>>>>>>>>>>>>>:", assistant_text)
        input_tokenizer = tokenizer(
            input_text,
            add_special_tokens=False,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        output_tokenizer = tokenizer(
            assistant_text,
            add_special_tokens=False,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids = (input_tokenizer["input_ids"] + output_tokenizer["input_ids"] + [tokenizer.eos_token_id])
        attention_mask = input_tokenizer["attention_mask"] + output_tokenizer["attention_mask"] + [1]
        labels = ([-100] * len(input_tokenizer["input_ids"]) + output_tokenizer["input_ids"] + [tokenizer.eos_token_id])

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[-1*max_seq_length:]
            attention_mask = attention_mask[-1*max_seq_length]
            labels = labels[-1*max_seq_length]

        samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        })    

    return samples


if __name__ == "__main__":
    args = configuration_parameter()

    print("***************load the tokenizer**********************")
    model_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    print("*****************proccess data*************************")
    data = pd.read_json(args.train_data, lines=True)
    train_ds = Dataset.from_pandas(data)
    processed_data = []
    for line in train_ds:
        processed_data.extend(process_data(line, tokenizer, args.max_seq_length))
    train_dataset = Dataset.from_list(processed_data)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

    print("**************load model and train**********************")
    trainer = load_model(args, train_dataset, data_collator)
    trainer.train()
    final_save_path = join(args.output_dir)
    trainer.save_model(final_save_path)
