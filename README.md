pip install --upgrade pip
pip install torch transformers accelerate peft bitsandbytes loguru
pip install swanlab[dashboard]

https://github.com/CherryHQ/cherry-studio

git lfs install
cd ./output
git clone https://oauth2:1EDrxJLfQrAcLsadR8yT@www.modelscope.cn/JiangDunchun/deepseek-7b-finetune.git

python train_data.py --prompt=./train_data/prompt.txt --question=./train_data/question.txt --topic=./train_data/vup_mcp.md --output=./train_data/train_data.jsonl

torchrun finetune.py --model_name_or_path=../.cache/modelscope/models/deepseek-ai/deepseek-llm-7b-chat/ --train_data=./train_data/train_data.jsonl

python merge_lora.py --pretrained=./.cache/modelscope/models/deepseek-ai/deepseek-llm-7b-chat --finetuned=./output --merged=./output/deepseek-7b-finetune

python run_model.py --model_path=./output/deepseek-7b-finetune
