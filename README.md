pip install --upgrade pip
pip install torch transformers accelerate peft bitsandbytes loguru
pip install swanlab[dashboard]

https://github.com/CherryHQ/cherry-studio

python proccess_data.py --conversation=./train_data/conversation1.md --train_data=./train_data/train_data.jsonl

torchrun finetune.py --model_name_or_path=./.cache/modelscope/models/deepseek-ai/deepseek-llm-7b-chat/ --train_data=./train_data/train_data.jsonl

git lfs install
cd ./output
git clone https://oauth2:1EDrxJLfQrAcLsadR8yT@www.modelscope.cn/JiangDunchun/deepseek-7b-finetune.git

python merge_lora.py --pretrained=./.cache/modelscope/models/deepseek-ai/deepseek-llm-7b-chat --finetuned=./output --merged=./output/deepseek-7b-finetune

python run_model.py --model_path=./output/deepseek-7b-finetune
