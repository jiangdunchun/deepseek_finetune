import argparse
from dotenv import load_dotenv
import os
import requests
import time
import json
from proccess_cherry_studio import Round, Conversation, dataclass_to_dict

load_dotenv()
auth_key = os.environ["DEEPSEEK_AUTH_KEY"]


parser = argparse.ArgumentParser(description="Ask Deepseek")
# parser.add_argument("--question", type=str, required=True,
#                     help="question data in lines")
# parser.add_argument("--prompt", type=str, required=True,
#                     help="question data in lines")
# parser.add_argument("--train_data", type=str, required=True,
#                     help="train data in jsonl format")
parser.add_argument("--question", type=str, default="./train_data/question.txt",
                    help="question data in lines")
parser.add_argument("--prompt", type=str, default="./train_data/prompt.txt",
                    help="prompt txt")
parser.add_argument("--train_data", type=str, default="./train_data/train_data.jsonl",
                    help="train data in jsonl format")
args = parser.parse_args()


url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {auth_key}"
}


prompt = ""
with open(args.prompt, 'r', encoding='utf-8') as file:
    prompt = file.read()
data = {
    "model": "deepseek-chat",
    "messages": [
        {
            "role": "system",
            "content": prompt
        }
    ],
    "stream": False
}


rounds = []
with open(args.question, 'r', encoding='utf-8') as file: 
    role, content = '', ''
    for line in file:
        ask = line.strip()
        if ask == '' or ask == '---': continue

        data["messages"].append({
            "role": "user",
            "content": ask
        })

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            data["messages"].append({
                "role": "assistant",
                "content": ask
            })
            print("user>>>>>>>>>>>>>>>>>>>>>>\n", ask)
            print("assistant>>>>>>>>>>>>>>>>>\n", answer)
            rounds.append(Round(user=ask, assistant=answer))

conversation = Conversation(conversation_id=int(time.time()), conversation=rounds)

with open(args.train_data, 'a', encoding='utf-8') as f:
    f.write(json.dumps(dataclass_to_dict(conversation)))
    f.write('\n')
