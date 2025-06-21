import argparse
from dotenv import load_dotenv
import os
import requests
import time
import json
from typing import List
from dataclasses import dataclass
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


@dataclass
class Round:
    user:str
    assistant:str
@dataclass
class Conversation:
    conversation_id:int
    conversation:List[Round]
def dataclass_to_dict(instance):
    if hasattr(instance, '__dataclass_fields__'):
        return {
            field.name: dataclass_to_dict(getattr(instance, field.name))
            for field in instance.__dataclass_fields__.values()
        }
    elif isinstance(instance, list):
        return [dataclass_to_dict(item) for item in instance]
    return instance


def init_args():
    parser = argparse.ArgumentParser(description="Prepare the train data")

    parser.add_argument("--prompt", type=str, default="./train_data/prompt.txt",
                        help="prompt text file")
    parser.add_argument("--question", type=str, default="./train_data/question.txt",
                        help="question file in lines")
    parser.add_argument("--topic", type=str, default="",
                        help="topic conversation exported from Cherry Studio")
    parser.add_argument("--auth_key", type=str, default="",
                        help="auth key of deekseek, also can be set in .env by DEEPSEEK_AUTH_KEY=**********")
    # parser.add_argument("--output", type=str, required=True,
    #                     help="train data in jsonl format")
    parser.add_argument("--output", type=str, default="./train_data/train_data.jsonl",
                        help="train data in jsonl format")


    args = parser.parse_args()

    args.prompt_txt = ""
    if args.prompt != "":
        with open(args.prompt, 'r', encoding='utf-8') as file:
            args.prompt_txt = file.read()

    load_dotenv()
    if args.auth_key == "":
        args.auth_key = os.getenv("DEEPSEEK_AUTH_KEY", "")

    return args


def ask_deepseek(args):
    if args.question == "": 
        return None

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.auth_key}"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [],
        "temperature": 0.3,
        "stream": False
    }

    if args.prompt_txt != "":
        data["messages"].append({
            "role": "system",
            "content": args.prompt_txt
        })

    rounds = []
    def request_answer(ask):
        data["messages"].append({
            "role": "user",
            "content": ask
        })
        response = requests.post(url, headers=headers, json=data)

        answer = "failed to get answer!!!"
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
        print("user>>>>>>>>>>>>>>>>>>>>>>\n", ask)
        print("assistant>>>>>>>>>>>>>>>>>\n", answer)

        data["messages"].append({
            "role": "assistant",
            "content": answer
        })
        rounds.append(Round(user=ask, assistant=answer))
        if "TYPE: MCP" in answer:
            request_answer("success")

    
    with open(args.question, 'r', encoding='utf-8') as file: 
        for line in file:
            ask = line.strip()
            if ask == '': continue

            request_answer(ask)
            time.sleep(1)
                
        
    conversation = Conversation(conversation_id=int(time.time()), conversation=rounds)
    return conversation

def process_topic(args):
    if args.topic == "": 
        return None

    with open(args.topic, 'r', encoding='utf-8') as file: 
        contents, role, content = [], '', ''
        for line in file:
            line = line.strip()
            if line == '' or line == '---': continue
            if line.startswith('###'):
                if role != '' and content != '': contents.append([role, content])

                words = line.split(' ')
                role = words[-1]
                content = ''
            else:
                if content != '': content += '\\n'
                content += line
        if role != '' and content != '': contents.append([role, content])

    
    num_contents, rounds = len(contents) // 2 * 2, [] 
    for i in range(num_contents//2):
        role0, content0, role1, content1 = contents[2*i][0], contents[2*i][1], contents[2*i+1][0], contents[2*i+1][1]
        if role0 == 'User' and role1 == 'Assistant':
            rounds.append(Round(user=content0, assistant=content1))
    
    conversation = Conversation(conversation_id=int(time.time()), conversation=rounds)
    return conversation


def append_conversation(args, conversation):
    with open(args.output, 'a', encoding='utf-8') as file:
        file.write(json.dumps(dataclass_to_dict(conversation)))
        file.write('\n')


if __name__ == "__main__":
    args = init_args()
    conversation = ask_deepseek(args)
    if conversation != None:
        append_conversation(args, conversation)
    conversation = process_topic(args)
    if conversation != None:
        append_conversation(args, conversation)
