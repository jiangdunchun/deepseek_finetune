import argparse
import time
import json
from typing import List
from dataclasses import dataclass


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process train data from Cherry Studio chat history")
    # parser.add_argument("--conversation", type=str, required=True,
    #                     help="conversation data in md format")
    # parser.add_argument("--train_data", type=str, required=True,
    #                     help="train data in jsonl format")
    parser.add_argument("--conversation", type=str, default="./train_data/cherry_studio.md",
                        help="conversation data in md format")
    parser.add_argument("--train_data", type=str, default="./train_data/train_data.jsonl",
                        help="train data in jsonl format")
    args = parser.parse_args()

    contents = []
    with open(args.conversation, 'r', encoding='utf-8') as file: 
        role, content = '', ''
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

    rounds = []
    num_contents = len(contents) // 2 * 2
    train_data_str = ""
    for i in range(num_contents//2):
        role0, content0, role1, content1 = contents[2*i][0], contents[2*i][1], contents[2*i+1][0], contents[2*i+1][1]
        if role0 == 'User' and role1 == 'Assistant':
            rounds.append(Round(user=content0, assistant=content1))

    conversation = Conversation(conversation_id=int(time.time()), conversation=rounds)

    with open(args.train_data, 'a', encoding='utf-8') as f:
        f.write(json.dumps(dataclass_to_dict(conversation)))
        f.write('\n')





