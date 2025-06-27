from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from typing import List, Optional
import time
import json
from threading import Thread
import argparse

#http://localhost:8001/v1/chat/completions
parser = argparse.ArgumentParser(description="Run Model")
parser.add_argument("--model_path", type=str, default="./output/deepseek-7b-finetune",
                    help="Path to the model directory downloaded locally")
parser.add_argument("--port", type=int, default=8001,
                    help="Port of the service")
parser.add_argument("--prompt", type=str, default="./train_data/prompt.md",
                    help="prompt text file, ignore the system content if existed")                    
parser.add_argument("--max_seq_length", type=int, default=512,
                    help="Maximum sequence length for the input")
args = parser.parse_args()
args.prompt_txt = ""
if args.prompt != "":
    with open(args.prompt, 'r', encoding='utf-8') as file:
        args.prompt_txt = file.read()


model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
).eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_path)


class ChatMessage(BaseModel):
    role: str
    content: object
class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9    
    stream: bool = False
    max_tokens: Optional[int] = args.max_seq_length


app = FastAPI()
@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    prompt = ""
    if args.prompt_txt != "":
        prompt = f"system: {args.prompt_txt}"
    for msg in request.messages:
        if msg.role == "system" and args.prompt_txt != "": continue
        if prompt != "": prompt += "\n "
        if type(msg.content) != str:  msg.content = "success" if "success" in str(msg.content) else "failed"
        prompt += f"{msg.role}: {msg.content}"
    prompt += "\n Assistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if len(inputs) > 4096: inputs = inputs[-4096:]
    if request.stream:
        def generate_stream():
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            chat_id = int(time.time())
            for answer in streamer:
                if answer == "": continue
                answer = answer.replace("<|endoftext|>", "").replace("<|end|>", "").replace("<｜end▁of▁sentence｜>", "").strip()
                reponse_frag = json.dumps({
                    "id": "chatcmpl-{chat_id}",
                    "object": "chat.completion.chunk",
                    "created": chat_id,
                    "model": request.model,
                    "system_fingerprint": chat_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": answer
                            },
                            "logprobs": None,
                            "finish_reason": None
                        }
                    ]
                })
                yield f"data: {reponse_frag}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        outputs = model.generate(**inputs, max_new_tokens=request.max_tokens)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        reponse = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": answer.split("Assistant:")[-1].strip()
                }
            }]
        }
        return reponse

@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    print("--------->\n*************************\n", body.decode('utf-8'), "\n*************************")
    response = await call_next(request)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)