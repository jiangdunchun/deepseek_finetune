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
parser.add_argument("--model_path", type=str, required=True,
                    help="Path to the model directory downloaded locally")
parser.add_argument("--port", type=int, default=8001,
                    help="Port of the service")
parser.add_argument("--max_seq_length", type=int, default=2048,
                    help="Maximum sequence length for the input")
args = parser.parse_args()


model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
).eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_path)


class ChatMessage(BaseModel):
    role: str
    content: str
class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = args.max_seq_length


app = FastAPI()
@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    prompt += "\n Assistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if len(inputs) > args.max_seq_length: inputs = inputs[-1*args.max_seq_length:]
    if request.stream:
        def generate_stream():
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=request.max_tokens,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            chat_id = int(time.time())
            for answer in streamer:
                if answer == "": continue
                #answer = answer.replace("<|endoftext|>", "").replace("<|end|>", "").strip()
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