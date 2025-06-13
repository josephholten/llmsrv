import sys
import time
from llama_cpp import Llama

MODEL_PATH = "/home/joseph/models/Llama-3.2-8B-Instruct.Q8_0.gguf"
N_CTX = 8192
N_BATCH = 512

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=N_CTX,
    n_batch=N_BATCH,
    verbose=False
)

class Request(BaseModel):
    prompt: str = "You are a text summarization assistant. Your task is to provide a concise summary of the following input text. If the input is not a coherent, meaningful text, or appears to be a random sequence of characters, binary data, or an image, you MUST respond ONLY with the exact phrase: 'Error: Invalid input for summarization.' Input text: %%%INPUT%%% Output text:"
    input_token: str = "%%%INPUT%%%"
    input_text: str

    max_tokens: int = 100
    stop_token: str = "<|EOF|>"

@app.get("/")
async def index():
    return {"message": "llm server up"}

@app.post("/call")
async def call(request: Request):
    short_input_text = request.input_text[:N_CTX*3] # heuristic to go from chars to tokens
    prompt = request.prompt.replace(request.input_token, short_input_text)

    start = time.time()
    output = llm(prompt, max_tokens=100, stop=["<|EOF|>"])
    duration = time.time() - start
    return {"output": output, "time": duration}

