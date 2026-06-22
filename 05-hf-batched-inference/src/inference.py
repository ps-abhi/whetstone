from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "Qwen/Qwen3-4B-Instruct-2507"  

prompt = ""

messages = [{"role":"user", "content":prompt}]

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")

tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.eval()
