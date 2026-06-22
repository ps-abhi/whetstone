from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "Qwen/Qwen3-4B-Instruct-2507"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prompts = ["What is 2+2?", "Explain gravity in one sentence.", "Name three primes."]

batch_messages = [[{"role":"user", "content":p}] for p in prompts] # looping over each of the prompts

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

inputs = tokenizer.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize = True, padding=True, return_tensors="pt", return_dict=True) # return_dict ensures that we get the attention mask as well

inputs = inputs.to(device)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.to(device)


with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    text = tokenizer.batch_decode(gen, skip_special_tokens=True)

print(text)