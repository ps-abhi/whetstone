from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "Qwen/Qwen3-4B-Instruct-2507"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loader(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return (model, tokenizer)

def generate(prompts, model, tokenizer, batch_size, max_new_tokens, do_sample=False, verbose=False):
    results = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i+batch_size]
        chunk_messages = [[{"role":"user", "content":p}] for p in chunk] # looping over each of the prompts
        inputs = tokenizer.apply_chat_template(chunk_messages, add_generation_prompt=True, tokenize = True, padding=True, return_tensors="pt", return_dict=True) # return_dict ensures that we get the attention mask as well
        inputs = inputs.to(device)
        model = model.to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
            gen = out[:, inputs["input_ids"].shape[1]:]
            for row in gen.tolist():
                results.append([token for token in row if token != tokenizer.pad_token_id])
            if verbose:
                text = tokenizer.batch_decode(gen, skip_special_tokens=True)
                print(text)
           
    return results

prompts = ["What is 2+2?", "Explain gravity in one sentence.", "Name three primes."]
