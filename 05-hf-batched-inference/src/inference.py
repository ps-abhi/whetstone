import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

MODEL = "Qwen/Qwen3-4B-Instruct-2507"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_prompts(id, num_samples, config, split, field):
    dataset = load_dataset(id, config, split=split) # config name "main" is required 
    subset = dataset.select(range(num_samples))
    question_field = subset[field]
    return question_field

def loader(model_name, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return (model, tokenizer)

def generate(prompts, model, tokenizer, batch_size, max_new_tokens, temperature=None, top_p=None, do_sample=False, verbose=False):
    results = []
    gen_kwargs = {"max_new_tokens" :max_new_tokens, "do_sample":do_sample}
    if do_sample:
        gen_kwargs["temperature"]=temperature
        gen_kwargs["top_p"]=top_p
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i+batch_size]
        chunk_messages = [[{"role":"user", "content":p}] for p in chunk] # looping over each of the prompts
        inputs = tokenizer.apply_chat_template(chunk_messages, add_generation_prompt=True, tokenize = True, padding=True, return_tensors="pt", return_dict=True) # return_dict ensures that we get the attention mask as well
        inputs = inputs.to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
            gen = out[:, inputs["input_ids"].shape[1]:]
            for row in gen.tolist():
                results.append([token for token in row if token != tokenizer.pad_token_id])
            if verbose:
                text = tokenizer.batch_decode(gen, skip_special_tokens=True)
                print(text)
           
    return results

def run_sweep(prompts, model, tokenizer, batch_sizes, max_new_tokens, do_sample=False, temperature=None, top_p=None):
    generate(prompts=prompts[:8], model=model, tokenizer=tokenizer, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None)
    results_sweep = []
    for b in batch_sizes:
        try: 
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            results = generate(prompts, model, tokenizer, batch_size=b, max_new_tokens=max_new_tokens,  do_sample=do_sample, temperature=temperature, top_p=top_p, verbose=False)
            torch.cuda.synchronize()
            wall_time = time.perf_counter()-t0

            n_tokens = sum(len(r) for r in results)
            tokens_per_sec  = n_tokens/wall_time
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            results_sweep.append({"batch_size": b, "tok_per_sec": tokens_per_sec, "peak_gb": peak_gb, "oom": False})
        except torch.cuda.OutOfMemoryError:
            results_sweep.append({"batch_size": b, "tok_per_sec": None, "peak_gb": None, "oom": True})
            torch.cuda.empty_cache()
    for row in results_sweep:
        print(row)
    return results_sweep

