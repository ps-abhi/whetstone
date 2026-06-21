import statistics
import time
import sglang as sgl
import requests
import re
from backend import sglang_run, prompt, MODEL
from transformers import AutoTokenizer


BASE_URL = "http://localhost:30000"

def count_tokens(answers, tokenizer) -> int:
    return sum(len(tokenizer.encode(a)) for a in answers) # answers is a list of strs

def get_cache_hit_rate(base_url) -> float:
    text = requests.get(f"{base_url}/metrics").text
    match = re.search(r"cache_hit_rate\S*\s+([0-9.]+)", text)
    if match:
        return float(match.group(1))
    return None 

def flush_cache(base_url) -> None:
    requests.post(f"{base_url}/flush_cache")

def benchmark(run_fn, prompt, n, max_tokens, temperature,  base_url, tokenizer, k=10, warmup=3):
    for _ in range(warmup):
        run_fn(prompt, n, max_tokens, temperature)
        
    throughputs = []
    for _ in range(k):
        t0 = time.perf_counter()  # performance counter
        answers = run_fn(prompt, n, max_tokens, temperature)

        elapsed = time.perf_counter()-t0 
        generated = count_tokens(answers, tokenizer)
        throughputs.append(generated/elapsed)
        
    return {
        "throughput_tokens_per_sec" : statistics.median(throughputs),
        "cache_hit_rate" : get_cache_hit_rate(base_url)
    }
    
def main():
    sgl.set_default_backend(sgl.RuntimeEndpoint(BASE_URL))
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    flush_cache(BASE_URL)
    results = benchmark(sglang_run, prompt=prompt, n=8, max_tokens=512, temperature=0.8, base_url=BASE_URL, tokenizer=tokenizer)
    print(results)

if __name__ == "__main__":
    main()