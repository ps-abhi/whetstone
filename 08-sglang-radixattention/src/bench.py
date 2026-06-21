import statistics
import time
import sglang as sgl
import requests
import re
from backend import self_consistency
from transformers import AutoTokenizer
from backend import prompt

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

def benchmark(prompt, n, max_tokens, temperature,  base_url, tokenizer, k=10, warmup=3):
    flush_cache(base_url)

    for _ in range(warmup):
        self_consistency.run(prompt=prompt, n=n, max_tokens=max_tokens, temperature=temperature)
        
    throughputs = []
    for _ in range(k):
        t0 = time.perf_counter()  # performance counter
        state = self_consistency.run(prompt=prompt, n=n, max_tokens=max_tokens, temperature=temperature)

        elapsed = time.perf_counter()-t0 
        generated = count_tokens(state["answers"], tokenizer)
        throughputs.append(generated/elapsed)
        
    return {
        "throughput_tokens_per_sec" : statistics.median(throughputs),
        "cache_hit_rate" : get_cache_hit_rate(base_url)
    }
    
def main():
    sgl.set_default_backend(sgl.RuntimeEndpoint(BASE_URL))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct")
    results = benchmark(prompt=prompt, n=8, max_tokens=512, temperature=0.8, base_url=BASE_URL, tokenizer=tokenizer)
    print(results)

if __name__ == "__main__":
    main()