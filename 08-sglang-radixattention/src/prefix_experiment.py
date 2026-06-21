from bench import flush_cache, benchmark, get_cache_hit_rate, BASE_URL
from backend import prompt
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct")
prefix_modified = "Now" + prompt

flush_cache(BASE_URL)
benchmark(prompt=prompt, max_tokens=512, n=8, base_url=BASE_URL, temperature=0.8, tokenizer=tokenizer, warmup=0, k=1)
original_cache_run = benchmark(prompt=prompt, max_tokens=512, n=8, base_url=BASE_URL, temperature=0.8, tokenizer=tokenizer, warmup=0, k=1)

flush_cache(BASE_URL)
benchmark(prompt=prompt, max_tokens=512, n=8, base_url=BASE_URL, temperature=0.8, tokenizer=tokenizer, warmup=0, k=1)
mutated_cache_run = benchmark(prompt=prefix_modified, max_tokens=512, n=8, base_url=BASE_URL, temperature=0.8, tokenizer=tokenizer, warmup=0, k=1)

print(f"Original: {original_cache_run}")
print(f"Broken: {mutated_cache_run}")