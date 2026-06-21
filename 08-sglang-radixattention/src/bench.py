import statistics
import time
import sglang as sgl
import requests
import re
from backend import sglang_run, prompt, vllm_run
from transformers import AutoTokenizer
import hydra
from omegaconf import OmegaConf
import wandb
from dotenv import load_dotenv
import os

load_dotenv()


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

@hydra.main(version_base=None, config_path = "../configs", config_name="config")    
def main(cfg):
    RUNNERS = {"sglang": sglang_run, "vllm": vllm_run}
    if cfg.engine.set_backend:
        sgl.set_default_backend(sgl.RuntimeEndpoint(cfg.engine.base_url))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    runner = wandb.init(project = cfg.wandb_project, name=f"{cfg.engine.name}-cache-{cfg.cache}",
                        config=OmegaConf.to_container(cfg, resolve=True))
    flush_cache(cfg.engine.base_url)
    run_fn = RUNNERS[cfg.engine.runner]
    results = benchmark(run_fn, prompt=prompt, n=cfg.n, max_tokens=cfg.max_tokens, temperature=cfg.temperature, base_url=cfg.engine.base_url, tokenizer=tokenizer, k=cfg.k, warmup=cfg.warmup)
    wandb.log(results)
    runner.finish()
    print(results)

if __name__ == "__main__":
    main()