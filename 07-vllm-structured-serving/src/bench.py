from concurrent.futures import ThreadPoolExecutor
import statistics
from client import BASE_URL, SYSTEM_PROMPT, kwargs, MODEL
from openai import OpenAI
from typing import NamedTuple
from transformers import AutoTokenizer
import time
import httpx
import re
import os
import hydra
from omegaconf import OmegaConf
import wandb
import random
from datasets import load_dataset
from dotenv import load_dotenv

client = OpenAI(base_url=BASE_URL, api_key="EMPTY")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
FILLER_IDS = tokenizer.encode("The following is reference context for the extraction task. " * 1000)

class Req(NamedTuple):
    system : str
    user : str

def one_request(system_prompt, user_text):
    kw = kwargs(system_prompt, user_text)
    resp = client.chat.completions.create(**kw)
    return resp.choices[0].message.content

def vllm_run(requests, concurrency):
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
       return list(ex.map(lambda r: one_request(r.system, r.user), requests))
    
def build_prefix(n_tokens, uid = None):
    body = tokenizer.decode(FILLER_IDS[:n_tokens])
    prefix = f"{SYSTEM_PROMPT}\n {body}"
    if uid is not None:
        prefix = f"[req-{uid}]{prefix}"
    return prefix

def shared_workload(texts, n_tokens):
    p = build_prefix(n_tokens)
    return [Req(p,t) for t in texts]

def unique_workload(texts, n_tokens):
    return [Req(build_prefix(n_tokens, uid=i), t ) for i, t in enumerate(texts)]


def count_tokens(answers, tokenizer):
    return sum(len(tokenizer.encode(a)) for a in answers)

def read_counter(text, name):
    m = re.search(rf"{re.escape(name)}(?:_total)?(?:\{{[^}}]*\}})?\s+([0-9.eE+]+)", text)
    return float(m.group(1) if m else 0.0)

def get_cache_stats(base_url):
    text = httpx.get(f"{base_url}/metrics").text
    return (read_counter(text, "vllm:prefix_cache_hits"),
            read_counter(text, "vllm:prefix_cache_queries"))

def benchmark(run_fn, requests, concurrency, base_url, tokenizer, k=10, warmup=3):
    for _ in range(warmup):
        run_fn(requests, concurrency)
    h0, q0 = get_cache_stats(base_url)
    throughputs = []
    for _ in range(k):
        t0 = time.perf_counter()
        answers  = run_fn(requests, concurrency)
        elapsed = time.perf_counter() - t0
        throughputs.append(count_tokens(answers, tokenizer)/elapsed)
    h1, q1 = get_cache_stats(base_url)
    difference = q1-q0 
    return {
         "throughput_tokens_per_sec": statistics.median(throughputs),
         "cache_hit_rate": (h1 - h0) / difference if difference > 0 else 0.0
    }


@hydra.main(version_base=None, config_path= "../configs", config_name="config")
def main(cfg):
    random.seed(cfg.seed)
    load_dotenv()
    ds = load_dataset("paraloq/json_data_extraction", split="train")
    texts = [r["text"] for r in ds if r["topic"] == "medical"][:cfg.workload.n_texts]

    builder = shared_workload if cfg.workload.mode == "shared" else unique_workload
    reqs = builder(texts, cfg.workload.prefix_length)


    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run = wandb.init(
        project = cfg.wandb_project,
        name = f"apc-{cfg.apc}-{cfg.workload.mode}-plen{cfg.workload.prefix_length}",
        config = OmegaConf.to_container(cfg, resolve=True)
    )

    results = benchmark(vllm_run, reqs, cfg.workload.concurrency, cfg.base_url, tokenizer, k=cfg.k, warmup=cfg.warmup)
    run.log(results)
    print(results)
    run.finish()

if __name__ == "__main__":
    main()
    