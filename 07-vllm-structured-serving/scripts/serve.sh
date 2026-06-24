#!/usr/bin/env bash
PORT=5000

MODEL="Qwen/Qwen3-4B-Instruct-2507"

uv run vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --seed 0 \
    "$@"