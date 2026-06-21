python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B-Instruct \
    --port 30000 \
    --enable-metrics "$@" # $@ allows other flags to be passed through
