from inference import generate, MODEL, load_prompts
from transformers import set_seed
SEED=42

def test_sampling(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompts = load_prompts()
    set_seed(SEED)
    run1 = generate(prompts[:8], model, tokenizer, batch_size=8, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9)
    set_seed(SEED)
    run2 = generate(prompts[:8], model, tokenizer, batch_size=8, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9)

    assert run1==run2