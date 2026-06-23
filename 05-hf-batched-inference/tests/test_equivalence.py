from inference import loader, generate, MODEL

def test_equivalence():
    model, tokenizer = loader(MODEL)
    prompts = ["What is 2+2?", "Explain gravity in one sentence.", "Name three primes."]

    reference = generate(prompts, model, tokenizer, batch_size=1,max_new_tokens=64)
    batched   = generate(prompts, model, tokenizer, batch_size=len(prompts), max_new_tokens=64)

    assert reference==batched