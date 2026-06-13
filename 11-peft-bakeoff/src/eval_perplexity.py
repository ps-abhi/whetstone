from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch 
import math 

def compute_perplexity(model, dataset, tokenizer, device):

    model.eval()
    total_loss = 0.0
    total_tokens = 0.0

    with torch.no_grad():
        for ex in dataset:
            messages = [{"role":"user", "content":ex['prompt']}]
            tokenized_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            prompt_len = len(tokenized_prompt['input_ids'])

            # e.g 10

            completed_message = messages + [{"role":"assistant", "content":ex["completion"]}]
            tokenized_ex = tokenizer.apply_chat_template(completed_message, return_tensors="pt")
            input_ids = tokenized_ex['input_ids']

            labels_ex = input_ids.clone()

            for i in range(prompt_len):
                labels_ex[0][i] = -100
            
            input_ids = input_ids.to(device)
            labels_ex = labels_ex.to(device)
            output = model(input_ids = input_ids, labels = labels_ex)

            n_tokens = (labels_ex != -100).sum().item()
            output_sum = output.loss.item() * n_tokens # sum = count*mean

            total_loss += output_sum
            total_tokens += n_tokens
        return math.exp(total_loss/total_tokens)

# Verified earlier on the 0.6B stand-in (50 examples): perplexity ≈ 3.40

