from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K")

def preprocessing(instance):
    return {
        "messages" : [
            {"role" : "user", "content" : instance["prompt"]},
            {"role" : "assistant", "content" : instance["completion"]}
        ]
    }

dataset = dataset.map(preprocessing, remove_columns=["prompt", "completion"])
train_dataset = dataset["train"]
test_dataset = dataset["test"]

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16)

lora_config = LoraConfig(
    r = 16,
    lora_alpha= 32,
    lora_dropout = 0.05,
    target_modules="all-linear",    
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()







args = SFTConfig(
        learning_rate=2e-4,
        bf16=True,
        assistant_only_loss=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=1000,
        seed=42,
        max_length=1024,
        output_dir="./lora_train",
        lr_scheduler_type="cosine"
    )


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args = args,
    processing_class=tokenizer,
)

trainer.train()