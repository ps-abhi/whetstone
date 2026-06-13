from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PrefixTuningConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocessing(instance):
    return {
        "messages" : [
            {"role" : "user", "content" : instance["prompt"]},
            {"role" : "assistant", "content" : instance["completion"]}
        ]
    }

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    dataset = load_dataset(cfg.dataset_name)
    dataset = dataset.map(preprocessing, remove_columns=["prompt", "completion"])
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    quant = cfg.peft.get("quantization")
    if quant:
        quant = OmegaConf.to_container(cfg.peft.quantization, resolve=True)
        bnb_config = BitsAndBytesConfig(**quant)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch.bfloat16, quantization_config=bnb_config)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch.bfloat16)

    peft_method = cfg.peft.name
    peft_kwargs = OmegaConf.to_container(cfg.peft, resolve=True)
    peft_kwargs.pop("name")
    peft_kwargs.pop("quantization", None)

    if peft_method == "prefix":
        peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", **peft_kwargs)
    
    else:
        peft_config = LoraConfig(task_type = "CAUSAL_LM", **peft_kwargs)
        
    model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()

    sft_kwargs = OmegaConf.to_container(cfg.train, resolve=True)

    args = SFTConfig(
        seed = cfg.seed, 
        report_to="wandb"
        **sft_kwargs
    )

    trainer = SFTTrainer(
        model = model,
        train_dataset = train_dataset,
        args = args,
        processing_class = tokenizer,
    )

    trainer.train()


if __name__=="__main__":
    main()

