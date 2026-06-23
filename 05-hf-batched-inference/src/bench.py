import hydra
from omegaconf import OmegaConf
import wandb
from inference import loader, load_prompts, run_sweep
from dotenv import load_dotenv
import os
from transformers import set_seed

load_dotenv()

@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg):
    set_seed(cfg.seed)
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    runner = wandb.init(project=cfg.wandb.project,
                        config = OmegaConf.to_container(cfg, resolve=True))
    model, tokenizer = loader(cfg.model.id, cfg.model.dtype)
    prompts = load_prompts(cfg.dataset.id, cfg.dataset.num_samples ,cfg.dataset.config, cfg.dataset.split, cfg.dataset.field)
    results = run_sweep(prompts, model, tokenizer, cfg.generate.batch_sizes, cfg.generate.max_new_tokens, temperature = cfg.generate.temperature, do_sample=cfg.generate.do_sample, top_p=cfg.generate.top_p)

    for row in results:
        runner.log(row)
    
    runner.finish()

if __name__ == "__main__":
    main()

