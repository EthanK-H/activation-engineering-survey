import wandb
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from syc_act_eng.evals.repe_reading.exp_runner import RepeReadingEval

def get_sweep():
    sweep = [
        {
            'reading_vec_dataset': "sycophancy_function_facts",
            'eval_datasets': ["anthropic_nlp_unsure"],
            'coeff': 2.0,
        },
    ]
    return sweep

def get_default_config():
    return {
        "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.1",
        "eval_n_samples": 20,
        "cache_dir": "/workspace/model_cache",
        "token": "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",
        "reading_batch_size": 8,
        "eval_batch_size": 8,
        "max_new_tokens": 10,
        "layer_id": list(range(-5, -18, -1)),
        "do_wandb_track": True,
        "verbose": True,
    }   

def main():
    # TODO: should use proper configs, or dict at least so can log to wandb
    
    config = get_default_config()

    # load model
    model = AutoModelForCausalLM.from_pretrained(config['model_name_or_path'], torch_dtype=torch.float16, device_map="auto", use_auth_token=config['token'], cache_dir=config['cache_dir'])
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
    tokenizer.pad_token_id = 0

    # Get sweep
    sweep = get_sweep()

    # Do sweep
    for reading_vec_sweep in sweep:
        config.update(reading_vec_sweep)

        # wandb
        if config['do_wandb_track']:
            wandb.init(
                project="sycophancy-activation-engineering",
                entity="robertmccarthy11",
                config=config,
                name="reading_" + config['reading_vec_dataset'],
            )
        
        # initialize
        exp_runner = RepeReadingEval(
            model=model,
            tokenizer=tokenizer,
            model_name_or_path=config['model_name_or_path'],
            reading_vec_dataset_name=config['reading_vec_dataset'],
            eval_n_samples=config['eval_n_samples'],
            reading_batch_size=config['reading_batch_size'],
            eval_batch_size=config['eval_batch_size'],
            coeff=config['coeff'],
            max_new_tokens=config['max_new_tokens'],
            layer_id=config['layer_id'],
            do_wandb_track=config['do_wandb_track'],
            verbose=config['verbose'],
        )

        # calc reading vecs
        exp_runner.init_reading_vector()

        for eval_dataset_name in config['eval_datasets']:
            # eval on eval dataset
            exp_runner.eval(eval_dataset_name)

        wandb.finish()

if __name__ == "__main__":
    main()