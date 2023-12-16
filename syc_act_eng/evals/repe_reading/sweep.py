
def get_sweep():
    sweep = [
        {
            'reading_vec_dataset': "sycophancy_function_facts",
            'eval_datasets': ["anthropic_nlp"],
            'coeff': 2.0
        },
    ]
    return sweep


def main():
    # TODO: should use proper configs, or dict at least so can log to wandb
    
    # reading_vec_dataset_name = "sycophancy_function_facts" # dataset to get reading vectors from
    # eval_dataset_name = "anthropic_nlp" # dataset to evaluate model on # OPTIONS=["anthropic_nlp", "feedback-math"]
    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1" # model to use
    eval_n_samples = 20 # number of samples to use for evaluation
    cache_dir = '/workspace/model_cache' # where to save and load model cache
    token = "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" # huggingface token
    reading_batch_size = 8 # batch size for evaluation (keep low to avoid memory issues)
    eval_batch_size = 8 # batch size for evaluation
    # coeff = 2.0 # reading vector coefficient
    max_new_tokens = 10 # maximum number of tokens for model to generate
    layer_id = list(range(-5, -18, -1)) # layers to apply reading vectors
    do_wandb_track = True

    # load model
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", use_auth_token=token, cache_dir=cache_dir)
        use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    else:
        use_fast_tokenizer = True
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
    tokenizer.pad_token_id = 0

    # Get sweep
    sweep = get_sweep()

    # Do sweep
    for reading_vec_sweep in sweep:
        reading_vec_dataset_name = reading_vec_sweep['reading_vec_dataset']
        eval_datasets = reading_vec_sweep['eval_datasets']
        coeff = reading_vec_sweep['coeff']
        
        # initialize
        exp_runner = RepeReadingEval(
            model=model,
            tokenizer=tokenizer,
            model_name_or_path=model_name_or_path,
            reading_vec_dataset_name=reading_vec_dataset_name,
            eval_n_samples=eval_n_samples,
            reading_batch_size=reading_batch_size,
            eval_batch_size=eval_batch_size,
            coeff=coeff,
            max_new_tokens=max_new_tokens,
            layer_id=layer_id,
            do_wandb_track=do_wandb_track
        )

        # calc reading vecs
        exp_runner.init_reading_vector()

        for eval_dataset_name in eval_datasets:
            # eval on eval dataset
            exp_runner.eval(eval_dataset_name)
    

if __name__ == "__main__":
    main()