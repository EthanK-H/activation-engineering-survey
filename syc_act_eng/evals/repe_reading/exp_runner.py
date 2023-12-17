import syc_act_eng

# import dotenv
import sys
import os

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb
import random

from torch.utils.data import Dataset, DataLoader

from repe import repe_pipeline_registry # TODO: install into env, ensure using common and up-to-date version
repe_pipeline_registry()

from syc_act_eng.data.reading_vector_data.reading_vector_data import get_reading_vector_data, statement_function_dataset
from syc_act_eng.data.eval_data.utils import get_eval_dataset
from syc_act_eng.utils import print_cuda_memory
from syc_act_eng.variables import PROJECT_DIR

class RepeReadingEval():
    def __init__(
        self,
        model,
        tokenizer,
        
        model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1", # model to use
        reading_vec_dataset_name = "sycophancy_function_facts", # dataset to get reading vectors from
        # eval_dataset_name = "feedback-math", # dataset to evaluate model on # OPTIONS=["anthropic_nlp", "feedback-math"]
        eval_n_samples = 20, # number of samples to use for evaluation
        
        reading_batch_size = 8, # batch size for evaluation (keep low to avoid memory issues)
        eval_batch_size = 8, # batch size for evaluation
        coeff = 2.0, # reading vector coefficient
        max_new_tokens = 10, # maximum number of tokens for model to generate
        layer_id = list(range(-5, -18, -1)), # layers to apply reading vectors
        
        do_wandb_track = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path
        self.reading_vec_dataset_name = reading_vec_dataset_name
        self.eval_n_samples = eval_n_samples
        self.reading_batch_size = reading_batch_size
        self.eval_batch_size = eval_batch_size
        self.coeff = coeff
        self.max_new_tokens = max_new_tokens
        self.layer_id = layer_id
        self.do_wandb_track = do_wandb_track
        
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # tags
        if self.model_name_or_path == "mistralai/Mistral-7B-Instruct-v0.1":
            self.user_tag = "[INST]"
            self.assistant_tag = "[/INST]"
        else:
            raise ValueError("Unknown model name or path. Please use a model from https://huggingface.co/mistralai")

        # wandb
        if self.do_wandb_track:
            wandb.init(
                project="sycophancy-activation-engineering",
                entity="robertmccarthy11",
                config=None, # TODO
                name="reading_" +self.reading_vec_dataset_name + "_" + self.eval_dataset_name,
            )
        
        # # reading vectors
        # self.init_reading_vector()

    def init_reading_vector(self):
        # calc reading vectors
        print("calculating reading vectors...")
        self.calc_reading_vectors()
        # simple reading eval
        print("doing simple reading eval...")
        self.simple_reading_eval()
        # test reading vectors
        print("Visualize free generations...")
        sycophancy_scenarios = pd.read_csv(PROJECT_DIR + "/data/raw_data/sycophancy_scenarios.csv")['Statement'].values.tolist()
        inputs = []
        for scenario in sycophancy_scenarios:
            inputs.append(f"{self.user_tag} {scenario} {self.assistant_tag}")
        self.generate_and_log(inputs, name="sycophancy_scenarios")
        
    def calc_reading_vectors(self):
        # TODO: these should be params?
        rep_token = -1
        hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
        n_difference = 1
        direction_method = 'pca'
        rep_reading_pipeline = pipeline("rep-reading", model=self.model, tokenizer=self.tokenizer)
        
        # dataset
        self.reading_dataset = get_reading_vector_data(self.reading_vec_dataset_name, self.tokenizer, self.user_tag, self.assistant_tag)
        
        # log some examples of the dataset
        self.log_reading_dataset_examples(self.reading_dataset)
        
        # get reading vectors
        self.rep_reader = rep_reading_pipeline.get_directions(
            self.reading_dataset['train']['data'], 
            rep_token=rep_token, 
            hidden_layers=hidden_layers, 
            n_difference=n_difference, 
            train_labels=self.reading_dataset['train']['labels'], 
            direction_method=direction_method,
            batch_size=self.reading_batch_size,
        )
        
        # prepare for control
        self.block_name="decoder_block"
        control_method="reading_vec"

        self.rep_control_pipeline = pipeline(
            "rep-control", 
            model=self.model, 
            tokenizer=self.tokenizer,
            layers=self.layer_id, 
            control_method=control_method)
        
        self.positive_activations = {}
        for layer in self.layer_id:
            self.positive_activations[layer] = torch.tensor(self.coeff * self.rep_reader.directions[layer] * self.rep_reader.direction_signs[layer]).to(self.model.device).half()
            
        self.negative_activations = {}
        for layer in self.layer_id:
            self.negative_activations[layer] = torch.tensor(-self.coeff * self.rep_reader.directions[layer] * self.rep_reader.direction_signs[layer]).to(self.model.device).half()
        
    def simple_reading_eval(self):
        rep_token = -1
        hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
        rep_reading_pipeline = pipeline("rep-reading", model=self.model, tokenizer=self.tokenizer)
        
        H_tests = rep_reading_pipeline(
            self.reading_dataset['test']['data'], 
            rep_token=rep_token, 
            hidden_layers=hidden_layers, 
            rep_reader=self.rep_reader,
            batch_size=32
        )
        
        results = {layer: {} for layer in hidden_layers}
        rep_readers_means = {}
        rep_readers_means['honesty'] = {layer: 0 for layer in hidden_layers}

        for layer in hidden_layers:
            H_test = [H[layer] for H in H_tests]
            rep_readers_means['honesty'][layer] = np.mean(H_test)
            H_test = [H_test[i:i+2] for i in range(0, len(H_test), 2)]
            
            sign = self.rep_reader.direction_signs[layer]

            eval_func = min if sign == -1 else max
            cors = np.mean([eval_func(H) == H[0] for H in H_test])
            
            results[layer] = cors

        # Create a figure and axis
        fig, ax = plt.subplots()
        ax.plot(hidden_layers, [results[layer] for layer in hidden_layers])
        if self.do_wandb_track:
            wandb.log({'reading_layers(?)': wandb.Image(fig)}) # TODO: this isn't working
        else:
            plt.show()
        
    def eval(self, eval_dataset_name):
        self.eval_dataset_name = eval_dataset_name
        # Get evaluation dataset
        self.load_eval_dataset()
        
        # Log some examples of the dataset
        self.log_eval_dataset_examples(self.eval_data)
        
        # Evaluate some generations on eval dataset
        # randomly sample 5 examples from eval dataset
        samples = random.sample(self.eval_data, 5)
        sample_prompts = [sample['prompt'] for sample in samples]
        self.generate_and_log(sample_prompts, name=self.eval_dataset_name + '/' + "eval_samples") # TODO: this doesn't work...
        
        # Quantitative evaluations
        # no control
        self.quantative_evaluate_model(activations=None, name=self.eval_dataset_name + '/' +"no_control")
        # positive control
        self.quantative_evaluate_model(activations=self.positive_activations, name=self.eval_dataset_name + '/' +"positive_control")
        # negative control
        self.quantative_evaluate_model(activations=self.negative_activations, name=self.eval_dataset_name + '/' +"negative_control")
        
    def load_eval_dataset(self):
        self.eval_dataset = get_eval_dataset(self.eval_dataset_name, n_samples=self.eval_n_samples)
        self.eval_data = self.eval_dataset.get_data_for_evaluation(user_tag=self.user_tag, assistant_tag=self.assistant_tag, n_samples=self.eval_n_samples)
        
    def generate_and_log(self, inputs, name="temp", max_new_tokens=100):
        baseline_outputs = self.rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)
        positive_control_outputs = self.rep_control_pipeline(inputs, activations=self.positive_activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)
        negative_control_outputs = self.rep_control_pipeline(inputs, activations=self.negative_activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)

        if self.do_wandb_track:
            text_table = wandb.Table(columns=["prompt", "coeff", "no_control", "positive_control", "negative_control"])
            for i in range(len(inputs)):
                prompt = inputs[i]
                coeff = self.coeff
                no_control = baseline_outputs[i][0]['generated_text'].replace(prompt, "")
                positive_control = positive_control_outputs[i][0]['generated_text'].replace(prompt, "")
                negative_control = negative_control_outputs[i][0]['generated_text'].replace(prompt, "")
                text_table.add_data(prompt, coeff, no_control, positive_control, negative_control)
            wandb.log({name: text_table})
        else:
            for i,s,p,n in zip(inputs, baseline_outputs, positive_control_outputs, negative_control_outputs):
                print("===== Prompt =====")
                print(i)
                print("===== No Control =====")
                print(s[0]['generated_text'].replace(i, ""))
                print(f"===== + Control =====")
                print(p[0]['generated_text'].replace(i, ""))
                print(f"===== - Control =====")
                print(n[0]['generated_text'].replace(i, ""))
                print()
                
    def quantative_evaluate_model(self, activations=None, name="temp"):
        # TODO: need to evaluate prompting also!!!
        
        model_answers = []

        # TODO: no need to do one example at a time?
        for example in tqdm(self.eval_data):
            inputs = [example['prompt']]
            
            outputs = self.rep_control_pipeline(inputs, activations=activations, batch_size=1, max_new_tokens=self.max_new_tokens, do_sample=False)
            answer = outputs[0][0]['generated_text'].replace(inputs[0], "")
            
            result = {
                'model_answer': answer,
                'eval_info': example['eval_infos']
            }
            model_answers.append(result)
            
        results = self.eval_dataset.evaluate_answers(model_answers)
        
        if self.do_wandb_track:
            wandb.log({f"{name}_quantative_evals": results})
            
    def log_reading_dataset_examples(self, reading_dataset, n_samples=5):
        if self.do_wandb_track:
            text_table = wandb.Table(columns=['positive', 'negative'])
        
        for s in range(n_samples):
            # pick random even example idx
            i = np.random.randint(len(reading_dataset['train']['labels']))
            # get
            i_data = i * 2
            pair_1 = i_data
            pair_2 = i_data + 1
            
            if self.do_wandb_track:
                text_table.add_data(reading_dataset['train']['data'][pair_1], reading_dataset['train']['data'][pair_2])
            else:
                print(f"\nExample {s}:")
                print(f"  Input 1: {reading_dataset['train']['data'][pair_1]}")
                print(f"  Input 2: {reading_dataset['train']['data'][pair_2]}")
                # print(f"  Label: {data['train']['labels'][i]}")
                
        if self.do_wandb_track:
            wandb.log({"reading_dataset_examples": text_table})
                
    
    def log_eval_dataset_examples(self, eval_data, n_samples=5):
        if self.do_wandb_track:
            text_table = wandb.Table(columns=['prompt'])
            
        for s in range(n_samples):
            # pick random even example idx
            i = np.random.randint(len(eval_data))
            
            # get sample
            sample = eval_data[i]
            prompt = sample['prompt']
            
            if self.do_wandb_track:
                text_table.add_data(prompt)
            else:
                print(f"\nExample {s}:")
                print(f"  Prompt: {prompt}")
                
        if self.do_wandb_track:
            wandb.log({self.eval_dataset_name + '/' + "eval_dataset_examples": text_table})