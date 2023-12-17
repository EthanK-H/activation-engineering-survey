import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import shutil
import os
from datetime import datetime
from glob import glob
import torch.nn.functional as F


def get_model_and_tokenizer(model_name, token=None, cache_dir=None, load_model=True): # TODO: cache_dir
    assert model_name == "meta-llama/Llama-2-7b-hf", "others not tested"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    tokenizer.pad_token = tokenizer.eos_token
    
    if load_model:
        model = Llama27BHelper(token=token, cache_dir=cache_dir)
    else:
        model = None
    
    return model, tokenizer

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.last_hidden_state = None
        self.add_activations = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs) # output after going through the block
        self.last_hidden_state = output[0] # in transformer models on hugging face, the hidden states of the token are at the first position (output[0]) of the tuple
        # the rest of the output tuple might have past hidden states, attention weights, weights
        if self.add_activations is not None:
            output = (output[0] + self.add_activations,) + output[1:] # you add the activations to the output of the hidden states only and then you keep going?
        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.last_hidden_state = None
        self.add_activations = None


class Llama27BHelper:
    def __init__(self, pretrained_model="meta-llama/Llama-2-7b-hf", token=None, cache_dir=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model, use_auth_token=token, cache_dir=cache_dir, torch_dtype=torch.float16).to(self.device) # gets the model weights
        for i, layer in enumerate(self.model.model.layers): # layer X will be specified, so you go from first layer to X layer
        # actual transformer layers are inside self.model.model.layers
            self.model.model.layers[i] = BlockOutputWrapper(layer)

    def generate_text(self, prompt, max_length=100): # TODO: does this work for a batch greater than 1??
        inputs = self.tokenizer(prompt, return_tensors="pt")
        max_length += inputs['input_ids'].shape[-1]
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        batch_decoded = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return batch_decoded[0]

    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens.to(self.device)).logits
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].last_hidden_state

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations)

    def reset_all(self):
        print(self.model.model.layers)
        for layer in self.model.model.layers:
            layer.reset()
            
class ComparisonDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        s_text = item['sycophantic_text']
        n_text = item['non_sycophantic_text']

        s_tokens = self.tokenizer.encode(s_text, return_tensors="pt")[0]
        n_tokens = self.tokenizer.encode(n_text, return_tensors="pt")[0]

        return s_tokens, n_tokens

    
def get_nina_vector(dataset, model, layers):
    '''
    TODO: implement saving
    
    '''
    # She only took activations from "intermediate activations at a late layer (I tested 28, 29, and 30)"
    layers = list(range(21, 30))

    # # dictionary of layer + making a filename with a timestamp along w the layer
    # filenames = dict([(layer, make_filename(layer)) for layer in layers])
    
    # dictionary of each layer and [] for each layer
    diffs = dict([(layer, []) for layer in layers])

    for s_tokens, n_tokens in tqdm(dataset, desc='Processing prompts'):
        
        s_out = model.get_logits(s_tokens.unsqueeze(0))
        for layer in layers: # for all the layers we are collecting activations for
            # in the Llama27BHelper class, you get the hidden state from the BlockOutputWrapper class, where you go forward in the model with the token
            # and after you reach the end of the decoder block
            s_activations = model.get_last_activations(layer)
            s_activations = s_activations[0, -2, :].detach().cpu()
            diffs[layer].append(s_activations)
        
        n_out = model.get_logits(n_tokens.unsqueeze(0))
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, -2, :].detach().cpu()
            diffs[layer][-1] -= n_activations

    for layer in layers:
        diffs[layer] = torch.stack(diffs[layer]) # torch.Size([x, 4096]) - now 2D tensor with all the x samples and their vector of 4096 activations in one layer
        # torch.save(diffs[layer], filenames[layer])
        
    # Important: mean and normalize
    for layer in layers:
        vec = diffs[layer].mean(dim=0)
        unit_vec = vec / torch.norm(vec, p=2)
        diffs[layer] = unit_vec
    
    return diffs