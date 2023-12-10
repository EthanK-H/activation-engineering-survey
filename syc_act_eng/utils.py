import torch

def print_cuda_memory():
    print("CUDA Memory Summary")
    print("===================")

    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    reserved_memory = torch.cuda.memory_reserved(0)
    free_memory = total_memory - allocated_memory

    print(f"Total Memory: {total_memory / 1e9:.2f} GB")
    print(f"Allocated Memory: {allocated_memory / 1e9:.2f} GB")
    print(f"Reserved Memory: {reserved_memory / 1e9:.2f} GB")
    print(f"Free Memory: {free_memory / 1e9:.2f} GB")