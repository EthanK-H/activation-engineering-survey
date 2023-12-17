import pandas as pd
import numpy as np
import random
from transformers import PreTrainedTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

from syc_act_eng.variables import PROJECT_DIR


def get_reading_vector_data(dataset_name: str, tokenizer: PreTrainedTokenizer, user_tag: str = "", assistant_tag: str = "", seed: int = 0) -> (list, list):
    
    if dataset_name == "sycophancy_function_facts":
        
        data_path = PROJECT_DIR + "/data/raw_data/repe/facts/facts_true_false.csv" # TODO: use better path finding
        # return sycophancy_function_dataset(data_path, tokenizer, user_tag, assistant_tag, seed)
        
        df = pd.read_csv(data_path)
        true_statements = df[df['label'] == 1]['statement'].values.tolist()
        statements = true_statements
        
        prompts = {
            "positive": "Pretend you're an honest person making statements about the world.",
            "negative": "Pretend you're a sycophantic person making statements about the world."
            }
        
        return statement_function_dataset(statements, tokenizer, prompts, user_tag, assistant_tag)
        
    
    elif dataset_name == "honesty_function_facts":
        
        data_path = PROJECT_DIR + "/data/raw_data/repe/facts/facts_true_false.csv"
        # return honesty_function_dataset(data_path, tokenizer, user_tag, assistant_tag, seed)
        
        df = pd.read_csv(data_path)
        true_statements = df[df['label'] == 1]['statement'].values.tolist()
        statements = true_statements
        
        prompts = {
            "positive": "Pretend you're an honest person making statements about the world.",
            "negative": "Pretend you're a dishonest person making statements about the world."
            }
        
        return statement_function_dataset(statements, tokenizer, prompts, user_tag, assistant_tag) # TODO: verify this gives same results as default dataset!
    
    else:
        raise ValueError(f"Dataset {dataset_name} not found.")
        

# def honesty_function_dataset(data_path: str, tokenizer: PreTrainedTokenizer, user_tag: str = "", assistant_tag: str = "", seed: int = 0) -> (list, list):
#     """
#     Processes data to create training and testing datasets based on honesty.

#     Args:
#     - data_path (str): Path to the CSV containing the data.
#     - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
#     - user_tag (str): Instruction template.
#     - assistant_tag (str): Instruction template user tag.
#     - seed (int): Random seed for reproducibility.

#     Returns:
#     - Tuple containing train and test data.
#     """

#     # Setting the seed for reproducibility
#     random.seed(seed)

#     # Load the data
#     df = pd.read_csv(data_path)
#     true_statements = df[df['label'] == 1]['statement'].values.tolist()
#     false_statements = df[df['label'] == 0]['statement'].values.tolist()

#     template_str = "Pretend you're {type} person making statements about the world."
#     honest_statements = []
#     untruthful_statements = []

#     # Process statements
#     for statement in true_statements:
#         tokens = tokenizer.tokenize(statement)
        
#         for idx in range(1, len(tokens) - 5):
#             truncated_tokens = tokens[:idx]
#             truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

#             honest_statements.append(f"{user_tag} {template_str.format(type='an honest')} {assistant_tag} " + truncated_statement)
#             untruthful_statements.append(f"{user_tag} {template_str.format(type='an untruthful')} {assistant_tag} " + truncated_statement)

#     # Create training data
#     ntrain = 512
#     combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
#     train_data = combined_data[:ntrain]

#     train_labels = []
#     for d in train_data:
#         true_s = d[0]
#         random.shuffle(d)
#         train_labels.append([s == true_s for s in d])
    
#     train_data = np.concatenate(train_data).tolist()

#     # Create test data
#     reshaped_data = np.array([[honest, untruthful] for honest, untruthful in zip(honest_statements[:-1], untruthful_statements[1:])]).flatten()
#     test_data = reshaped_data[ntrain:ntrain*2].tolist()

#     print(f"Train data: {len(train_data)}")
#     print(f"Test data: {len(test_data)}")

#     return {
#         'train': {'data': train_data, 'labels': train_labels},
#         'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
#     }

# def sycophancy_function_dataset(data_path: str, tokenizer: PreTrainedTokenizer, user_tag: str = "", assistant_tag: str = "", seed: int = 0) -> (list, list):
#     """
#     Processes data to create training and testing datasets based on honesty.

#     Args:
#     - data_path (str): Path to the CSV containing the data.
#     - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
#     - user_tag (str): Instruction template.
#     - assistant_tag (str): Instruction template user tag.
#     - seed (int): Random seed for reproducibility.

#     Returns:
#     - Tuple containing train and test data.
#     """

#     # Setting the seed for reproducibility
#     random.seed(seed)

#     # Load the data
#     df = pd.read_csv(data_path)
#     true_statements = df[df['label'] == 1]['statement'].values.tolist()
#     false_statements = df[df['label'] == 0]['statement'].values.tolist()

#     template_str = "Pretend you're {type} person making statements about the world."
#     honest_statements = []
#     untruthful_statements = []

#     # Process statements
#     for statement in true_statements:
#         tokens = tokenizer.tokenize(statement)
        
#         for idx in range(1, len(tokens) - 5):
#             truncated_tokens = tokens[:idx]
#             truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

#             honest_statements.append(f"{user_tag} {template_str.format(type='an honest')} {assistant_tag} " + truncated_statement)
#             untruthful_statements.append(f"{user_tag} {template_str.format(type='a sycophantic')} {assistant_tag} " + truncated_statement)

#     # Create training data
#     ntrain = 512
#     combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
#     train_data = combined_data[:ntrain]

#     train_labels = []
#     for d in train_data:
#         true_s = d[0]
#         random.shuffle(d)
#         train_labels.append([s == true_s for s in d])
    
#     train_data = np.concatenate(train_data).tolist()

#     # Create test data
#     reshaped_data = np.array([[honest, untruthful] for honest, untruthful in zip(honest_statements[:-1], untruthful_statements[1:])]).flatten()
#     test_data = reshaped_data[ntrain:ntrain*2].tolist()

#     print(f"Train data: {len(train_data)}")
#     print(f"Test data: {len(test_data)}")

#     return {
#         'train': {'data': train_data, 'labels': train_labels},
#         'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
#     }
    
def statement_function_dataset(
    statements: str,
    tokenizer: PreTrainedTokenizer,
    prompts: dict[str: str],
    user_tag: str = "",
    assistant_tag: str = "",
    ntrain: int = 512,
    seed: int = 0
) -> (list, list):

    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - statements (list): List of statements to be used as the basis for the dataset.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - prompts (dict[str: str]): Dictionary of prompts to be used for the dataset.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)
    
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for statement in statements:
        tokens = tokenizer.tokenize(statement)
        
        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

            honest_statements.append(f"{user_tag} {prompts['positive']} {assistant_tag} " + truncated_statement)
            untruthful_statements.append(f"{user_tag} {prompts['negative']} {assistant_tag} " + truncated_statement)

    # Create training data
    assert len(honest_statements) > 2 * ntrain, "assumes this is true"
    combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array([[honest, untruthful] for honest, untruthful in zip(honest_statements[:-1], untruthful_statements[1:])]).flatten()
    test_data = reshaped_data[ntrain:ntrain*2].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
    }
    
    
def qa_function_dataset(
    qas: list,
    tokenizer: PreTrainedTokenizer,
    prompts: dict[str: str],
    user_tag: str = "",
    assistant_tag: str = "",
    ntrain: int = 512,
    seed: int = 0
) -> (list, list):
    """
    Processes data to create training and testing datasets based on honesty.
    
    Questions and answers (rather than just statements)

    Args:
    - qas (list): List of question-answer pairs in dicts
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - prompts (dict[str: str]): Dictionary of prompts to be used for the dataset.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)
    
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for qa in qas:
        ans_tokens = tokenizer.tokenize(qa['answer'])
        
        for idx in range(1, len(ans_tokens) - 5):
            truncated_tokens = ans_tokens[:idx]
            truncated_answer = tokenizer.convert_tokens_to_string(truncated_tokens)

            honest_statements.append(f"{user_tag} {qa['question']} {prompts['positive']} {assistant_tag} " + truncated_answer)
            untruthful_statements.append(f"{user_tag} {qa['question']} {prompts['negative']} {assistant_tag} " + truncated_answer)

    # Create training data
    assert len(honest_statements) > 2 * ntrain, "assumes this is true"
    combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array([[honest, untruthful] for honest, untruthful in zip(honest_statements[:-1], untruthful_statements[1:])]).flatten()
    test_data = reshaped_data[ntrain:ntrain*2].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
    }
    