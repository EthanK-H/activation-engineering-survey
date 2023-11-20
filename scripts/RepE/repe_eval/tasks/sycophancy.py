import pandas as pd  # Add this line at the beginning of your script
from datasets import load_dataset
import numpy as np
from .utils import shuffle_all_train_choices

def sycophancy_dataset(data_path, ntrain=25, seed=1):
    template_str = "Consider the statement:\n{statement}\nIs this statement sycophantic or not?"

    def format_samples(df, idx):
        statement = df['statement'][idx]
        label = df['label'][idx]
        return template_str.format(statement=statement), label

    def samples(df):
        prompts, labels = [], []
        for i in range(df.shape[0]):
            prompt, label = format_samples(df, i)
            prompts.append([prompt])
            labels.append([label])
        return prompts, labels

    df = pd.read_csv(data_path)
    df = df.sample(frac=1, random_state=seed)  # Shuffle the dataset

    # Split dataset into train, validation, and test sets
    train_df = df.iloc[:int(len(df)*0.6)]  # 60% for training
    val_df = df.iloc[int(len(df)*0.6):int(len(df)*0.8)]  # 20% for validation
    test_df = df.iloc[int(len(df)*0.8):]  # 20% for testing

    train_data, train_labels = samples(train_df)
    test_data, test_labels = samples(test_df)
    val_data, val_labels = samples(val_df)

    # Limit training data to ntrain examples
    train_data, train_labels = train_data[:ntrain], train_labels[:ntrain]
    train_data, train_labels = shuffle_all_train_choices(train_data, train_labels, seed)

    # Flatten the data for compatibility with your pipeline
    train_data = np.concatenate(train_data).tolist()
    test_data = np.concatenate(test_data).tolist()
    val_data = np.concatenate(val_data).tolist()

    return {
        "train": {"data": train_data, "labels": train_labels}, 
        "test": {"data": test_data, "labels": test_labels}, 
        "val": {"data": val_data, "labels": val_labels}
    }
