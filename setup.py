from setuptools import setup, find_packages

setup(
    name='syc_act_eng',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "transformers",
        "accelerate",
        "matplotlib",
        "seaborn",
        "pandas",
        "datasets",
        "numpy",
        "sentencepiece",
        "protobuf",
        "scikit-learn",
        "torch",
        "tqdm",
        "wandb",
        'repe@git+https://github.com/RobertMcCarthy97/representation-engineering.git'
    ],
    dependency_links=[
        'git+https://github.com/RobertMcCarthy97/representation-engineering.git#egg=repe'
    ],
)
