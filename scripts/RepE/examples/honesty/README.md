`honesty.ipynb` demonstrates how we could potentially use representation reading techniques to detect lies and hallucinations generated by large language models. It shows how we extract a vector representation corresponding to "honesty" from the model by using LAT. We then visualize this honesty representation across layers and tokens to detect instances where the model is being dishonest or lying in its generations. Specifically, this notebook loads a pretrained language model and tokenizer, extracts an honesty direction using LAT on an unlabelled dataset of facts, and applies this to test on scenarios where the model is given incentives to lie. It generates visualizations showing the dishonesty scores across layers for each token. By summing the dishonesty scores at each token, we build a simple lie detector that distinguishes between honest and dishonest behaviors in the model's generation. The notebook demonstrates how these techniques can be used to monitor and control honesty and truthfulness in large language models.

`honesty_control_TQA.ipynb` contains code to reproduce the Contrast Vector control baseline results on TruthfulQA.

For more details, please check out section 4 of [our RepE paper](https://arxiv.org/abs/2310.01405).