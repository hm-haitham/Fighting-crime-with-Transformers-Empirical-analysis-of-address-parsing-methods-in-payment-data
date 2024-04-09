# Fighting-crime-with-Transformers-Empirical-analysis-of-address-parsing-methods-in-payment-data
Code linked to the paper: https://arxiv.org/abs/2404.05632

## Table of Contents

- [Address Parser](#address-parser)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Data](#data)
  - [Project Structure](#project-structure)
  - [Usage](#usage)

## Introduction

To ensure adherence with regulatory requirements,
it is essential for financial institutions to under-
stand precisely where the money is originating and
where it is flowing. The new standard of interna-
tional payment messages ISO 20022 for SWIFT
has the potential to simplify the task of locating
payment parties by enabling the beneficiary and
originator address to be delivered in a structured
format. However, a considerable amount of mes-
sages are still delivered with an address in free text
form. This problem is further exacerbated by the
use of legacy payment processing platforms. Thus
Address Parsing is required to extract address fields
such as street, postal code, city, or country.

Our work has three main contributions. Firstly,
it offers an open-sourced, augmented dataset, ad-
dressing the limitations of bench-marking on clean
datasets and enabling research on noisy real-world
payment data. Secondly, by empirically analyz-
ing and comparing various techniques, this paper
uncovers an effective approach for multinational
address parsing on distorted data. Lastly, we open-
source the fine-tuned state-of-the-art model, aiding
future research and application in a multinational
setup written in Latin alphabet and transliterated in
ASCII format.


## Data

You can download the data from https://paperswithcode.com/dataset/v2-train-pkl
Or directly from HF : https://huggingface.co/datasets/hm-haitham/address_parser_data

The downloaded datasets should be placedin the data folder

## Project Structure

The project structure is as follows:

```
├── const.py
├── data
├── DatasetLoader.py
├── llm_inference.ipynb
├── llm_inference_script.py
├── llm_training.ipynb
├── llm_training_script.py
├── metrics.py
├── predict.py
├── README.md
├── requirements.txt
├── train.py
└── utils.py


```

## Usage
After downloading the data, you can run train.py to train a Transformer. You can specify which model to train by adding the argument -model <your-model-from-huggingface>, default is xlm-roberta-large. Example:
```bash
python train.py -model distilbert-base-uncased
```
For prediction, you can run predict.py while specifying the path to your fine-tuned model. Given that the model is saved in the path "models/results-<your-model>", you could run:
```bash
python predict.py -model your-model
```
We have the same structure for training and inference of LLMs, with the default being "mistralai/Mistral-7B-Instruct-v0.2". We have additionally provided jupyter notebooks showing the procedure of both scripts.
```


