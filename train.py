import random
import warnings
warnings.filterwarnings('ignore')
import argparse
from pathlib import Path
import re
import os
import numpy as np
import pandas as pd
import string
import pickle
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, AutoTokenizer, EarlyStoppingCallback
from ast import literal_eval
from DatasetLoader import DatasetLoader
from sklearn.model_selection import KFold
from utils import *
import torch
from const import *

def train_(arg_model):

    # Set a random seed for reproducibility
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    rng = np.random.default_rng(random_seed)
    torch.cuda.empty_cache()
    if arg_model:
        model_path = arg_model
    else:
        model_path = "FacebookAI/xlm-roberta-large" 
    
    MODEL = model_path.split("/")[-1]
    print("Model to train is",MODEL)
    path = r"data/V2_train.pkl"
    df = pickle.load(open(path, 'rb')).reset_index()

    print("Data loaded")

    # We use 4-fold for our experiments. For training a single model we will only train for fold 0
    num_folds = 4

    # Create KFold object
    kf = KFold(n_splits=num_folds, shuffle=False)

    # Add a column to the DataFrame to store the fold index
    df['fold'] = -1
    texts, tags = list(df["sentence"].values), list(df["tags"].values)

    # Assign a fold index to each row
    for fold_index, (train_index, val_index) in enumerate(kf.split(df)):
        df.loc[val_index, 'fold'] = fold_index

    # create the mapping tag <=> id
    unique_tags = sorted(set(tag for doc in tags for tag in doc))
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}

    # save the dict
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('models/results-'+MODEL):
        os.makedirs('models/results-'+MODEL)
    with open(os.path.join('models/results-'+MODEL, 'tag2id.pkl'), 'wb') as f:
        pickle.dump(tag2id, f)

    id2tag = {id: tag for tag, id in tag2id.items()}
    # tokenize the word
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Use zero-shot data for early stopping check
    df_zs = pickle.load(open("data/V2_zero_shot.pkl", 'rb')).head(10000)

    texts_zs , tags_zs = list(df_zs["sentence"].values), list(df_zs["tags"].values)
    zs_encodings = tokenizer(texts_zs, is_split_into_words=True, return_offsets_mapping=False, padding="max_length",
                                truncation=True, max_length=50)
    zs_labeled_encodings = tokenize_and_align_labels(zs_encodings, tags_zs, tag2id)
    zs_labels = zs_labeled_encodings["labels"]
    zs_dataset = DatasetLoader(zs_encodings, zs_labels)

    # You can iterate over the folds using the fold index, here we fix it as 0
    fold_index = 0

    # Extract training and validation sets for the current fold
    train_set = df[df['fold'] != fold_index]
    val_set = df[df['fold'] == fold_index].head(5000)

    train_texts, train_tags = list(train_set["sentence"].values), list(train_set["tags"].values)
    val_texts, val_tags = list(val_set["sentence"].values), list(val_set["tags"].values)

    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=False, padding="max_length",
                              truncation=True, max_length=50)    
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=False, padding="max_length",
                              truncation=True, max_length=50)
    print("words tokenized")

    train_encodings = tokenize_and_align_labels(train_encodings, train_tags, tag2id)
    val_encodings = tokenize_and_align_labels(val_encodings, val_tags, tag2id)

    train_labels = train_encodings["labels"]
    val_labels = val_encodings["labels"]

    train_dataset = DatasetLoader(train_encodings, train_labels)
    val_dataset = DatasetLoader(val_encodings, val_labels)

    def model_init():
        model = AutoModelForTokenClassification.from_pretrained(model_path, 
                                                               num_labels=len(unique_tags) 
                                                               )
        model.gradient_checkpointing_enable()
        return model

    training_args = TrainingArguments(
        output_dir='models/results-' + MODEL + " " + str(fold_index),  # output directory
        num_train_epochs=1,  # total number of training epochs
        per_device_train_batch_size=1024,  # batch size per device during training
        per_device_eval_batch_size=1024,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='models/results-' + MODEL + " " + str(fold_index),  # directory for storing logs
        logging_steps=20,
        optim = "adamw_torch",
        eval_steps=20,
        save_steps=20,
        save_strategy="steps",
        evaluation_strategy = "steps",
        save_total_limit=5,
        load_best_model_at_end=True,
        seed=42,
        report_to="none",
        #metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model_init(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=zs_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("start training")
    trainer.train()
    trainer.save_model("models/results-"+MODEL + " " + str(fold_index))


    lh = trainer.state.log_history
    with open(os.path.join('models/results-' + MODEL+ " "+ str(fold_index), 'history_log'+'.pkl'), 'wb') as f:
        pickle.dump(lh, f)
    print("model saved")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="Specify model name")
    args = parser.parse_args()

    model = args.model  # access the argument value using its name
    train_(model)