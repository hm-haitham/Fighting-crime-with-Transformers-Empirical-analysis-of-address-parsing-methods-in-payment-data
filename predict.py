import warnings

from metrics import *
import transformers
warnings.filterwarnings('ignore')

from pathlib import Path
import re
import os
import numpy as np
import pandas as pd
import string
import pickle
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, AutoTokenizer
from ast import literal_eval
from DatasetLoader import DatasetLoader
from sklearn.model_selection import train_test_split
from utils import *
from datetime import datetime
from const import *
import argparse



def predict_(arg_model):

    # Set a random seed for reproducibility
    random_seed = 42
    random.seed(random_seed)
    rng = np.random.default_rng(random_seed)
    if arg_model:
        model_path = arg_model
    else:
        print("You need to add a correct path to the model")
        return None
    
    MODEL = model_path.split("/")[-1]
    print("Model to predict is",MODEL)

    for path in [
        r"data/V2_train.pkl",
        r"data/V2_test.pkl",
        r"data/V2_zero_shot.pkl",
    ]:
        if "zero_shot" in path:
            print("\nrunning predictions for zero_shot data")
            output_path = "zero_shot"
        elif "test" in path:
            print("\nrunning predictions for test data")
            output_path = "test"
        else:
            print("\nrunning predictions for train data")
            output_path="train"
        with open(path, 'rb') as f:
            df=pickle.load(f).head(500)
        df = df[df["tags"].apply(lambda x : len(x) > 0 )]
        texts, tags = list(df["sentence"].values), list(df["tags"].values)

        # create the mapping tag <=> id
        unique_tags = sorted(set(tag for doc in tags for tag in doc))
        #with open(os.path.join('results-'+MODEL, 'tag2id.pkl'), 'rb') as f:
        #    tag2id = pickle.load(f)
        tag2id = {tag: id for id, tag in enumerate(unique_tags)}
        id2tag = {id: tag for tag, id in tag2id.items()}
        # tokenize the word
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                    truncation=True, max_length=256)
        labeled_encodings = tokenize_and_align_labels(encodings, tags, tag2id)
        labels = labeled_encodings["labels"]
        dataset = DatasetLoader(encodings, labels)
        

        #MODEL = "distilbert" #VERSION + " model"
        model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(unique_tags))

        model.eval()

        training_args = TrainingArguments(
            output_dir='./results-' + MODEL,  # output directory
            per_device_train_batch_size=100,  # batch size per device during training
            per_device_eval_batch_size=100,  # batch size for evaluation
        )
        trainer = Trainer(
            model=model,
            args=training_args)
        preds = trainer.predict(dataset)

        # LOG RESULTS
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists('models/results-'+MODEL):
            os.makedirs('models/results-'+MODEL)
        pred_labels = np.argmax(preds.predictions, axis=2)
        total_acc_count = 0
        total_label_count = 0
        sum_acc = 0
        unk_mask = np.array([-100] * len(dataset.labels[0]))
        for input_id, pred_label, label in zip(dataset.encodings.input_ids, pred_labels, dataset.labels):
            acc_count = sum(pred_label == label)
            label_count = sum(label != unk_mask)
            sample_acc = acc_count / label_count
            total_acc_count += acc_count
            total_label_count += label_count
            sum_acc += sample_acc
        print("average acc:", sum_acc / len(dataset.labels))
        print("weighted avg acc:", total_acc_count / total_label_count)
        preds_trunc = []
        for i in range(len(pred_labels)):
            preds_trunc.append(pred_labels[i, get_first_occurrence_indices(encodings.word_ids(i))])
        df["preds"] = preds_trunc
        df["preds_tags"] = df["preds"].apply(lambda y: [id2tag[x] for x in y])
        df["TP"] = df.apply(lambda x: sum((y == z for y, z in zip(x["tags"], x["preds_tags"]))), axis=1)

        df["class"] = df["tags"].apply(lambda y: [x[2:] for x in y])
        df["class_preds"] = df["preds_tags"].apply(lambda y: [x[2:] for x in y])

        df["precision"] = df.apply(lambda x: sum(c == cp for c, cp in zip(x["class"], x["class_preds"]) if c not in ["HardSep"]) / sum([c not in ["HardSep"] for c in x["class"]]), axis=1)
        df["recall"] = df.apply(
            lambda x: sum(
                [
                    c == cp != "OOA" 
                    for c, cp in zip(x["class"], x["class_preds"])
                    if c not in ["HardSep"]
                ]
                ) 
                / sum([c not in ["HardSep", "OOA"] for c in x["class"]])
                  if sum([c!="OOA" for c in x["class"]]) !=0 
                  else 1, 
                  axis=1)
        df["f1"] = df.apply(lambda x: 2*x["precision"] * x["recall"] / (x["precision"] + x["recall"]) if (x["precision"] + x["recall"]) != 0 else 0, axis=1)
        df["len"] = df["sentence"].apply(len)
        def get_clean_address(x):
            if "B-OOA" in x or "OOA" in x:
                return None
            else:
                return "X"
        df["gt_CLEAN_ADDRESS"]=df.tags.apply(get_clean_address)
        print("f1:", df["f1"].mean())
        timestamp = str(datetime.now()).replace(":","-").split(".")[0]
        with open(os.path.join('models/results-' + MODEL, 'df_preds_'+output_path+'.pkl'), 'wb') as f:
            pickle.dump(df, f)
        print("prediction done at", datetime.now())
        metrics_dict = compute_metrics(df, tag2id, id2tag)

        with open(os.path.join('models/results-' + MODEL, 'metrics_'+output_path+'.pkl'), 'wb') as f:
            pickle.dump(metrics_dict, f)
        print("metrics computed at", datetime.now())

        # Filter columns with the suffix '_metric'
        metric_columns = df.filter(like='_metric')

        # Concatenate 'Country' with the filtered metric columns
        df_filtered = pd.concat([df['Country'], metric_columns], axis=1)

        # Group by 'Country' and calculate the mean for columns with the suffix '_metric'
        result = pd.DataFrame(df_filtered.groupby('Country').mean())
        with open(os.path.join('models/results-' + MODEL, 'country_df_'+output_path+'.pkl'), 'wb') as f:
            pickle.dump(result, f)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="Specify model path")
    args = parser.parse_args()

    model = args.model  # access the argument value using its name
    predict_(model)