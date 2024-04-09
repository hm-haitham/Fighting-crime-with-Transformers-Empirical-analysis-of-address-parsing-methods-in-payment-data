import warnings
warnings.filterwarnings('ignore')

import numpy as np

from tqdm import tqdm
import pickle
import string
from ast import literal_eval
from pickle import load
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime
from utils import *
from const import class_set



def compute_metrics(df, tag2id, id2tag):
    """
    tag_pairs = {}
    for t in set([k[2:] for k in tag2id.keys()]):
        tag_pairs[t] = [0, 0]
    for k, v in tag2id.items():
        prefix, t = k.split("-")
        if prefix == "B":
            tag_pairs[t][0] = v
        else:
            tag_pairs[t][1] = v

    int_ids = [v for k, v in tag2id.items() if k[0] == "I"]
    beginnings_ids = [v for k, v in tag2id.items() if k[0] == "B"]
    """
    for class_tag in class_set:
        df[class_tag+"_metrics"] = df.apply(lambda x: class_metrics(x, class_tag), axis=1)
    
    cond = df["len"] < 10
    df_clean_adr = df[df["gt_CLEAN_ADDRESS"] == "X"]
    df_no_clean_adr = df[df["gt_CLEAN_ADDRESS"] != "X"]

    print("clean addr total", len(df_clean_adr))
    print("f1 for less than 10 chars:", df_clean_adr[df_clean_adr["len"] < 10]["f1"].mean(), "with total", len(df_clean_adr[df_clean_adr["len"] < 10]))
    print("f1 for 10 chars or more:", df_clean_adr[df_clean_adr["len"] >= 10]["f1"].mean(), "with total", len(df_clean_adr[df_clean_adr["len"] >= 10]))
    print("not clean addr total", len(df_no_clean_adr))
    print("f1 for less than 10 chars:", df_no_clean_adr[df_no_clean_adr["len"] < 10]["f1"].mean(), "with total", len(df_no_clean_adr[df_no_clean_adr["len"] < 10]))
    print("f1 for 10 chars or more:", df_no_clean_adr[df_no_clean_adr["len"] >= 10]["f1"].mean(), "with total", len(df_no_clean_adr[df_no_clean_adr["len"] >= 10]))

    metrics_dict = dict()
    metrics_dict["overall"] = dict()
    metrics_dict["overall"]["total"] = np.round(df["f1"].mean(), 5)
    metrics_dict["overall"]["clean_short"] = np.round(df_clean_adr[df_clean_adr["len"] < 10]["f1"].mean(), 5)
    metrics_dict["overall"]["clean_long"] = np.round(df_clean_adr[df_clean_adr["len"] >= 10]["f1"].mean(), 5)
    metrics_dict["overall"]["not_clean_short"] = np.round(df_no_clean_adr[df_no_clean_adr["len"] < 10]["f1"].mean(), 5)
    metrics_dict["overall"]["not_clean_long"] = np.round(df_no_clean_adr[df_no_clean_adr["len"] >= 10]["f1"].mean(), 5)

    for class_tag in class_set:
        metrics_dict[class_tag] = dict()
        metrics_dict[class_tag]["total"] = np.round(df[class_tag+"_metrics"].mean(), 5)
        metrics_dict[class_tag]["clean_short"] = np.round(df_clean_adr[df_clean_adr["len"] < 10][class_tag+"_metrics"].mean(), 5)
        metrics_dict[class_tag]["clean_long"] = np.round(df_clean_adr[df_clean_adr["len"] >= 10][class_tag+"_metrics"].mean(), 5)
        metrics_dict[class_tag]["not_clean_short"] = np.round(df_no_clean_adr[df_no_clean_adr["len"] < 10][class_tag+"_metrics"].mean(), 5)
        metrics_dict[class_tag]["not_clean_long"] = np.round(df_no_clean_adr[df_no_clean_adr["len"] >= 10][class_tag+"_metrics"].mean(), 5)

    return metrics_dict
