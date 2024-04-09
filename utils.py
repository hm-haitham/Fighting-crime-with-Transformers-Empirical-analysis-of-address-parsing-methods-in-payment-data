from ast import literal_eval
import string
import random


def fix_tags(tags):
    if len(tags) == 0:
        return tags

    new_tags = []
    for i in range(0, len(tags)):
        if tags[i] == "postbox":
            new_tags.append("OOA")
        else:
            new_tags.append(tags[i])
    return new_tags

def convert_to_list(s):
    try:
        return literal_eval(s)
    except (SyntaxError, ValueError):
        return s

converters = {"gt_tags": convert_to_list, "final_address": convert_to_list}

def remove_ooa(row):
    new_sentence = []
    new_tags = []

    for a, t in zip(row["sentence"], row["tags"]):
        if t != "OOA":
            new_tags.append(t)
            new_sentence.append(a)

    return [new_sentence, new_tags]

def remove_tags(row, tags_to_remove):
    tags_with_prefix = [f"B-{item}"  for item in tags_to_remove]
    tags_with_prefix.extend([f"I-{item}"  for item in tags_to_remove])
    new_sentence = []
    new_tags = []

    for a, t in zip(row["sentence"], row["tags"]):
        if t not in tags_with_prefix:
            new_tags.append(t)
            new_sentence.append(a)

    return [new_sentence, new_tags]

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None

    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        else:
            # Special token
            new_labels.append(-100)
    return new_labels

def tokenize_and_align_labels(tokenized_inputs, tags, tag2id):
    all_labels = [[tag2id[tag] for tag in doc] for doc in tags]
    
    new_labels = []
    for i, labels in enumerate(all_labels):
        
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def get_first_occurrence_indices(lst):
    indices = {}
    result = []
    for i, element in enumerate(lst):
        if element not in indices and element is not None:
            indices[element] = i
            result.append(i)
    return result

def class_metrics(row, class_tag):
    TP = 0
    FP = 0
    FN = 0
    for gt, pred in zip(row["class"], row["class_preds"]):
        if gt not in ["SoftSep", "HardSep"]:
            if gt == class_tag:
                if pred == class_tag:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred == class_tag:
                    FP += 1
    if (TP+FP) == 0:
        return None
    else:
        precision_score = TP / (TP+FP)
    if (TP+FN) == 0:
        return None
    else:
        recall_score = TP / (TP+FN)
    
    if (precision_score + recall_score) > 0:
        return 2 * precision_score * recall_score / (precision_score + recall_score)
    else:
        return None
    


def compact_tags(tags):
    c_tags = []
    previous = None
    for t in tags:
        if t != previous or previous == "CountryCode":
            c_tags.append(t)
        previous = t
    return c_tags


def compact_tags(tags):
    c_tags = []
    previous = None
    for t in tags:
        if t != previous or previous == "CountryCode":
            c_tags.append(t)
        previous = t
    return c_tags

def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def tag_prefix(tags, mode="prod"):
    if len(tags) == 0:
        return tags

    new_tags = ["B-" + tags[0]]
    for i in range(1, len(tags)):
        if mode == "prod":
            cond = tags[i] == tags[i-1] and tags[i] != "CountryCode"
        elif mode == "synth":
            cond = tags[i] == tags[i-1] or (i > 1 and tags[i-1] in ["OOA"] and tags[i] == tags[i-2])
        else:
            raise AttributeError

        if cond:
            new_tags.append("I-" + tags[i])
        else:
            new_tags.append("B-" + tags[i])

    return new_tags

