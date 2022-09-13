import numpy as np
import logging
import json
import sys
sys.path.append("../")
from ops import config 
from ops.config import parser
from os.path import join, exists

import random 
from random import choice


NLI_LABEL2ID = {k: i for i, k in enumerate(config.NLI_LABELS)}
# NLI_LABEL2ID = {'contradiction': 0, 'entailment': 1, 'neutral': 2, 'hidden': 1}
NLI_ID2LABEL = {i: k for i, k in enumerate(config.NLI_LABELS)}
NLI_LABEL2ID["hidden"] = NLI_LABEL2ID["entailment"]

FEVER_LABEL2ID = {k: i for i, k in enumerate(config.FEVER_LABELS)}
# FEVER_MAPS = {"REFUTES": 0, "SUPPORTS": 1, "NOT ENOUGH INFO": 2}

def load_hans(n_samples=None, filter_label=None, filter_subset=None):
    out = []

    if filter_label is not None and filter_subset is not None:
        logging.info("Loading hans subset: {}-{}...".format(filter_label, filter_subset))
    else:
        logging.info("Loading hans all...")

    src = join(config.HANS_SOURCE, "heuristics_evaluation_set.txt")
    
    with open(src, "r") as f:
        f.readline()
        lines = f.readlines()

    if n_samples is not None:
        lines = np.random.RandomState(16349 + n_samples).choice(lines, n_samples,
                                                                replace=False)

    for line in lines:
        parts = line.split("\t")
        label = parts[0]

        if filter_label is not None and filter_subset is not None:
            if label != filter_label or parts[-3] != filter_subset:
                continue

        if label == "non-entailment":
            label = 0
        elif label == "entailment":
            label = 1
        else:
            raise RuntimeError()
        s1, s2, pair_id = parts[5:8]
       
        out.append(config.PairExample(pair_id, s1, s2, label))
    return out


def load_mnli(mode="train", sample=None):
    if mode == "train":
        filename = join(config.MULTINLI_SOURCE, "train.tsv")
    elif mode == "match_dev":
        filename = join(config.MULTINLI_SOURCE, "dev_matched.tsv")
    elif mode == "mismatch_dev":
        filename = join(config.MULTINLI_SOURCE, "dev_mismatched.tsv")
        
    logging.info("Loading mnli " + mode)
    with open(filename) as f:
        f.readline()
        lines = f.readlines()

    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample, replace=False)
    
    out = []
    for line in lines:
        line = line.split("\t")
        s1 = line[8]
        s2 = line[9]
        out.append(
            config.PairExample(line[0], line[8], line[9], NLI_LABEL2ID[line[-1].rstrip()]))
    return out


def load_hans_subsets():
    src = join(config.HANS_SOURCE, "heuristics_evaluation_set.txt")
    
    hans_datasets = []
    labels = ["entailment", "non-entailment"]
    subsets = set()
    with open(src, "r") as f:
        for line in f.readlines()[1:]:
            line = line.split("\t")
            subsets.add(line[-3])
    subsets = [x for x in subsets]

    for label in labels:
        for subset in subsets:
            name = "hans_{}_{}".format(label, subset)
            examples = load_hans(filter_label=label, filter_subset=subset)
            hans_datasets.append((name, examples))

    return hans_datasets




####### -------------------------------------------------------FEVER---------------------------------------------------------------------  #######
def load_fever(data_dir=config.FEVER_SOURCE, mode="train", sample=None):
    if mode == "train":
        filename = join(data_dir, "fever.train.jsonl")
    elif mode == "dev":
        filename = join(data_dir, "fever.dev.jsonl")
    elif mode == "symmv2_dev":                ### sym v2
        filename = join(data_dir, "symmetric_v0.2/fever_symmetric_dev.jsonl")
    elif mode == "symmv2_test":              ### sym v2 test
        filename = join(data_dir, "symmetric_v0.2/fever_symmetric_test.jsonl")
    elif mode == "symmv1_generated":         ### sym v1 test
        filename = join(data_dir, "symmetric_v0.1/fever_symmetric_generated.jsonl")
    elif mode == "symmv1_full":              ### sym v1
        filename = join(data_dir, "symmetric_v0.1/fever_symmetric_full.jsonl")
    else:
        raise Exception("invalid split name")

    out = []
    logging.info("Loading jsonl from {}...".format(filename))
    with open(filename, 'r') as jsonl_file:
        for i, line in enumerate(jsonl_file):
            example = json.loads(line)

            if "unique_id" in example:
                id = example["unique_id"]
            else:
                id = example["id"]

            claim = example["claim"]
            try:
                evidence = example["evidence"]
                label = example["gold_label"]
            except:
                evidence = example["evidence_sentence"]
                label = example["label"]

            out.append(config.PairExample(id, claim, evidence, FEVER_LABEL2ID[label]))

    if sample:
        random.shuffle(out)
        out = out[:sample]

    return out


####### ----------------------------------------------------------------QQP------------------------------------------------------------  #######
def load_qqp_paws(data_dir=config.FEVER_SOURCE, mode="qqp_train"):
    if mode == "qqp_train":
        filename = join(config.QQP_PAWS_SOURCE, "qqp_train.tsv")
    elif mode == "qqp_dev":
        filename = join(config.QQP_PAWS_SOURCE, "qqp_dev.tsv")
    elif mode == "qqp_test":
        filename = join(config.QQP_PAWS_SOURCE, "qqp_test.tsv")
    elif mode == "paws_train":
        filename = join(config.QQP_PAWS_SOURCE, "paws_train.tsv")
    elif mode == "paws_devtest":
        filename = join(config.QQP_PAWS_SOURCE, "paws_devtest.tsv")


    with open(filename) as f:
        a = f.readline()
        lines = f.readlines()

    out = []
    if mode.startswith("paws"):   ##### id  q1  q2  label
        for line in lines:
            line = line.split("\t")
            out.append(config.PairExample(int(line[0]), line[1], line[2], int(line[-1])))
    else:    ##### startswith("qqp")   label  q1  q2   id
        for line in lines:
            line = line.split("\t")
            out.append(config.PairExample(int(line[-1]), line[1], line[2], int(line[0])))
    return out

