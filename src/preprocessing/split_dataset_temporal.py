#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create training & test datasets.

After running this script, you will get
    - true labels of full dataset
    - masks indicating the training & test set
i.e.
    store = {
        "annotation": full dataset
        "mask": {
            "train": training mask
            "test": test mask
        }
    }
"""
import json
import pickle
from functools import reduce
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from src.utils.ontology import HumanPhenotypeOntology


with open("../../config/preprocessing/split_dataset_temporal.json") as fp:
    config = json.load(fp)

# load old HPO
old_ontology = HumanPhenotypeOntology(config["old_ontology"]["path"],
                                      version=config["old_ontology"]["version"])

# load old HPO annotations
with open(config["old_annotation"]) as fp:
    old_annotation = json.load(fp)

# leave only terms in PA
filter_old_annotation = {term: old_annotation[term]
                         for term in old_annotation
                         if old_ontology[term].ns == 'pa'}

# set of proteins in old annotations
old_proteins = set(reduce(lambda a, b: set(a) | set(b),
                          filter_old_annotation.values()))

# load new HPO
new_ontology = HumanPhenotypeOntology(config["new_ontology"]["path"],
                                      version=config["new_ontology"]["version"])

# load new HPO annotations
with open(config["new_annotation"]) as fp:
    new_annotation = json.load(fp)

# leave only terms in PA
filter_new_annotation = {term: new_annotation[term]
                         for term in new_annotation
                         if new_ontology[term].ns == 'pa'}

# adapt new annotations to old ones
adapted_new_annotation = dict()
for term in filter_new_annotation:
    # if "veteran" HPO term, just copy down
    if term in old_ontology:
        adapted_new_annotation[term] = filter_new_annotation[term]
    # else if has old, alternative HPO terms, replace it
    elif term in new_ontology.alt_ids:
        for alternative in new_ontology.alt_ids[term]:
            if alternative in old_ontology:
                print("Replace %s to %s" % (term, alternative))
                adapted_new_annotation[alternative] = filter_new_annotation[term]
            else:
                print("Though %s can be replaced by %s, %s is not in old ontology" % (term, alternative, alternative))
    # if not found, then discard
    else:
        print("Discard %s" % term)

# set of proteins in new annotations
new_proteins = set(reduce(lambda a, b: set(a) | set(b),
                          adapted_new_annotation.values()))

# proteins in the training set
train_proteins = old_proteins
# proteins in the test set
test_proteins = new_proteins - old_proteins

combined_annotation = defaultdict(set)
for term in filter_old_annotation:
    for protein in filter_old_annotation[term]:
        if protein in train_proteins:
            combined_annotation[term].add(protein)
for term in adapted_new_annotation:
    for protein in adapted_new_annotation[term]:
        if protein in test_proteins:
            combined_annotation[term].add(protein)

# list of HPO terms
term_list = list(combined_annotation.keys())
# list of proteins
protein_list = list(reduce(lambda a, b: set(a) | set(b),
                           combined_annotation.values()))

# transform HPO annotations to DataFrame like
#           term1   term2   term3
# protein1      1       0       1
# protein2      0       1       0
# protein3      0       0       1
mlb = MultiLabelBinarizer()
df_combined_annotation = pd.DataFrame(mlb.fit_transform(combined_annotation.values()),
                                      columns=mlb.classes_,
                                      index=combined_annotation.keys()).reindex(
                                      columns=protein_list, index=term_list, fill_value=0).transpose()

# object to be stored
# we first insert labels into object
store = {"annotation": df_combined_annotation}

# make up train & test dataset masks for each fold
train_dataset = dict()
test_dataset = dict()
for term in combined_annotation:
    test_dataset[term] = test_proteins
    train_dataset[term] = train_proteins

# transform train & test masks to DataFrame
store["mask"] = dict()
mlb = MultiLabelBinarizer()
df_train_dataset = pd.DataFrame(mlb.fit_transform(train_dataset.values()),
                                columns=mlb.classes_,
                                index=train_dataset.keys()).reindex(
                                columns=protein_list, index=term_list, fill_value=0).transpose()

mlb = MultiLabelBinarizer()
df_test_dataset = pd.DataFrame(mlb.fit_transform(test_dataset.values()),
                               columns=mlb.classes_,
                               index=test_dataset.keys()).reindex(
                               columns=protein_list, index=term_list, fill_value=0).transpose()

store["mask"]["train"] = df_train_dataset
store["mask"]["test"] = df_test_dataset

# write to pickle file
with open(config["dataset"], "wb") as fp:
    pickle.dump(store, fp)
