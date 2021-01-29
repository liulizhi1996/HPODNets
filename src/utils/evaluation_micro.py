#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def f1_score_threshold(y_true, y_score, n_split=100):
    threshold = np.linspace(np.min(y_score), np.max(y_score), n_split)
    f1 = np.max([f1_score(y_true, y_score >= theta) for theta in threshold])
    return f1


def evaluate_group(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    f1 = f1_score_threshold(y_true, y_score)
    return auc, aupr, f1


if __name__ == "__main__":
    with open("../../config/utils/temporal/evaluation_micro_temporal_HPODNets.json") as fp:
        config = json.load(fp)

    # load HPO annotations
    with open(config["dataset"], 'rb') as fp:
        dataset = pickle.load(fp)
    # list of proteins
    protein_list = list(dataset["annotation"].index)
    # list of HPO terms
    term_list = list(dataset["annotation"].columns)
    # full HPO annotations
    full_annotation = dataset["annotation"]

    # calculate the frequency of HPO terms
    term_freq = full_annotation.sum(axis=0)

    if config["mode"] == "cv":
        for method in config["result"]:
            print(method["name"], end='\n')

            for fold in range(5):
                test_mask = dataset["mask"][fold]["test"].reindex(
                    index=protein_list, columns=term_list, fill_value=0)
                test_annotation = test_mask * full_annotation

                with open(method["prediction"][fold]) as fp:
                    result = json.load(fp)
                result = pd.DataFrame(result).fillna(0).reindex(
                    index=protein_list, columns=term_list, fill_value=0
                )

                y_true_uncommon = np.zeros(0)
                y_true_common = np.zeros(0)
                y_true_very_common = np.zeros(0)
                y_true_extremely_common = np.zeros(0)
                y_score_uncommon = np.zeros(0)
                y_score_common = np.zeros(0)
                y_score_very_common = np.zeros(0)
                y_score_extremely_common = np.zeros(0)
                for term in term_list:
                    y_mask = test_mask[[term]].values.flatten()
                    y_true = test_annotation[[term]].values.flatten()[y_mask == 1]
                    if len(np.unique(y_true)) < 2:
                        continue
                    y_score = result[[term]].values.flatten()[y_mask == 1]
                    if 11 <= term_freq[term] <= 30:
                        y_true_uncommon = np.concatenate((y_true_uncommon, y_true))
                        y_score_uncommon = np.concatenate((y_score_uncommon, y_score))
                    elif 31 <= term_freq[term] <= 100:
                        y_true_common = np.concatenate((y_true_common, y_true))
                        y_score_common = np.concatenate((y_score_common, y_score))
                    elif 101 <= term_freq[term] <= 300:
                        y_true_very_common = np.concatenate((y_true_very_common, y_true))
                        y_score_very_common = np.concatenate((y_score_very_common, y_score))
                    elif term_freq[term] >= 301:
                        y_true_extremely_common = np.concatenate((y_true_extremely_common, y_true))
                        y_score_extremely_common = np.concatenate((y_score_extremely_common, y_score))

                auc_uncommon, aupr_uncommon, f1_uncommon = evaluate_group(y_true_uncommon, y_score_uncommon)
                auc_common, aupr_common, f1_common = evaluate_group(y_true_common, y_score_common)
                auc_very_common, aupr_very_common, f1_very_common = evaluate_group(y_true_very_common,
                                                                                   y_score_very_common)
                auc_extremely_common, aupr_extremely_common, f1_extremely_common = evaluate_group(y_true_extremely_common,
                                                                                                  y_score_extremely_common)

                print("Fold", fold)
                print("AUC: %.4lf\t%.4lf\t%.4lf\t%.4lf" %
                      (auc_uncommon, auc_common, auc_very_common, auc_extremely_common))
                print("AUPR: %.4lf\t%.4lf\t%.4lf\t%.4lf" %
                      (aupr_uncommon, aupr_common, aupr_very_common, aupr_extremely_common))
                print("F1: %.4lf\t%.4lf\t%.4lf\t%.4lf" %
                      (f1_uncommon, f1_common, f1_very_common, f1_extremely_common))
                print()
    else:
        for method in config["result"]:
            print(method["name"], end='\n')

            test_mask = dataset["mask"]["test"].reindex(
                index=protein_list, columns=term_list, fill_value=0)
            test_annotation = test_mask * full_annotation

            with open(method["prediction"]) as fp:
                result = json.load(fp)
            result = pd.DataFrame(result).fillna(0).reindex(
                index=protein_list, columns=term_list, fill_value=0
            )

            y_true_uncommon = np.zeros(0)
            y_true_common = np.zeros(0)
            y_true_very_common = np.zeros(0)
            y_true_extremely_common = np.zeros(0)
            y_score_uncommon = np.zeros(0)
            y_score_common = np.zeros(0)
            y_score_very_common = np.zeros(0)
            y_score_extremely_common = np.zeros(0)
            for term in term_list:
                y_mask = test_mask[[term]].values.flatten()
                y_true = test_annotation[[term]].values.flatten()[y_mask == 1]
                if len(np.unique(y_true)) < 2:
                    continue
                y_score = result[[term]].values.flatten()[y_mask == 1]
                if 11 <= term_freq[term] <= 30:
                    y_true_uncommon = np.concatenate((y_true_uncommon, y_true))
                    y_score_uncommon = np.concatenate((y_score_uncommon, y_score))
                elif 31 <= term_freq[term] <= 100:
                    y_true_common = np.concatenate((y_true_common, y_true))
                    y_score_common = np.concatenate((y_score_common, y_score))
                elif 101 <= term_freq[term] <= 300:
                    y_true_very_common = np.concatenate((y_true_very_common, y_true))
                    y_score_very_common = np.concatenate((y_score_very_common, y_score))
                elif term_freq[term] >= 301:
                    y_true_extremely_common = np.concatenate((y_true_extremely_common, y_true))
                    y_score_extremely_common = np.concatenate((y_score_extremely_common, y_score))

            auc_uncommon, aupr_uncommon, f1_uncommon = evaluate_group(y_true_uncommon, y_score_uncommon)
            auc_common, aupr_common, f1_common = evaluate_group(y_true_common, y_score_common)
            auc_very_common, aupr_very_common, f1_very_common = evaluate_group(y_true_very_common, y_score_very_common)
            auc_extremely_common, aupr_extremely_common, f1_extremely_common = evaluate_group(y_true_extremely_common,
                                                                                              y_score_extremely_common)

            print("AUC: %.4lf\t%.4lf\t%.4lf\t%.4lf" %
                  (auc_uncommon, auc_common, auc_very_common, auc_extremely_common))
            print("AUPR: %.4lf\t%.4lf\t%.4lf\t%.4lf" %
                  (aupr_uncommon, aupr_common, aupr_very_common, aupr_extremely_common))
            print("F1: %.4lf\t%.4lf\t%.4lf\t%.4lf" %
                  (f1_uncommon, f1_common, f1_very_common, f1_extremely_common))
            print()
