#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import pickle
from multiprocessing import Pool
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# number of processes
num_worker = 20


def f1_score_threshold(y_true, y_score, n_split=100):
    threshold = np.linspace(np.min(y_score), np.max(y_score), n_split)
    f1 = np.max([f1_score(y_true, y_score >= theta) for theta in threshold])
    return f1


def evaluate_term(y_true, y_score, term):
    auc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    f1 = f1_score_threshold(y_true, y_score)
    return {"term": term, "auc": auc, "aupr": aupr, "f1": f1}


if __name__ == "__main__":
    with open("../../config/utils/temporal/evaluation_marco_temporal_HPODNets.json") as fp:
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

                removed_terms = set()
                res_ids = list()
                pool = Pool(num_worker)
                for term in term_list:
                    y_mask = test_mask[[term]].values.flatten()
                    y_true = test_annotation[[term]].values.flatten()[y_mask == 1]
                    if len(np.unique(y_true)) < 2:
                        removed_terms.add(term)
                        continue
                    y_score = result[[term]].values.flatten()[y_mask == 1]

                    res = pool.apply_async(
                        evaluate_term,
                        args=(y_true, y_score, term)
                    )
                    res_ids.append(res)
                perf = [res_id.get() for res_id in res_ids]

                auc = pd.Series({x["term"]: x["auc"] for x in perf})
                aupr = pd.Series({x["term"]: x["aupr"] for x in perf})
                f1 = pd.Series({x["term"]: x["f1"] for x in perf})

                # groups of HPO terms
                uncommon = list(set(term_freq[(11 <= term_freq) & (term_freq <= 30)].index.tolist()) - removed_terms)
                common = list(set(term_freq[(31 <= term_freq) & (term_freq <= 100)].index.tolist()) - removed_terms)
                very_common = list(set(term_freq[(101 <= term_freq) & (term_freq <= 300)].index.tolist()) - removed_terms)
                extremely_common = list(set(term_freq[term_freq >= 301].index.tolist()) - removed_terms)

                # average on each group
                auc_uncommon = auc[uncommon].mean()
                aupr_uncommon = aupr[uncommon].mean()
                f1_uncommon = f1[uncommon].mean()

                auc_common = auc[common].mean()
                aupr_common = aupr[common].mean()
                f1_common = f1[common].mean()

                auc_very_common = auc[very_common].mean()
                aupr_very_common = aupr[very_common].mean()
                f1_very_common = f1[very_common].mean()

                auc_extremely_common = auc[extremely_common].mean()
                aupr_extremely_common = aupr[extremely_common].mean()
                f1_extremely_common = f1[extremely_common].mean()

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

            removed_terms = set()
            res_ids = list()
            pool = Pool(num_worker)
            for term in term_list:
                y_mask = test_mask[[term]].values.flatten()
                y_true = test_annotation[[term]].values.flatten()[y_mask == 1]
                if len(np.unique(y_true)) < 2:
                    removed_terms.add(term)
                    continue
                y_score = result[[term]].values.flatten()[y_mask == 1]

                res = pool.apply_async(
                    evaluate_term,
                    args=(y_true, y_score, term)
                )
                res_ids.append(res)
            perf = [res_id.get() for res_id in res_ids]

            auc = pd.Series({x["term"]: x["auc"] for x in perf})
            aupr = pd.Series({x["term"]: x["aupr"] for x in perf})
            f1 = pd.Series({x["term"]: x["f1"] for x in perf})

            # groups of HPO terms
            uncommon = list(set(term_freq[(11 <= term_freq) & (term_freq <= 30)].index.tolist()) - removed_terms)
            common = list(set(term_freq[(31 <= term_freq) & (term_freq <= 100)].index.tolist()) - removed_terms)
            very_common = list(set(term_freq[(101 <= term_freq) & (term_freq <= 300)].index.tolist()) - removed_terms)
            extremely_common = list(set(term_freq[term_freq >= 301].index.tolist()) - removed_terms)

            # average on each group
            auc_uncommon = auc[uncommon].mean()
            aupr_uncommon = aupr[uncommon].mean()
            f1_uncommon = f1[uncommon].mean()

            auc_common = auc[common].mean()
            aupr_common = aupr[common].mean()
            f1_common = f1[common].mean()

            auc_very_common = auc[very_common].mean()
            aupr_very_common = aupr[very_common].mean()
            f1_very_common = f1[very_common].mean()

            auc_extremely_common = auc[extremely_common].mean()
            aupr_extremely_common = aupr[extremely_common].mean()
            f1_extremely_common = f1[extremely_common].mean()

            print("AUC: %.4lf\t%.4lf\t%.4lf\t%.4lf" %
                  (auc_uncommon, auc_common, auc_very_common, auc_extremely_common))
            print("AUPR: %.4lf\t%.4lf\t%.4lf\t%.4lf" %
                  (aupr_uncommon, aupr_common, aupr_very_common, aupr_extremely_common))
            print("F1: %.4lf\t%.4lf\t%.4lf\t%.4lf" %
                  (f1_uncommon, f1_common, f1_very_common, f1_extremely_common))
            print()
