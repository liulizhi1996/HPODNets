#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import torch as t
import torch.nn as nn
from torch import optim
from sklearn.metrics import roc_auc_score, average_precision_score
from src.method.utils import PPMI_matrix
from src.method.model import Model

t.cuda.set_device(0)


def train(model, optimizer, train_data):
    train_index = train_data["train_index"]
    test_index = train_data["test_index"]

    num_protein, num_term = train_data["train_target"].shape
    num_pos = t.sum(train_data["train_target"], dim=0)
    num_neg = num_protein * t.ones_like(num_pos) - num_pos
    pos_weight = num_neg / (num_pos + 1e-5)
    classify_criterion = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)

    label_index = np.where(np.any(train_data["test_target"], 0))[0]
    y_true = np.take(train_data["test_target"], label_index, axis=1)

    def train_epoch():
        model.train()
        optimizer.zero_grad()
        score = model(train_data["feature"], train_data["network"])
        loss = classify_criterion(score[train_index], train_data["train_target"])
        loss.backward()
        optimizer.step()
        return loss.item()

    def test_epoch():
        model.eval()
        with t.no_grad():
            score = t.sigmoid(model(train_data["feature"], train_data["network"]))
            score = score.cpu().detach().numpy()
            y_score = np.take(score[test_index], label_index, axis=1)
            test_auc = roc_auc_score(y_true, y_score, average='macro')
            test_aupr = average_precision_score(y_true, y_score, average='macro')
            return test_auc, test_aupr

    for epoch in range(300):
        trn_loss = train_epoch()
        if epoch % 25 == 0:
            tst_auc, tst_aupr = test_epoch()
        else:
            tst_auc, tst_aupr = 0, 0
        print("Epoch", epoch, "\t", trn_loss, "\t", tst_auc, "\t", tst_aupr)


def test(model, feat, net):
    model.eval()
    with t.no_grad():
        score = t.sigmoid(model(feat, net))
        score = score.cpu().detach().numpy().astype(float)
        return score


if __name__ == "__main__":
    with open("../../config/method/main_temporal.json") as fp:
        config = json.load(fp)

    # load HPO annotations
    with open(config["dataset"], 'rb') as fp:
        dataset = pickle.load(fp)
    # calculate the frequency of HPO terms
    term_freq = dataset["annotation"].sum(axis=0)
    # list of HPO terms whose frequencies > 10
    # here we discard low-frequency terms
    term_list = term_freq[term_freq > 10].index.tolist()
    # only keep the annotations of HPO terms with frequencies <= 10
    full_annotation = dataset["annotation"][term_list]
    # list of proteins
    protein_list = list(full_annotation.index)
    # convert dataframe to numpy array
    full_annotation = full_annotation.values

    # load protein-protein similarity networks
    # format: { protein1: { protein2: score2, protein3: score3, ... }, ... }
    networks = list()
    for path in config["network"]:
        with open(path) as fp:
            ppi = json.load(fp)
        # only select sub-graph with vertex in proteins
        ppi = pd.DataFrame(ppi).fillna(0).reindex(
            columns=protein_list, index=protein_list, fill_value=0).values
        # compute negative half power of degree matrix which is a diagonal matrix
        diag = 1 / np.sqrt(np.sum(ppi, 1))
        diag[diag == np.inf] = 0  # if row-sum is 0 (isolated vertex in graph), then let to 0
        neg_half_power_degree_matrix = np.diag(diag)
        # construct normalized similarity network
        normalized_ppi = np.matmul(np.matmul(neg_half_power_degree_matrix, ppi),
                                   neg_half_power_degree_matrix)
        networks.append(normalized_ppi)

    protein_features = [PPMI_matrix(net) for net in networks]

    if config["mode"] == "cv":
        for fold in range(5):
            print("Fold", fold)

            train_mask = dataset["mask"][fold]["train"].reindex(
                index=protein_list, columns=term_list, fill_value=0).values
            test_mask = dataset["mask"][fold]["test"].reindex(
                index=protein_list, columns=term_list, fill_value=0).values
            train_annotation = full_annotation * train_mask

            train_protein_index = np.where(train_mask.any(1))[0]
            test_protein_index = np.where(test_mask.any(1))[0]
            train_target = full_annotation[train_protein_index]
            test_target = full_annotation[test_protein_index]

            train_data = {
                "train_target": t.FloatTensor(train_target).cuda(),
                "test_target": test_target,
                "feature": t.stack([t.FloatTensor(feat) for feat in protein_features]).cuda(),
                "network": t.stack([t.FloatTensor(net) for net in networks]).cuda(),
                "train_index": train_protein_index,
                "test_index": test_protein_index
            }

            model = Model(train_annotation.shape[0], train_annotation.shape[1], len(networks)).cuda()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            train(model, optimizer, train_data)
            pred_Y = test(model, train_data["feature"], train_data["network"])

            prediction = defaultdict(dict)
            for term_id, term in enumerate(term_list):
                prot_idx = np.where(test_mask[:, term_id] == 1)[0]
                y_pred = pred_Y[prot_idx, term_id]
                for i in range(len(y_pred)):
                    prediction[term][protein_list[prot_idx[i]]] = y_pred[i]

            with open(config["result"].format(fold), 'w') as fp:
                json.dump(prediction, fp, indent=2)
    else:
        train_mask = dataset["mask"]["train"].reindex(
            index=protein_list, columns=term_list, fill_value=0).values
        test_mask = dataset["mask"]["test"].reindex(
            index=protein_list, columns=term_list, fill_value=0).values
        train_annotation = full_annotation * train_mask

        train_protein_index = np.where(train_mask.any(1))[0]
        test_protein_index = np.where(test_mask.any(1))[0]
        train_target = full_annotation[train_protein_index]
        test_target = full_annotation[test_protein_index]

        train_data = {
            "train_target": t.FloatTensor(train_target).cuda(),
            "test_target": test_target,
            "feature": t.stack([t.FloatTensor(feat) for feat in protein_features]).cuda(),
            "network": t.stack([t.FloatTensor(net) for net in networks]).cuda(),
            "train_index": train_protein_index,
            "test_index": test_protein_index
        }

        model = Model(train_annotation.shape[0], train_annotation.shape[1], len(networks)).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, train_data)
        pred_Y = test(model, train_data["feature"], train_data["network"])

        prediction = defaultdict(dict)
        for term_id, term in enumerate(term_list):
            prot_idx = np.where(test_mask[:, term_id] == 1)[0]
            y_pred = pred_Y[prot_idx, term_id]
            for i in range(len(y_pred)):
                prediction[term][protein_list[prot_idx[i]]] = y_pred[i]

        with open(config["result"], 'w') as fp:
            json.dump(prediction, fp, indent=2)
