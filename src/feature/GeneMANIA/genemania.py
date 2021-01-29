#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct GeneMANIA PPI network.

First, download the (latest) combined PPI network of human from
    http://genemania.org/data/current/Homo_sapiens.COMBINED/
    COMBINED.DEFAULT_NETWORKS.BP_COMBINING.txt
Then, download ID mappings file from
    http://genemania.org/data/archive/2017-03-12/Homo_sapiens/
    identifier_mappings.txt
This file will help to convert the gene ID to UniProt ID.
"""
import json
from collections import defaultdict


def ensg2uniprot(file_path):
    """Mapping Ensembl gene ID to UniProt accession.
    :param file_path: path to mapping file provide by GeneMANIA
        url: http://genemania.org/data/archive/2017-03-12/Homo_sapiens/
             identifier_mappings.txt
    :return: dict, key: Ensembl gene ID, value: UniProt accession
        { ensg_ac1: uniprot_ac1, ensg_ac2: uniprot_ac2, ... }
    """
    mapping = dict()
    with open(file_path) as fp:
        for line in fp:
            # pass the first line (table head)
            if line.startswith('Preferred_Name'):
                continue
            entries = line.strip().split('\t')
            if entries[-1] == 'Uniprot ID':
                ensg_ac = entries[0]
                uniprot_ac = entries[1]
                mapping[ensg_ac] = uniprot_ac
    return mapping


def get_genemania_network(path_to_network, path_to_mapping):
    """Construct GeneMANIA PPI network.
    :param path_to_network: path to GeneMANIA network data
        url: http://genemania.org/data/current/Homo_sapiens.COMBINED/
             COMBINED.DEFAULT_NETWORKS.BP_COMBINING.txt
    :param path_to_mapping: path to ID mapping file
        url: http://genemania.org/data/archive/2017-03-12/Homo_sapiens/
             identifier_mappings.txt
    :return: dict, PPI network
        { protein1: { protein1a: score1a, protein1b: score1b, ... },
          protein2: { protein2a: score2a, protein2b: score2b, ... },
          ... }
    """
    network = defaultdict(dict)
    mapping = ensg2uniprot(path_to_mapping)
    with open(path_to_network) as fp:
        for line in fp:
            # pass the first line (table head)
            if line.startswith("Gene_A"):
                continue
            ensg_ac1, ensg_ac2, score = line.strip().split()
            try:    # if no matched accession found, pass it
                protein1 = mapping[ensg_ac1]
                protein2 = mapping[ensg_ac2]
            except KeyError:
                continue
            score = float(score)
            network[protein1][protein2] = network[protein2][protein1] = score
    return network


if __name__ == "__main__":
    with open("../../../config/feature/GeneMANIA/genemania.json") as fp:
        config = json.load(fp)

    # get PPI network
    network = get_genemania_network(config["network"], config["mapping"])
    # write into file
    with open(config["feature"], 'w') as fp:
        json.dump(network, fp, indent=2)
