#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct HumanNet.

Download integrated network HumanNet-XN from
    https://www.inetbio.org/humannet/download.php
and download Entrez GeneID to UniProt Protein ID mapping file from
    https://www.uniprot.org/mapping/
"""
import json
from collections import defaultdict


def get_entrez_mapping(path_to_file):
    """Extract mapping from Entrez GeneID to UniProt Protein ID.
    :param path_to_file: path to the mapping file
        The Entrez GeneID to UniProt Protein ID mapping file is downloaded
        from https://www.uniprot.org/mapping/.
    :return: dict, key: Entrez GeneID, value: set of UniProt Protein IDs
        { gene1: { protein1a, protein1b, ... }, ... }
    """
    entrez2uniprot = defaultdict(set)
    with open(path_to_file) as fp:
        for line in fp:
            if line.startswith("Entry"):
                continue
            protein_id, gene_ids = line.split('\t')
            if len(gene_ids) == 0:
                continue
            else:
                id_list = gene_ids.strip().split(';')
                for gene_id in id_list:
                    if len(gene_id) > 0:
                        entrez2uniprot[gene_id].add(protein_id)
    return entrez2uniprot


def get_network(path_to_file, entrez2uniprot):
    """Construct HumanNet.
    :param path_to_file: path to the HumanNet-XN file, see
        https://www.inetbio.org/humannet/download.php
    :param entrez2uniprot: nested dict, like
        { protein1: { protein1a: score1a, protein1b: score1b, ... }, ... }
    :return:
    """
    network = defaultdict(dict)
    with open(path_to_file) as fp:
        for line in fp:
            if line.startswith('#'):
                continue
            gene_a, gene_b, score = line.strip().split('\t')
            if gene_a in entrez2uniprot and gene_b in entrez2uniprot:
                for protein_a in entrez2uniprot[gene_a]:
                    for protein_b in entrez2uniprot[gene_b]:
                        network[protein_a][protein_b] = float(score)
                        network[protein_b][protein_a] = float(score)
    return network


if __name__ == "__main__":
    with open("../../../config/feature/HumanNet/humannet.json") as fp:
        config = json.load(fp)

    # get Entrez GeneID to UniProt Protein ID mapping
    gene_mapping = get_entrez_mapping(config["entrez-uniprot"])
    # get network
    humannet = get_network(config["network"], gene_mapping)
    # write into file
    with open(config["output"], 'w') as fp:
        json.dump(humannet, fp, indent=2)
