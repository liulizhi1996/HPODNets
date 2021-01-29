#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Readers of files with different formats.
"""
import json
from collections import defaultdict
from src.utils.ontology import get_root, get_subontology


def gene2uniprot(file_path, gene_column, uniprot_column):
    """Mapping entrez gene id to uniprot protein id.

    :param file_path: path to mapping file
    :param gene_column: the column index of gene id
    :param uniprot_column: the column index of uniprot id
    :return: a dict with key being gene id and value being list of uniprot ids
    { gene_id: [uniprot_id1, uniprot_id2, ...] }
    """
    gene_to_protein = defaultdict(list)
    with open(file_path) as fp:
        for line in fp:
            if line.startswith("y"):    # omit the header line
                continue
            entries = line.strip().split('\t')
            # multi-genes mapped to the same protein
            if ',' in entries[gene_column]:
                genes = entries[gene_column].split(',')
                protein = entries[uniprot_column]
                for gene in genes:
                    gene_to_protein[gene].append(protein)
            # one gene mapped to one protein
            else:
                gene, protein = entries[gene_column], entries[uniprot_column]
                gene_to_protein[gene].append(protein)
    return gene_to_protein
