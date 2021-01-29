#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create HPO annotations from raw file.

Output format:
{
  hpo_term1: [ protein_id1, protein_id2, ... ],
  hpo_term2: [ protein_id1, protein_id2, ... ],
  ...
}
"""
import json
from collections import defaultdict
from src.utils.file_reader import gene2uniprot


with open("../../config/preprocessing/create_annotation.json") as fp:
    config = json.load(fp)

# load mapping of gene id to uniprot id
gene2protein = gene2uniprot(config["mapping"], gene_column=0, uniprot_column=1)

# load hpo annotations
annotation = defaultdict(list)
with open(config["raw_annotation"]) as fp:
    for line in fp:
        if line.startswith('#'):
            continue
        hpo_term, _, gene_id, *_ = line.strip().split('\t')
        for protein_id in gene2protein[gene_id]:
            annotation[hpo_term].append(protein_id)

# output annotation
with open(config["processed_annotation"], 'w') as fp:
    json.dump(annotation, fp, indent=2)
