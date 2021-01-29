#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract entrez gene ids from 
ALL_SOURCES_ALL_FREQUENCIES_phenotype_to_genes.txt.

Download gene annotations file from 
    http://compbio.charite.de/jenkins/job/hpo.annotations.monthly/
    ALL_SOURCES_ALL_FREQUENCIES_phenotype_to_genes.txt
The last second column of file is what we want.
Extract and save them into a txt file in one column. This file will be uploaded
to Uniprot ID Mapping Tool (http://www.uniprot.org/mapping/) to get gene2uniprot
mapping file.
"""
import json


with open("../../config/preprocessing/extract_gene_id.json") as fp:
    config = json.load(fp)

gene_set = set()
with open(config["anno_file"]) as fp:
    for line in fp:
        if line.startswith("#"):    # pass the header
            continue
        gene = line.strip().split('\t')[-2]
        gene_set.add(gene)

with open(config["gene_list"], "w") as fp:
    for gene in gene_set:
        fp.write("%s\n" % gene)
