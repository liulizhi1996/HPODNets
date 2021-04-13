# HPODNets

**HPODNets: deep graph convolutional networks for predicting human protein-phenotype associations**

*Please run the programs in order.*

## Dependencies

Our model is implemented by Python 3.6 with Pytorch 1.4.0 and Pytorch-geometric 1.5.0, and run on Nvidia GPU with CUDA 10.0.

## Preprocessing

- `extract_gene_id.py`: First, please download gene annotations file from [http://compbio.charite.de/jenkins/job/hpo.annotations.monthly/](http://compbio.charite.de/jenkins/job/hpo.annotations.monthly/) with all sources and all frequencies: `ALL_SOURCES_ALL_FREQUENCIES_phenotype_to_genes.txt`. Then run the script, you will get a .txt file containing all gene ids. Finally, please upload this file to [http://www.uniprot.org/mapping/](http://www.uniprot.org/mapping/) to map Entrez Gene ID to UniProt ID.

- `create_annotation.py`: After generating Gene ID mapping file, you can run this script to generate HPO term-protein association file. The output json file contains associated proteins of each HPO term, like

	```
	{
	  hpo_term1: [ protein_id1, protein_id2, ... ],
	  hpo_term2: [ protein_id1, protein_id2, ... ],
	  ...
	}
	```

### Cross-validation

- `split_dataset_cv.py`: First, the script will remove the proteins that are added after a certain previous time. Then, only the HPO terms in PA sub-ontology (HP:0000118) will be left. Finally, the script split the protein set into five folds and generate the dataset pickle file like

	```
	store = {
		"annotation": full dataset
    	"mask": [
			{
				"train": training mask of 1st fold
				"test": test mask of 1st fold
			},
			...
		]
	}
	```

### Temporal validation

- `split_dataset_temporal.py`: We still focus on HPO term in PA sub-ontology. We use the proteins added before a certain previous time as the training set, and the proteins added after that time as the test set. The generated dataset pickle file is organized as

	```
    store = {
        "annotation": full dataset
        "mask": {
            "train": training mask
            "test": test mask
        }
    }
	```

## Feature Extraction
### STRING

- `string.py`: Please firstly open [https://string-db.org/cgi/download.pl](https://string-db.org/cgi/download.pl) and choose "organism" as "Homo sapiens", then download "9606.protein.links.v11.0.txt.gz" (version number may change). Meanwhile, download mapping file under "ACCESSORY DATA" category, or open website
[https://string-db.org/mapping\_files/uniprot\_mappings/](https://string-db.org/mapping_files/uniprot_mappings/) to download it. After downloading, you can run this code to get a json file containing PPI data organized as

	```
	{
		protein1: {
			protein1a: score1a, 
			protein1b: score1b, 
			...
		}, 
		...
	}
	```
Here the scores are scaled to [0, 1].


### GeneMANIA-Net

- `genemania.py`: First, please download `COMBINED.DEFAULT_NETWORKS.BP_COMBINING.txt` from [http://genemania.org/data/current/Homo\_sapiens.COMBINED/](http://genemania.org/data/current/Homo_sapiens.COMBINED/), and then download `identifier_mappings.txt` from [http://genemania.org/data/current/Homo_sapiens/](http://genemania.org/data/current/Homo_sapiens/). Run the `genemania.py` and you will obtain the PPI network as a json file.

### HumanNet

- `humannet.py`: First, open [https://www.inetbio.org/humannet/download.php](https://www.inetbio.org/humannet/download.php), and then click `HumanNet-XN` in the `Integrated Networks` section. Next, open [https://www.uniprot.org/uniprot/?query=*&fil=reviewed%3Ayes+AND+organism%3A%22Homo+sapiens+%28Human%29+%5B9606%5D%22](https://www.uniprot.org/uniprot/?query=*&fil=reviewed%3Ayes+AND+organism%3A%22Homo+sapiens+%28Human%29+%5B9606%5D%22). Click the `Columns` button in the middle of the page, and click the crosses in the upper right corner of all the dotted boxes in the `Columns to be displayed` of the new page. Then enter `GeneID` in the `Search:` search box in the middle of the `Add more columns` column, and click the associated word that pops up. At this point, click the `Save` button on the far right. Jump back to the original page, now only `Entry` and `Cross-reference (GeneID)` are left in the form. Click the `Download` button in the middle of the page, select `Format: Tab-separated`, and then click the `Go` button to download the file. Rename the file to `entrez2uniprot.txt` and place it under `data/feature/HumanNet/raw`. After the download is complete, in the UniProt page just now, click the `Columns` button again, in the new page, click `Reset to default` on the right side (*note: do not click the word default, but click Reset*), and then click ` Save`. Now, the UniProt interface is restored to its original appearance. Finally, run `humannet.py` and then get the processed PPI network in json format.

## Run the model

- `main.py`: Run the `main.py` script, and you will obtain the prediction results. Be careful to the cuda device ID!

## Evaluation

- `evaluation_macro.py` & `evaluation_micro.py`: There are two modes of evaluation: 1) macro-averaged; 2) micro-averaged. You can run the corresponding script to get the performance.


## Download the prediction results

We upload the prediction results of 15,033 proteins stored in STRING, GeneMANIA-Net and HumanNet databases that have not been annotated using the HPO annotations of 4,424 proteins released in October 2020 as the training set. The data is available at:

[https://doi.org/10.6084/m9.figshare.14222732](https://doi.org/10.6084/m9.figshare.14222732)

The file is organized in .json format, where the key is the HPO term, and the values are proteins and their corresponding predictive scores. The file is so large (1.98 GB), and you are free to download it.