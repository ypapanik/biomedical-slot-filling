# biomedical-slot-filling
Biomedical Slot filling with Dense Passage Retrieval
Accompanying code for our paper [1]. This code is only meant to be used so as to replicate our results.

## Data
The data, indices and BioSF slot filling dataset can be found [here](https://drive.google.com/drive/u/0/folders/1dwhfvl7zy6BEGhPWVAFBTOAg8YYd_jQz).

The BioSF dataset has been built using the following publicly available RE datasets:

- [Biocreative ChemProt](https://biocreative.bioinformatics.udel.edu/media/store/files/2017/ChemProt_Corpus.zip)
- [Biocreative DDI-2013](https://github.com/isegura/DDICorpus/blob/master/DDICorpus-2013.zip)
- [Biocreative CDR](https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus/)
## Models
Our finetuned models are available through huggingface:
- https://huggingface.co/healx/biomedical-dpr-qry-encoder
- https://huggingface.co/healx/biomedical-dpr-ctx-encoder
- https://huggingface.co/healx/biomedical-slot-filling-reader-base
- https://huggingface.co/healx/biomedical-slot-filling-reader-large

## Replicating results
Although we provide also the code to train a DPR or a reader model, we here focus only in replicating our evaluation experiments.
Assuming that we 've created a virtual environment and install requirements, we can then run either
```python
PYTHONPATH=. python biomedical_slot_filling/scripts/train_eval_reader --eval
```
or
```python
PYTHONPATH=. python biomedical_slot_filling/scripts/evaluate_retrieval
```
The above scripts will download the neccessary files from GDrive, the finetuned models from HF and perform the relevant evaluation.

To train a slot filling reader, we can run:
```python
PYTHONPATH=. python biomedical_slot_filling/scripts/train_eval_reader.py --model-name-or-path dmis-lab/biobert-base-cased-v1.2 --train
```

Note: A large portion of our code is adapted from the following two repositories:
- https://github.com/facebookresearch/DPR
- https://github.com/IBM/kgi-slot-filling


## Cite our work
[1] [Slot Filling for Biomedical Information Extraction](https://arxiv.org/abs/2109.08564)
