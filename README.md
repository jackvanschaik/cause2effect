# cause2effect
Pointwise Causal Information: Autoencoders for Causal Strength Detection from Text

# Files

## `01_causenet_prep.Rmd`

Does all the parsing, preparation, and filtering of the CauseNet data.

## `02_word2vec_training.ipynb`

Trains cause2effect-SGNS with GenSim

## `03_cause2effect_null.R`

Trains cause2effect-NLL with R Torch

## `04_semeval_prep.py`

Normalization of tokenized semeval data for evaluation

## `05_vector_preparation.Rmd`

Code for preparing pretrained word2vec vectors and PCA based vectors

## `06_strength_evaluation.Rmd`

PCI* strength evaluation via binary classification task and qualitative evaluation

## `07_vector_evaluation`

Code for downstream evaluation task on SemEval

## `08_paper_figures.Rmd`

Code for making first plot: PCI histogram with tables of values