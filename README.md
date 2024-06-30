# RANLP2023 Reading Between the Lines: Information Extraction from Textual Requirements

This repository contains the code used to run the experiments in the paper: Reading Between the Lines: Information Extraction from Textual Requirements, presented at RANLP23.

Note: You need requirement sentences extracted from PDF documents to conduct the experiments. In addition, you need to manually label the sentences with Scope, Condition, and Demand. Tools to extract requirements from PDF documents and convert them into JSON/JSONL are found in the [req_extractor library](https://github.com/oholter/req_extractor).

## Setup the environment

1. Install the requirements in ``requirements.txt``  
``python -m pip install -r requirements.txt``

2. Download Spacy and NLTK resources:  
`python -m spacy download en_core_web_sm`

`>>> import nltk`
`>>> nltk.download('punkt')`
`>>> nltk.download('wordnet')`

## Training and experiments

Train one model:
`python -m concat_ner.runner [args]`

To reproduce the experiments with one model run:
``./run_experiments.sh``


## Do n-fold validation (train n models)
`python -m evaluation.nfold [--input INPUT -n N --output OUTPUT]`

Remember to do all of this both with context and without context !


## Train n models
``./evaluation/train_all.sh [EXPERIMENTS_FOLDER]``

## Evaluate n models
`./evaluation/evaluate_all.sh [EXPERIMENTS_FOLDER] [true|false]`

## Aggregate the scores
`python -m evaluation.aggregate [EXPERIMENTS_FOLDER]`

## Print latex macros
`./evaluation/print_latex [EXPERIMENTS_FOLDER]``
``
