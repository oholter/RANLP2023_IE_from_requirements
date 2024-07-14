# RANLP2023 Reading Between the Lines: Information Extraction from Textual Requirements

This repository contains the code used to run the experiments in the paper:
[Reading Between the Lines: Information Extraction from Textual Requirements](https://aclanthology.org/2023.ranlp-1.76.pdf),
presented at RANLP23.

Note: You need requirement sentences extracted from PDF documents to conduct the experiments. In addition, you need to manually label the sentences with SCOPE, CONDITION, and DEMAND. Tools to extract requirements from PDF documents and convert them into JSON/JSONL are found in the [req_extractor library](https://github.com/oholter/req_extractor). For the experiments in the paper, I used Prodigy to annotate the sentences, once with context and once without context as described in the paper. Each sentence should also be annotated manually with the textual representation of the scope, the condition and the demand by adding, for example ``"scopes" : ["equipment"], "conditions" : [], "demands" : ["corrosion protection"]`` to each sentence in the resulting JSONL file.

As of June 2024, the documents used in the paper can be downloaded from DNV at https://www.veracity.com/.

## Setup the environment

1. Install the requirements in ``requirements.txt``
``python -m pip install -r requirements.txt``

2. Download Spacy and NLTK resources:
`python -m spacy download en_core_web_sm`
`>>> import nltk`
`>>> nltk.download('punkt')`
`>>> nltk.download('wordnet')`



## Experiments: Sequence Labelling

You will most likely want to use GPUs for training and testing as it can take a long time on a CPU.

### Train one model:
```
python -m sequence_labelling.runner [OPTIONS]

OPTIONS:
--epochs (-e) (INT): number of epochs
--save (TEXT): Path to save the model file
--train (TEXT): Path to train data
--test (TEXT): Path to test data
--lr (FLOAT): Learning rate
--eps (FLOAT): EPS
--full_finetuning (BOOL): Fine-tune the Bert embeddings
--test_size (FLOAT): the size of the test split
--model (TEXT): which model to use ([=roberta] in the experiments)
--hidden (INT): size of the hidden layer
--graph (-g) (TEXT): path to save loss/f-score graph
--preprocess (BOOL): case normalization and lemmatization
--batch_size (-b) (INT): batch size
```

### Reproduce the experiments using one model:
``./run_experiments.sh``
You may have to change the paths to the annotated documents in ``run_experiments.sh``.


### n-fold validation experiment
This will create the experiment structure for an n-fold validation experiment
Note: Remember to do the experiments both with context and without context.
`python -m evaluation.nfold [--input INPUT -n N --output OUTPUT]`


#### Train n models
``./evaluation/train_all.sh [EXPERIMENTS_FOLDER]``

#### Evaluate n models
`./evaluation/evaluate_all.sh [EXPERIMENTS_FOLDER] [true|false]`

#### Aggregate the scores
`python -m evaluation.aggregate [EXPERIMENTS_FOLDER]`

#### Print latex macros
``./evaluation/print_latex [EXPERIMENTS_FOLDER]``





## Experiments with GPT-3
In the gpt folder, change the settings in the ``config.json`` file. You need to supply the OpenAI API key.

### Test on one or a number of sentences
To run the experiment on the requirement with id=1 and id=2:
``python -m gpt.runner_one -i 1,2``


### Run all the experiments
``python -m gpt.runner``

