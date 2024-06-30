#!/bin/bash

# This script runs the training for all n-fold experiments in folder $1
# must have loaded virtualenv before running this script


if [[ $# -eq 0 ]] ; then
        echo 'Error: No experiment folder provided'
        echo 'Exiting'
            exit 1
fi


EXP_DIR=$1
#cd $EXP_DIR

for f in $EXP_DIR/*; do
        if [ -d "$f" ]; then
            echo $f
            python -m concat_ner.runner \
                -e 4 \
                --save $f/model.bin \
                --train $f/train.jsonl \
                --test $f/test.jsonl \
                --lr 1e-5 \
                --full_finetuning \
                --test_size 0 \
                --model roberta \
                --batch_size 16 || exit 1
        fi
done

