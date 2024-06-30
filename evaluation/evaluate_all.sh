#!/bin/bash

# This script runs all the tests and saves all the results as .json files
# $1 is the path with all the experiment folders
#
# must have loaded virtualenv before running this script

if [[ $# < 2 ]] ; then
        echo 'Error: No experiment folder provided or use context'
        echo 'Exiting'
            exit 1
fi


EXP_DIR=$1
DATA_PATH=~/src/req_annot
MODEL=model.bin
USE_CONTEXT=$2 # must be true to use context

if [ $USE_CONTEXT = "true" ]; then
    echo "Using context !"
else
    echo "Not using context !"
fi


for f in $EXP_DIR/*; do
        if [ -d "$f" ]; then
            echo $f

            # sequence experiments
            #python -m sequence_labelling.test $f/$MODEL $DATA_PATH/annotations_with_scd/test.jsonl -r $f/test.json || exit 1
            python -m sequence_labelling.test $f/$MODEL $f/test.jsonl -r $f/test.json || exit 1
            python -m sequence_labelling.test $f/$MODEL $DATA_PATH/annotations_test_documents/with_scope/dnv_os_e101_with_scope.jsonl -r $f/os.json || exit 1
            python -m sequence_labelling.test $f/$MODEL $DATA_PATH/annotations_test_documents/with_scope/ru_hslc_pt5_with_scope.jsonl --r $f/hs5.json || exit 1
            python -m sequence_labelling.test $f/$MODEL $DATA_PATH/annotations_test_documents/with_scope/ru_hslc_pt6_with_scope.jsonl -r $f/hs6.json || exit 1
            python -m sequence_labelling.test $f/$MODEL $DATA_PATH/annotations_test_documents/with_scope/ru-ou-0503_with_scope.jsonl -r $f/ou.json || exit 1

            if [ $USE_CONTEXT = "true" ]; then
                # scope with context
                #python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_with_scd/test.jsonl scope -r $f/with_test_scope.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $f/test.jsonl scope -r $f/with_test_scope.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd/dnv_os_e101.jsonl scope -r $f/with_os_scope.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd/ru_hslc_pt5.jsonl scope -r $f/with_hs5_scope.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd/ru_hslc_pt6.jsonl scope -r $f/with_hs6_scope.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd/ru-ou-0503.jsonl scope -r $f/with_ou_scope.json || exit 1

                # condition with context
                #python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_with_scd/test.jsonl condition -r $f/with_test_condition.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $f/test.jsonl condition -r $f/with_test_condition.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd/dnv_os_e101.jsonl condition -r $f/with_os_condition.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd/ru_hslc_pt5.jsonl condition -r $f/with_hs5_condition.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd/ru_hslc_pt6.jsonl condition -r $f/with_hs6_condition.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd/ru-ou-0503.jsonl condition -r $f/with_ou_condition.json || exit 1

                # demand with context
                #python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_with_scd/test.jsonl demand -r $f/with_test_demand.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $f/test.jsonl demand -r $f/with_test_demand.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd/dnv_os_e101.jsonl demand -r $f/with_os_demand.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd/ru_hslc_pt5.jsonl demand -r $f/with_hs5_demand.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd/ru_hslc_pt6.jsonl demand -r $f/with_hs6_demand.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd/ru-ou-0503.jsonl demand -r $f/with_ou_demand.json || exit 1
            else
                # scope without context
                #python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_with_scd/test_without_context.jsonl scope -r $f/test_scope.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $f/test.jsonl scope -r $f/test_scope.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/dnv_os_e101.jsonl scope -r $f/os_scope.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru_hslc_pt5.jsonl scope -r $f/hs5_scope.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru_hslc_pt6.jsonl scope -r $f/hs6_scope.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru-ou-0503.jsonl scope -r $f/ou_scope.json || exit 1

                # condition without context
                #python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_with_scd/test_without_context.jsonl condition -r $f/test_condition.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $f/test.jsonl condition -r $f/test_condition.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/dnv_os_e101.jsonl condition -r $f/os_condition.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru_hslc_pt5.jsonl condition -r $f/hs5_condition.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru_hslc_pt6.jsonl condition -r $f/hs6_condition.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru-ou-0503.jsonl condition -r $f/ou_condition.json || exit 1

                # demand without context
                #python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_with_scd/test_without_context.jsonl demand -r $f/test_demand.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $f/test.jsonl demand -r $f/test_demand.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/dnv_os_e101.jsonl demand -r $f/os_demand.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru_hslc_pt5.jsonl demand -r $f/hs5_demand.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru_hslc_pt6.jsonl demand -r $f/hs6_demand.json || exit 1
                python -m sequence_labelling.test_scd $f/$MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru-ou-0503.jsonl demand -r $f/ou_demand.json || exit 1
            fi
        fi
done
