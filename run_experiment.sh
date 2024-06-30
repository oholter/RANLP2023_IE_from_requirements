#!/bin/bash

# This script runs all the tests and creates all the latex files
# $1 is supposed to be the path to the model to test
#
# must have loaded virtualenv before running this script

if [[ $# -eq 0 ]] ; then
        echo 'Error: No model provided'
        echo 'Exiting'
            exit 1
fi


MODEL=$1
DATA_PATH=~/src/req_annot



# sequence experiments
python -m sequence_labelling.test $MODEL $DATA_PATH/annotation_with_scd/test.jsonl --latex test.tex
python -m sequence_labelling.test $MODEL $DATA_PATH/annotations_test_documents/with_scope/dnv_os_e101_with_scope.jsonl --latex os.tex
python -m sequence_labelling.test $MODEL $DATA_PATH/annotations_test_documents/with_scope/ru_hslc_pt5_with_scope.jsonl --latex hs5.tex
python -m sequence_labelling.test $MODEL $DATA_PATH/annotations_test_documents/with_scope/ru_hslc_pt6_with_scope.jsonl --latex hs6.tex
python -m sequence_labelling.test $MODEL $DATA_PATH/annotations_test_documents/with_scope/ru-ou-0503_with_scope.jsonl --latex ou.tex

# scope without context
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_with_scd/test_without_context.jsonl scope --latex test_scope.tex --prefix test
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/dnv_os_e101.jsonl scope --latex os_scope.tex --prefix ose
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru_hslc_pt5.jsonl scope --latex hs5_scope.tex --prefix hsfive
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru_hslc_pt6.jsonl scope --latex hs6_scope.tex --prefix hssix
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru-ou-0503.jsonl scope --latex ou_scope.tex --prefix ou

# scope with context
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_with_scd/test.jsonl scope --latex with_test_scope.tex --prefix withTest
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd/dnv_os_e101.jsonl scope --latex with_os_scope.tex --prefix withOs
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd/ru_hslc_pt5.jsonl scope --latex with_hs5_scope.tex --prefix withHsfive
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd/ru_hslc_pt6.jsonl scope --latex with_hs6_scope.tex --prefix withHssix
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd/ru-ou-0503.jsonl scope --latex with_ou_scope.tex --prefix withOu

# condition without context
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_with_scd/test_without_context.jsonl condition --latex test_condition.tex --prefix test
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/dnv_os_e101.jsonl condition --latex os_condition.tex --prefix ose
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru_hslc_pt5.jsonl condition --latex hs5_condition.tex --prefix hsfive
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru_hslc_pt6.jsonl condition --latex hs6_condition.tex --prefix hssix
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru-ou-0503.jsonl condition --latex ou_condition.tex --prefix ou

# condition with context
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_with_scd/test.jsonl condition --latex with_test_condition.tex --prefix withTest
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd/dnv_os_e101.jsonl condition --latex with_os_condition.tex --prefix withOs
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd/ru_hslc_pt5.jsonl condition --latex with_hs5_condition.tex --prefix withHsfive
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd/ru_hslc_pt6.jsonl condition --latex with_hs6_condition.tex --prefix withHssix
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd/ru-ou-0503.jsonl condition --latex with_ou_condition.tex --prefix withOu

# demand without context
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_with_scd/test_without_context.jsonl demand --latex test_demand.tex --prefix test
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/dnv_os_e101.jsonl demand --latex os_demand.tex --prefix ose
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru_hslc_pt5.jsonl demand --latex hs5_demand.tex --prefix hsfive
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru_hslc_pt6.jsonl demand --latex hs6_demand.tex --prefix hssix
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd_without_context/ru-ou-0503.jsonl demand --latex ou_demand.tex --prefix ou

# demand with context
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_with_scd/test.jsonl demand --latex with_test_demand.tex --prefix withTest
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd/dnv_os_e101.jsonl demand --latex with_os_demand.tex --prefix withOs
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd/ru_hslc_pt5.jsonl demand --latex with_hs5_demand.tex --prefix withHsfive
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd/ru_hslc_pt6.jsonl demand --latex with_hs6_demand.tex --prefix withHssix
python -m sequence_labelling.test_scd $MODEL $DATA_PATH/annotations_test_documents/with_scd/ru-ou-0503.jsonl demand --latex with_ou_demand.tex --prefix withOu
