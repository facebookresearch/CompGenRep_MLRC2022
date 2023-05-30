# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
#!/bin/bash
# This script trains an NQG instance and evaluate its performance
export base_dir=${BASE_DIR}/baseline_replication/TMCD
export model_dir=${BASE_DIR}/trained_models
export data_dir=$base_dir/data
export dataset_name='geoquery'
export split='template'

## NQG specific params
export BERT_DIR="${BASE_DIR}/trained_models/BERT/"
export TRAIN_TSV="${base_dir}/data/${dataset_name}/${split}_split/train.tsv"
export TEST_TSV="${base_dir}/data/${dataset_name}/${split}_split/test.tsv"
export RULES="${base_dir}/data/${dataset_name}/${split}_split/rules.txt"
export TF_EXAMPLES="${base_dir}/data/${dataset_name}/${split}_split/tf_samples"
export CONFIG="${base_dir}/model/parser/configs/${dataset_name}_config.json"
export MODEL_DIR="${base_dir}/trained_models/${dataset_name}/nqg_${split}"

export NQG_TRAIN_TSV="${base_dir}/data/${dataset_name}/${split}_split/nqg_train.tsv"
export NQG_TEST_TSV="${base_dir}/data/${dataset_name}/${split}_split/nqg_test.tsv"
export SPIDER_TABLES=${data_dir}/spider/tables.json
export target_grammar="${base_dir}/model/parser/inference/targets/spider_grammar.txt"

if [[ $dataset_name == *"geoquery"* ]]
then
    export sample_size=0
    export terminal_codelength=8
    export allow_repeated_target_nts=false
    export target_grammar="${base_dir}/model/parser/inference/targets/funql.txt"
    if [[ $split == *"template"* ]]
    then
        export CONFIG="${base_dir}/model/parser/configs/geoquery_xl_config.json"
    fi
elif [[ $dataset_name == *"SCAN"* ]]
then
    export sample_size=500
    export terminal_codelength=32
    export allow_repeated_target_nts=true
    export CONFIG="${base_dir}/model/parser/configs/scan_config.json"
elif [[ $dataset_name == *"spider"* ]]
then
    export sample_size=1000
    export terminal_codelength=8
    export allow_repeated_target_nts=true
elif [[ $dataset_name == *"COGS"* ]]
then
    export sample_size=1000
    export terminal_codelength=8
    export allow_repeated_target_nts=true
    export CONFIG="${base_dir}/model/parser/configs/COGS_config.json"
fi


cd $base_dir

if [[ $dataset_name == *"spider"* ]]
then
    # Run the converting command
    # Note that for Spider, the script tasks/spider/nqg_preprocess.py should be run on the dataset TSV file to prepare the input for the space separated tokenization used by NQG.
    python tasks/spider/nqg_preprocess.py --input=$TRAIN_TSV --output=$NQG_TRAIN_TSV
    python tasks/spider/nqg_preprocess.py --input=$TEST_TSV --output=$NQG_TEST_TSV

    python model/induction/induce_rules.py  \
      --input=${NQG_TRAIN_TSV} \
      --output=${RULES} \
      --sample_size=${sample_size} \
      --terminal_codelength=${terminal_codelength} \
      --allow_repeated_target_nts=${allow_repeated_target_nts}

    echo "Starting to write TF examples"
    python model/parser/data/write_examples.py \
      --input=${NQG_TRAIN_TSV} \
      --output=${TF_EXAMPLES} \
      --config=${CONFIG} \
      --rules=${RULES} \
      --bert_dir=${BERT_DIR}

    # Generate SPIDER CFG target grammar
    python model/parser/inference/targets/generate_spider_grammars.py \
      --spider_tables=${SPIDER_TABLES}  \
      --output=${target_grammar}

else
    python model/induction/induce_rules.py  \
      --input=${TRAIN_TSV} \
      --output=${RULES} \
      --sample_size=${sample_size} \
      --terminal_codelength=${terminal_codelength} \
      --allow_repeated_target_nts=${allow_repeated_target_nts}

    python model/parser/data/write_examples.py \
      --input=${TRAIN_TSV} \
      --output=${TF_EXAMPLES} \
      --config=${CONFIG} \
      --rules=${RULES} \
      --bert_dir=${BERT_DIR}
fi

python model/parser/training/train_model.py \
  --input=${TF_EXAMPLES} \
  --config=${CONFIG} \
  --model_dir=${MODEL_DIR} \
  --bert_dir=${BERT_DIR} \
  --init_bert_checkpoint=False \
  --use_gpu

if [[ $dataset_name == *"spider"* ]]
then
    python model/parser/inference/eval_model.py \
      --input=${NQG_TEST_TSV}  \
      --config=${CONFIG}   \
      --model_dir=${MODEL_DIR}   \
      --bert_dir=${BERT_DIR}   \
      --rules=${RULES}  \
      --target_grammar=${target_grammar}
else
    python model/parser/inference/eval_model.py \
      --input=${TEST_TSV}  \
      --config=${CONFIG}   \
      --model_dir=${MODEL_DIR}   \
      --bert_dir=${BERT_DIR}   \
      --rules=${RULES}  \
      --target_grammar=${target_grammar}
fi
