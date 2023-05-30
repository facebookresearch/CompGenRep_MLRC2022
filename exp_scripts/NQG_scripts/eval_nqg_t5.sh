# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
#!/bin/bash
# Evaluate the ensemble, NQG-T5
export dataset_name='geoquery'
export split='standard'

export base_dir=${BASE_DIR}/baseline_replication/TMCD
export model_dir=${BASE_DIR}/trained_models
export data_dir=$base_dir/data
export pred_dir=${BASE_DIR}/results/predictions

## NQG specific params
export TEST_TSV="${base_dir}/data/${dataset_name}/${split}_split/test.tsv"
export RULES="${base_dir}/data/${dataset_name}/${split}_split/rules.txt"
export BERT_DIR="${base_dir}/trained_models/BERT/"
export TF_EXAMPLES="${base_dir}/data/${dataset_name}/${split}_split/tf_samples"
export CONFIG="${base_dir}/model/parser/configs/${dataset_name}_config.json"
export MODEL_DIR="${model_dir}/${dataset_name}/nqg_${split}"

export NQG_TEST_TSV="${base_dir}/data/${dataset_name}/${split}_split/nqg_test.tsv"

export T5_pred="${pred_dir}/${dataset_name}/t5_${split}.txt"
export OUTPUT="${pred_dir}/${dataset_name}/"

if [[ $dataset_name == *"SCAN"* ]]
then
    export CONFIG="${base_dir}/model/parser/configs/scan_config.json"
elif [[ $dataset_name == *"spider"* ]]
then
    export target_grammar="${base_dir}/model/parser/inference/targets/spider_grammar.txt"
    export T5_pred="${pred_dir}/${dataset_name}/t5_${split}_cleaned.txt"
elif [[ $dataset_name == *"COGS"* ]]
then
    export CONFIG="${base_dir}/model/parser/configs/COGS_config.json"
elif [[ $dataset_name == *"geoquery"* ]]
then
    export target_grammar="${base_dir}/model/parser/inference/targets/funql.txt"
    if [[ $split == *"template"* ]]
    then
        export CONFIG="${base_dir}/model/parser/configs/geoquery_xl_config.json"
    fi
fi


cd $base_dir

# Check if corresponding T5 prediction exists, if not run the command go generate T5 predictions
if [ ! -f ${t5_pred}]
then
  # Generate T5 prediction
  bash ${BASE_DIR}/exp_scripts/gen_pred_hf.sh ${dataset_name} ${split}
fi

if [[ $dataset_name == *"spider"* ]]
then
    python model/parser/inference/eval_model.py \
      --input=${NQG_TEST_TSV}  \
      --config=${CONFIG}   \
      --model_dir=${MODEL_DIR}   \
      --bert_dir=${BERT_DIR}   \
      --target_grammar=${target_grammar}  \
      --rules=${RULES} \
      --fallback_predictions=${T5_pred}

    # Generate TXT file for sources
    python tasks/strip_targets.py \
        --input=${NQG_TEST_TSV} \
        --output=${SOURCE_TXT}

    python model/parser/inference/generate_predictions.py \
      --input=${NQG_TEST_TSV}  \
      --output=${OUTPUT}/nqgt5_${split}.txt \
      --config=${CONFIG}   \
      --model_dir=${MODEL_DIR}   \
      --bert_dir=${BERT_DIR}   \
      --target_grammar=${target_grammar}  \
      --rules=${RULES} \
      --fallback_predictions=${T5_pred}
else
    python model/parser/inference/eval_model.py \
      --input=${TEST_TSV}  \
      --config=${CONFIG}   \
      --model_dir=${MODEL_DIR}   \
      --bert_dir=${BERT_DIR}   \
      --rules=${RULES} \
      --target_grammar=${target_grammar}  \
      --fallback_predictions=${T5_pred}

    # Generate TXT file for sources
    python tasks/strip_targets.py \
        --input=${TEST_TSV} \
        --output=${SOURCE_TXT}

    python model/parser/inference/generate_predictions.py \
      --input=${TEST_TSV}  \
      --output=${OUTPUT}/nqgt5_${split}.txt \
      --config=${CONFIG}   \
      --model_dir=${MODEL_DIR}   \
      --bert_dir=${BERT_DIR}   \
      --rules=${RULES}  \
      --target_grammar=${target_grammar}  \
      --fallback_predictions=${T5_pred}
fi
