# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
#!/bin/bash
# The script to evaluate spider performance

# Input the model name to evaluate
export model_name='nqgt5'
export results_dir=${BASE_DIR}/results

export base_dir=${BASE_DIR}/baseline_replication/TMCD
export data_dir=$base_dir/data
export lr='1e-4'
export batch_size='1'
export seed='42'
export dir_model_name=$model_name

for split in random length tmcd template
do
    pred_file_name=${model_name}_${split}.txt

    if [[ $model_name == *"t5-"* ]]
    then
        dir_model_name="t5"
        pred_file_name=${dir_model_name}_${split}_cleaned.txt
    elif [[ $model_name == *"bart"* ]]
    then
        dir_model_name="bart"
        pred_file_name=${dir_model_name}_${split}_cleaned.txt
    fi
    cd $base_dir

    # Generate the gold target file
    python tasks/spider/generate_gold.py --input="${data_dir}/spider/${split}_split/test.tsv" --output="${data_dir}/spider/${split}_split/target_test.txt"


    if [[ $model_name == *"bart"* ]] || [[ $model_name == *"-t5"* ]]
    then
        # Remove pad and /s
        python tasks/spider/restore_oov.py --input="${results_dir}/predictions/spider/${dir_model_name}_${split}.txt" --output="${results_dir}/predictions/spider/$pred_file_name"
    fi


    python tasks/spider/evaluation.py --gold="${data_dir}/spider/${split}_split/target_test.txt" --pred="${results_dir}/predictions/spider/$pred_file_name" --table="${data_dir}/orig_spider/tables.json" --etype="all" --db="${data_dir}/orig_spider/database/"


done