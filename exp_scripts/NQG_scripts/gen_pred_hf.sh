# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
#!/bin/bash
# This script generate the prediction of T5 so that NQG can be evaluated into an ensemble (NQG-T5)
export base_dir=${BASE_DIR}/baseline_replication/TMCD
export model_dir=${BASE_DIR}/trained_models
export data_dir=$base_dir/data
export dataset_name='geoquery'
export split='template'
export lr='1e-4'
export batch_size='1'
export seed='42'
export model_name='t5-base'
export results_dir=${BASE_DIR}/results

if [[ ! $# -eq 0 ]];
then
  export dataset_name=$1
  export split=$2
  export model_name='t5-base'
fi

# load_best_model_at_end set to False because of TMCD
if [[ $model_name == *"t5"* ]]
then
    dir_model_name=$model_name
elif [[ $model_name == *"bart"* ]]
then
    dir_model_name="bart"
fi
cd $base_dir

mkdir -p "${results_dir}/predictions/${dataset_name}"

python ${hf_training}/BASE_DIR/fine_tune_t5.py \
    --model_name_or_path "$model_dir/$dataset_name/${dir_model_name}_${split}_${lr}/checkpoint-2400/" \
    --validation_file "$data_dir/$dataset_name/${split}_split/test.tsv" \
    --do_eval \
    --do_predict \
    --seed $seed \
    --predict_with_generate \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps 16 \
    --max_seq_length 512  \
    --max_output_length 256 \
    --save_strategy "epoch" \
    --load_best_model_at_end False \
    --metric_for_best_model "exact_match" \
    --evaluation_strategy "epoch" \
    --generation_num_beams 20 \
    --generation_max_length 256 \
    --output_dir "$model_dir/$dataset_name/${dir_model_name}_${split}_${lr}/"
 