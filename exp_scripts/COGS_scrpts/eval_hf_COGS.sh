# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
#!/bin/bash
# The script to evalaute a trained Huggingface model on Test and Generalization split

export base_dir=$BASE_DIR
export model_dir=$base_dir/trained_models
export dataset_name='COGS'
export data_dir=$base_dir/baseline_replication/${dataset_name}/data
export lr='1e-4'
export batch_size='1'
export seed='42'
export epoch='20'
export model_name='t5-base'

if [[ $model_name == *"t5"* ]]
then
    dir_model_name=$model_name
elif [[ $model_name == *"bart"* ]]
then
    dir_model_name="bart"
fi

cd $base_dir

python ${BASE_DIR}/hf_training/fine_tune_t5.py \
    --model_name_or_path "$model_dir/$dataset_name/${dir_model_name}_${lr}/" \
    --validation_file "$data_dir/test.csv" \
    --do_eval \
    --seed $seed \
    --predict_with_generate \
    --per_device_eval_batch_size $batch_size \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --gradient_accumulation_steps 32 \
    --max_seq_length 512  \
    --max_output_length 512 \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --metric_for_best_model "exact_match" \
    --evaluation_strategy "epoch" \
    --generation_num_beams 20 \
    --generation_max_length 512 \
    --output_dir "$model_dir/$dataset_name/${dir_model_name}_${lr}/"

python ${BASE_DIR}/hf_training/fine_tune_t5.py \
    --model_name_or_path "$model_dir/$dataset_name/${dir_model_name}_${lr}/" \
    --validation_file "$data_dir/gen.csv" \
    --do_eval \
    --seed $seed \
    --predict_with_generate \
    --per_device_eval_batch_size $batch_size \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --gradient_accumulation_steps 32 \
    --max_seq_length 512  \
    --max_output_length 512 \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --metric_for_best_model "exact_match" \
    --evaluation_strategy "epoch" \
    --generation_num_beams 20 \
    --generation_max_length 512 \
    --output_dir "$model_dir/$dataset_name/${dir_model_name}_${lr}/"