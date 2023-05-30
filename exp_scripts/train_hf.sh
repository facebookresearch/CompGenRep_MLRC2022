# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
#!/bin/bash
export dataset_name='SCAN'
export split='template_around_right'
export model_name='t5-base'

export base_dir=${BASE_DIR}
export model_dir=$base_dir/trained_models
export script_dir=$base_dir/baseline_replication/TMCD
if [[ $dataset_name == *"COGS"* ]]
then
    export data_dir=$base_dir/baseline_replication/COGS/data
else
    export data_dir=$base_dir/baseline_replication/TMCD/data
fi

# Hyperparmeters
export lr='1e-4'
export batch_size='1'
export seed='42'
export epoch='70'

# load_best_model_at_end set to False because of TMCD
if [[ $model_name == *"t5"* ]]
then
    dir_model_name=$model_name
elif [[ $model_name == *"bart"* ]]
then
    dir_model_name="bart"
fi
mkdir -p $model_dir/$dataset_name/

cd $base_dir
if [[ $dataset_name != *"COGS"* ]]
then
    python $BASE_DIR/hf_training/fine_tune_t5.py \
        --model_name_or_path $model_name \
        --train_file "$data_dir/$dataset_name/${split}_split/train.tsv" \
        --validation_file "$data_dir/$dataset_name/${split}_split/test.tsv" \
        --do_train \
        --do_eval \
        --seed $seed \
        --predict_with_generate \
        --per_device_train_batch_size $batch_size \
        --per_device_eval_batch_size $batch_size \
        --learning_rate $lr \
        --num_train_epochs $epoch \
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
else
    # Because COGS has a development set, we can select load_best_checkpoint as true at last
    python $BASE_DIR/hf_training/fine_tune_t5.py \
        --model_name_or_path $model_name \
        --train_file "$data_dir/train.csv" \
        --validation_file "$data_dir/dev.csv" \
        --do_train \
        --do_eval \
        --seed $seed \
        --predict_with_generate \
        --per_device_train_batch_size $batch_size \
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
fi
