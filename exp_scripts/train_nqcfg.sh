# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
#!/bin/bash

# The script to train neural-QCFG
export dataset_name='SCAN'
export split='addprim_turn_left'
export seed='42'

export base_dir=${BASE_DIR}/baseline_replication/neural-qcfg
export model_dir=${BASE_DIR}/trained_models
export data_dir=$base_dir/baseline_replication/data/$dataset_name

mkdir -p $model_dir/${dataset_name}
cd $base_dir

python train_scan.py --train_file data/${dataset_name}/tasks_train_${split}.txt \
    --save_path $model_dir/${dataset_name}/nqcfg-${split}-${seed}.pt --seed ${seed}

python predict_scan.py --data_file data/${dataset_name}/tasks_test_${split}.txt \
    --model_path $model_dir/${dataset_name}/nqcfg-${split}-${seed}.pt --seed ${seed}
    
