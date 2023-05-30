# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
#!/bin/bash
# The script to train and evaluate LSTM
# Remember to specify the BASE_DIR and change the dataset/split/model_name before running the script

export dataset="geoquery"
export split="length"
export model_name="lstm_uni"   # Model name \in {lstm_uni, lstm_bi, transformer}

export base_dir=${BASE_DIR}
export input_path="${base_dir}/baseline_replication/TMCD/data/${dataset}/${split}_split"
export data_output_path="${base_dir}/baseline_replication/COGS/processed_data/${dataset}/${split}"

cd ${base_dir}/baseline_replication/COGS


############### Preparing for data ################
## Reformat data for NQG
python scripts/reformat_nqg_data_for_opennmt.py --input_path ${input_path} --output_path ${data_output_path}

export OPENNMT_DIR="${base_dir}/baseline_replication/COGS/src/OpenNMT-py"
## Preprocess data into OpenNMT format
python $OPENNMT_DIR/preprocess.py \
    -train_src $data_output_path/train_source.txt   \
    -train_tgt $data_output_path/train_target.txt   \
    -save_data $data_output_path/1_example  \
    -src_seq_length 5000 -tgt_seq_length 5000   \
    -src_vocab $data_output_path/source_vocab.txt -tgt_vocab $data_output_path/target_vocab.txt


############### Model Training and Inference ################
export EXAMPLES=1_example                      # Number of exposure examples (1 or 100)
export SAVE_PATH=${base_dir}/trained_models/${dataset}/${split}  # Save path for checkpoints
export SAVE_NAME=${EXAMPLES}_${model_name}          # Checkpoint name
export LOG_PATH=${base_dir}/logs           # Log path
export PRED_PATH=${base_dir}/baseline_replication/COGS/preds         # Predictions path
export SEED=1                                  # Random seed
export CUDA_VISIBLE_DEVICES=0                  # GPU machine number

if [[ $model_name == *"lstm_uni"* ]]
then
    encoder_type="rnn"
elif [[ $model_name == *"lstm_bi"* ]]
then
    encoder_type="brnn"
fi

mkdir -p $SAVE_PATH
mkdir -p $LOG_PATH
## Training
python -u $OPENNMT_DIR/train.py -data ${data_output_path}/$EXAMPLES -save_model $SAVE_PATH/${SAVE_NAME}/s$SEED \
	-layers 2 -rnn_size 512 -word_vec_size 512 \
	-encoder_type $encoder_type -decoder_type rnn -rnn_type LSTM \
        -global_attention dot \
	-train_steps 30000  -max_generator_batches 2 -dropout 0.1 \
	-batch_size 128 -batch_type sents -normalization sents  -accum_count 4 \
	-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 4000 -learning_rate 2 \
	-max_grad_norm 5 -param_init 0  \
	-valid_steps 500 -save_checkpoint_steps 500 \
	-early_stopping 5 --early_stopping_criteria loss \
	-world_size 1 -gpu_ranks 0 -seed $SEED --log_file ${LOG_PATH}/${dataset}_${split}_${SAVE_NAME}_s${SEED}.log 
	
## Inference
for SPLIT in test
do
    python $OPENNMT_DIR/translate.py -model $SAVE_PATH/${SAVE_NAME}/s${SEED}_best.pt \
                                      -src ${data_output_path}/${SPLIT}_source.txt \
                                      -tgt ${data_output_path}/${SPLIT}_target.txt \
                                      -output ${PRED_PATH}/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt \
                                      -replace_unk -verbose -shard_size 0 \
                                      -gpu 0 -batch_size 128 \
                                      --max_length 2000

    paste ${data_output_path}/${SPLIT}_source.txt ${data_output_path}/${SPLIT}_target.txt ${PRED_PATH}/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt > ${PRED_PATH}/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.tsv
done
