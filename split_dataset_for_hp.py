# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
# The script to randomly split a dataset for hyperparameter tunning
import os
from absl import app
from absl import flags
import pdb

from datasets import load_dataset, concatenate_datasets

FLAGS = flags.FLAGS
flags.DEFINE_string("input", "", "Input directory that contains train.tsv and test.tsv .")
flags.DEFINE_string("dataset", "", "Input dataset name. Output will be stored at dataset_hp")

def main(unused_argv):
    # Concatenate train and test file
    data_files = {}
    data_files["train"] = FLAGS.input + '/train.tsv'
    data_files["test"] = FLAGS.input + '/test.tsv'
    # pdb.set_trace()
    raw_datasets = load_dataset("csv", data_files=data_files, sep='\t', column_names=["input", "output"])
    concat_data = concatenate_datasets([raw_datasets["train"], raw_datasets["test"]])

    # Split the dataset by 90:10 train test ratio
    splitted = concat_data.train_test_split(test_size=0.1, shuffle=True, seed=42)

    if not os.path.exists('data/' + FLAGS.dataset + '_hp'):
        os.makedirs('data/' + FLAGS.dataset + '_hp')
    # Output the corresponding splits to target directory
    splitted["train"].to_csv('data/' + FLAGS.dataset + '_hp' + '/train.csv', sep="\t", index=False)
    splitted["test"].to_csv('data/' + FLAGS.dataset + '_hp' + '/test.csv', sep="\t", index=False)

if __name__ == "__main__":
    app.run(main)