# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
import os
import re

from datasets import load_dataset

base_dir = os.environ['BASE_DIR']

def load_dataset_with_name(dataset_name, split):
    """
    Take a dataset name and split name, load the dataset.
    Returns a huggingface dataset dict.
    """
    # TODO: Uncomment this line after refactor
    # path = base_dir + '/data/' + dataset_name + '/' + split + '_split/'
    path = base_dir + '/baseline_replication/TMCD/data/' + dataset_name + '/' + split + '_split/'

    data_files = {}
    
    if os.path.exists(path + 'train.tsv'):
        data_files["train"] = path + 'train.tsv'
    if os.path.exists(path + 'dev.tsv'):
        data_files["validation"] = path + 'dev.tsv'
    if os.path.exists(path + 'test.tsv'):
        data_files["test"] = path + 'test.tsv'
    if os.path.exists(path + 'gen.tsv'):
        data_files["gen"] = path + 'gen.tsv'

    raw_datasets = load_dataset("csv", data_files=data_files, sep='\t', column_names=["input", "output"])
    return raw_datasets

def list_datasets_and_their_splits(data_path):
    """
    data_path (str): The directory that include all the dataset files
    returns:
        dataset_names (list of str)
        splits_mapping (dict, key in dataset_names): values are the available splits
    """
    avail_datasets = os.listdir(data_path)
    dataset_names = []
    splits_mapping = dict()

    for dir in avail_datasets:
        if 'orig' not in dir and '_hp' not in dir:
            dataset_names.append(dir)
            avail_splits = os.listdir(data_path +'/' + dir)
            # Add splits to the dict mapping
            for split in avail_splits:
                if '_split' in split:
                    if dir not in splits_mapping:
                        splits_mapping[dir] = []
                    splits_mapping[dir].append(re.sub('_split', '', split))
    return dataset_names, splits_mapping