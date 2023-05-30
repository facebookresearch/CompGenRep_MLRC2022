# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
import os

BASE_DIR = os.environ.get('BASE_DIR')

MODEL_DIR = os.path.join(BASE_DIR, 'trained_models/')
TMCD_MODEL_DIR = os.path.join(BASE_DIR, 'baseline_replication/TMCD/trained_models/')

DATA_DIR = os.path.join(BASE_DIR, 'data/')
TMCD_DATA_DIR = os.path.join(BASE_DIR, 'baseline_replication/TMCD/data/')

TMCD_DATASETS = {'SCAN', 'geoquery', 'spider'}