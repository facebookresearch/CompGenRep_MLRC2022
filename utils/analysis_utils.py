# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
import os
import json

from constants import TMCD_DATASETS, TMCD_MODEL_DIR, MODEL_DIR

def load_training_curve_info(model_name, dataset, split, checkpoint=None):
    """
    Returns steps [list], ems [list], best_em float
    """
    ems = []
    steps = []
    best_em = 0.0

    # Find the path to the model
    if dataset in TMCD_DATASETS:
        # Load the model in TMCD data dir
        path = os.path.join(TMCD_MODEL_DIR, dataset, model_name + '_' + split + '_1e-4')
    else:
        path = os.path.join(MODEL_DIR, dataset, model_name + '_' + split + '_1e-4')
    if checkpoint is not None:
        path = os.path.join(path, 'checkpoint-' + checkpoint)
    # Load the model's trainer_state
    trainer_state = json.load(open(path + '/trainer_state.json'))
    for metrics in trainer_state['log_history']:
        if 'eval_exact_match' in metrics:
            ems.append(metrics['eval_exact_match'])
            steps.append(metrics['step'])
            if metrics['eval_exact_match'] > best_em:
                best_em = metrics['eval_exact_match']

    return steps, ems, best_em