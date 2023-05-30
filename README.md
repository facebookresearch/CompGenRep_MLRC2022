# A Replication Study of Compositional Generalization Works on Semantic Parsing
This is the repository for the project: A Replication Study of Compositional Generalization Works on Semantic Parsing, in which we are replicating the results from three papers, [Shaw et al., 2021](https://aclanthology.org/2021.acl-long.75/), [Kim and Linzen, 2021](https://www.aclweb.org/anthology/2020.emnlp-main.731/), and [Kim, 2021](https://arxiv.org/abs/2109.01135).

The directories under `baseline_replication/` are the repository cloned from the original authors, with some minor changes and refactoring to fit into a single repository. You can find the link to the original repositories here: [COGS](https://github.com/najoungkim/COGS), [NQG](https://github.com/google-research/language/tree/master/language/compgen/nqg), [Neural-QCFG](https://github.com/yoonkim/neural-qcfg).

**We describe the process to reproduce the results below:**

First, create a virtual environment with `environment.yml` and `export BASE_DIR=/path/to/this/dir`.
## COGS

### Data Preparation
Download the `data/` directory from the original [COGS](https://github.com/najoungkim/COGS) repository and put the files under `baseline_replication/COGS/data/`; Download the folder `/src/OpenNMT-py` from the original repository and put it under `baseline_replication/COGS/`

For each split (train, test, gen)

The `opennmt_path` should point to the `src/OpenNMT` directory under `baseline_replication/COGS/`.

### Training and Inference for LSTM
The `train_lstm.sh` file under `exp_scripts` is constructed generally for both COGS dataset and the datasets used in NQG paper.

To train an LSTM model, simply modify the `model_name` (`lstm_bi` or `lstm_uni`) and dataset name in `train_lstm.sh` to the desired ones and run `bash exp_scripts/train_lstm.sh`

The script will first preprocess the data into a format that OpenNMT recognize, train the model, and finally evaluate the model. The trained model will be automatically saved to `trained_models/COGS/standard/`. The log file for running the script will be automatically saved to `logs/`

### Additional Exp: Fine-tuning T5 on COGS
`train_hf.sh` is the script for training any Huggingface model that has a FastTokenizer. In the project, we only trained T5. To fine-tune T5 on COGS with the same hyperparameter we used, change the `dataset_name` into COGS and `split` into standard. Then, run
```
bash exp_scripts/train_hf.sh
```
The model along with the checkpoints, again, will be automatically saved to `trained_models/COGS/standard`.

## NQG and NQG-T5

### Data Preparation
NQG and T5 are trained on the dataset splits generated by [Shaw et al., 2021](https://aclanthology.org/2021.acl-long.75/). One should follow [their strategy]((https://github.com/google-research/language/tree/master/language/compgen/nqg)) to obtain the data.

To convert the COGS dataset into the format that is suitable for NQG, use the script `baseline_replication/COGS/convert_to_nqg_format.py`:

```
python baseline_replication/COGS/convert_to_nqg_format.py --tsv=/path/to/inputCOGS --output=/desired/path/for/converted/COGS
```

When data are successfully gathered, the `data/` directory should have structure like below:

    data
    |--geoquery/
        |--standard_split/
            |--train.tsv
            |--test.tsv
        |--template_split/
        |--length_split/
        |--tmcd_split/
    |--spider
        |--tables.json
        |--standard_split/
        |--template_split/
        |--length_split/
        |--tmcd_split/
    |--SCAN
        |--standard_split/
        |--template_split/
        |--length_split/
        |--tmcd_split/
    |--COGS
        |--standard_split/

### Training and Evaluating NQG
To train an NQG model, modify the `dataset_name` and `split` into desire dataset name and run
```
python exp_scripts/train_nqg.sh
```
The script will preprocess the dataset (if it is SPIDER), induce the grammar rules, and convert the instances into TF examples. Finally, the script will evaluate the trained model.
Similar to prior section, the trained model will be saved to `trained_models/${dataset_name}/nqg_${split}`.

The process is the same to train T5 on GEOQUERY, SPIDER, and SCAN with training T5 on COGS.

To evaluate NQG-T5, first genearte T5 predictions into a `.txt` file running `exp_scripts/NQG_scrips/gen_pred_hf.sh`, then run `exp_scripts/NQG_scrips/eval_nqg_t5.sh`. Remember to adjust the `dataset_name` and `split` in these two files.
```
bash exp_scripts/NQG_scrips/gen_pred_hf.sh
bash exp_scripts/NQG_scrips/eval_nqg_t5.sh
```

#### Evaluation for SPIDER
SPIDER uses a special evaluation script, thus a separate `.txt` file will be generated when you run with the prediction option for both training NQG and T5.

The generation process is the same for evaluating NQG-T5 -- `eval_nqg_t5.sh` will also generate a prediction file for NQG-T5. Both scripts will be saving prediction in `results/predictions/` directory.

To evaluate on SPIDER, change the model name into {t5-base, nqg, nqgt5} and run `exp_scripts/evaluate_spider.sh`, which will evaluate the model on SPIDER for all splits, according to the prediction files generated in previous step.

## Neural-QCFG
### Code Preparation
Download the `neural_qcfg` code from the [original repository](https://github.com/yoonkim/neural-qcfg) (Code cannot be included here because of license limitations) into `baseline_replication/neural-qcfg/`. Do not overwrite `train_scan.py` file. 

### Data Preparation
Download the `data/` dir in the repository of [Neural-QCFG](https://github.com/yoonkim/neural-qcfg) and place it under `baseline_replication/neural-qcfg/`.

### Training and Inference
Change the desired dataset_name and split in `exp_scripts/train_qcfg.sh` and run
```
bash exp_scripts/train_qcfg.sh
```

Similar to all the scripts above, the script will be saving model to the `/trained_model` directory and evaluate the model.

## T5 training curve
Under the folder `utils/`, run the code blocks Jupyter notebook `analysis_plot.ipynb` to generate the training curves we presented in the paper. It works for any models with a `trainer_state.json` file in the directory.

## Citation
```
Todo: Add when ReScience finished processing
```

Please also cite the original paper:

NQG:
```
@inproceedings{shaw-etal-2021-compositional,
    title = "Compositional Generalization and Natural Language Variation: Can a Semantic Parsing Approach Handle Both?",
    author = "Shaw, Peter  and Chang, Ming-Wei  and Pasupat, Panupong  and Toutanova, Kristina",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.75",
    doi = "10.18653/v1/2021.acl-long.75",
    pages = "922--938",
}
```
COGS:
```
@inproceedings{kim-linzen-2020-cogs,
    title = "{COGS}: A Compositional Generalization Challenge Based on Semantic Interpretation",
    author = "Kim, Najoung  and Linzen, Tal",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.731",
    doi = "10.18653/v1/2020.emnlp-main.731",
    pages = "9087--9105",
}
```

Neural-QCFG:
```
@inproceedings{
kim2021sequencetosequence,
title={Sequence-to-Sequence Learning with Latent Neural Grammars},
author={Yoon Kim},
booktitle={Advances in Neural Information Processing Systems},
editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
year={2021},
url={https://openreview.net/forum?id=0vaPiltED1N}
}
```

## License
The majority of this project is licensed under CC-BY-NC, however portions of the project are available under separate license terms: [`baseline_replication/COGS`](https://github.com/najoungkim/COGS) and [`baseilne_replication/neural-qcfg`](https://github.com/yoonkim/neural-qcfg) are licensed under the MIT license, while [`baseline_replication/TMCD`](https://github.com/google-research/language/tree/master/language/compgen/nqg) are licensed under the Apache License 2.0.

CC-BY-NC statement:
This work is licensed under the Creative Commons Attribution-NonCommercial 2.0 Generic License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/2.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.