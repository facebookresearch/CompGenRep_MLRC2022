# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
import logging
import os
import re
import sys
import json
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pdb

import datasets
from datasets import load_dataset, load_metric
from ast import literal_eval

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    AdamW,
    Adafactor,
    get_scheduler,
)
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from trainer_seq2seq_sp import SemanticParsingSeq2SeqTrainer

torch.cuda.empty_cache()
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0")

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    is_tuning: bool = field(
        default=False,
        metadata={
            "help": "Whether we are tunning hyperparameters. "
            "If True, will automatically split the training set into validation set "
        },
    )

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_output_length: int = field(
        default=512,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )

    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
 
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    num_beams: Optional[int] = field(
        default=20,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "tsv"], "`train_file` should be a csv or tsv file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "tsv"], "`validation_file` should be a csv or tsv file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "tsv"], "`test_file` should be a csv or tsv file."

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        if data_args.dataset_name == 'scan':
            raw_datasets = raw_datasets.rename_column('commands', 'input')
            raw_datasets = raw_datasets.rename_column('actions', 'output')
            # Temporaraily set val to be test
            raw_datasets["validation"] = raw_datasets["test"]
        logger.warning(f"Changed column names of SCAN dataset into {raw_datasets}")
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]

        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
            
        if extension == "tsv":
            # When extension is tsv, it follows NQG format and will not have column names
            raw_datasets = load_dataset("csv", data_files=data_files, sep='\t', column_names=["input", "output"])
        else:
            raw_datasets = load_dataset(extension, data_files=data_files, sep='\t')

        if data_args.is_tuning:
            raw_datasets = raw_datasets['train'].train_test_split(test_size=0.1)
            raw_datasets['validation'] = raw_datasets['test']

    
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    # Temporarily set max_answer_length for training.
    
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    max_answer_length = min(data_args.max_output_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        inputs = examples['input']
        if 't5' in model_args.model_name_or_path or 'COGS' not in data_args.train_file:
            inputs = ['semanticparse: ' + x for x in inputs]
        else:
            inputs = [x for x in inputs]
        targets = examples['output']
        model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True, return_offsets_mapping=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                desc="Running tokenizer on train dataset",
            )
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                preprocess_function,
                batched=True,
                desc="Running tokenizer on validation dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    metric = load_metric("exact_match")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids, ignore_case=True, ignore_punctuation=True, regexes_to_ignore=' ')

    # Post-processing:
    def post_processing_function(
        examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, stage="eval"
    ):
        # Decode the predicted tokens.
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds)
        decoded_preds = [pred.replace(" ⁇ ", "<").replace("<pad> ", "").replace("<pad>", "").replace("</s>", "").replace("<unk>", "<").replace("<s>", "") for pred in decoded_preds]
        predictions = []
        raw_references = []
        # Fix white space
        def white_space_fix(text):
            return " ".join(text.split())
        # Let's loop over all the examples!
        for i in range(len(features)):
            predictions.append(white_space_fix(decoded_preds[i]))
            raw_references.append(white_space_fix(features[i]['output'].replace(" ,", ",")))

        # Save predictions
        prefix = 'eval'
        prediction_file = os.path.join(
            training_args.output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(predictions, indent=4) + "\n")
        # Save ground truth
        ground_truth_file = os.path.join(
            training_args.output_dir, "golds.json" if prefix is None else f"{prefix}_golds.json"
        )
        logger.info(f"Saving predictions to {ground_truth_file}.")
        with open(ground_truth_file, "w") as writer:
            writer.write(json.dumps(raw_references, indent=4) + "\n")
        
        return EvalPrediction(predictions=predictions, label_ids=raw_references)
    

    # Initialize optimizer and scheduler
    if training_args.do_train:
        if 't5' in model_args.model_name_or_path:
        # optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=0.01)
            optimizer = Adafactor(model.parameters(), lr=training_args.learning_rate, relative_step=False)
        else:
            optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=0.01)
        lr_scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps= training_args.num_train_epochs * (len(train_dataset) // training_args.per_device_train_batch_size))

    # Initialize our Trainer
    if training_args.do_train:
        trainer = SemanticParsingSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            eval_examples=eval_examples if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, lr_scheduler),
            # generation_num_beams=data_args.num_beams,
            post_process_function=post_processing_function,
            # num_beams=data_args.num_beams,
        )
    else:
        trainer = SemanticParsingSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            eval_examples=eval_examples if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            post_process_function=post_processing_function,
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        data_args.max_seq_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval and not training_args.do_predict:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")

        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        res = trainer.predict(eval_dataset)
        # Save the prediction files for spider evaluation
        prediction_list = []
        for pred_idx, pred_id in enumerate(res.predictions):
            prediction_list.append(pred_id)
        
        # Output to result dir
        base_dir = os.environ["BASE_DIR"]
        # Strip the dataset name and split
        test_list = data_args.validation_file.split('/')
        dataset_name = test_list[test_list.index('data') + 1]
        split = test_list[test_list.index('data') + 2].split('_')[0]
        if 't5' in model_args.model_name_or_path:
            model_name = 't5'
        else:
            model_name = 'bart'
            
        logger.info("Writing model predictions to txt file...")
        with open(base_dir + '/results/predictions/' + dataset_name + '/' + model_name + '_' + split + '.txt', 'w') as f:
            for line in prediction_list:
                f.write(f"{line}\n")

if __name__ == "__main__":
    main()