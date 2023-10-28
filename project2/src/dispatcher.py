import argparse
import json
import random
import os, subprocess
from csv import DictWriter
import multiprocessing
import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))
from src.lightning import MLP, RiskModel
from src.dataset import PathMnist, NLST
from lightning.pytorch.cli import LightningArgumentParser
import lightning.pytorch as pl
import json
from sklearn.metrics import roc_auc_score
import numpy as np
import time

NAME_TO_MODEL_CLASS = {
    "mlp": MLP,
    "risk_model": RiskModel
}

NAME_TO_DATASET_CLASS = {
    "pathmnist": PathMnist,
    "nlst": NLST
}

def add_main_args(parser: LightningArgumentParser) -> LightningArgumentParser:
   
    parser.add_argument(
        "--config_path",
        type=str,
        default="grid_search.json",
        help="Location of config file"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of processes to run in parallel"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Location of experiment logs and results"
    )

    parser.add_argument(
        "--grid_search_results_path",
        default="grid_results.csv",
        help="Where to save grid search results"
    )

    parser.add_argument(
        "--model_name",
        default="mlp",
        help="Name of model to use. Options include: mlp, cnn, resnet",
    )

    parser.add_argument(
        "--dataset_name",
        default="pathmnist",
        help="Name of dataset to use. Options: pathmnist, nlst"
    )

    parser.add_argument(
        "--project_name",
        default="cornerstone",
        help="Name of project for wandb"
    )

    parser.add_argument(
        "--monitor_key",
        default="val_loss",
        help="Name of metric to use for checkpointing. (e.g. val_loss, val_acc)"
    )

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to checkpoint to load from. If None, init from scratch."
    )

    parser.add_argument(
        "--train",
        default=True,
        action="store_true",
        help="Whether to train the model."
    )
    return parser

def parse_args() -> argparse.Namespace:
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, nested_key="trainer")
    for model_name, model_class in NAME_TO_MODEL_CLASS.items():
        parser.add_lightning_class_args(model_class, nested_key=model_name)
    for dataset_name, data_class in NAME_TO_DATASET_CLASS.items():
        parser.add_lightning_class_args(data_class, nested_key=dataset_name)
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

import itertools
def get_experiment_list(config: dict) -> list:
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item, but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of dicts, each of which encapsulates one job.
        *Example: {learning_rate: 0.001 , batch_size: 16 ...}
    '''

    # TODO: Go through the tree of possible jobs and enumerate into a list of jobs

    f = open('grid_search.json')
    data = json.load(f)
    f.close()
    
    keypair_dicts = []
    for key in data.keys():
        pairs = []
        for i in range(len(data[key])):
            pairs.append((key, data[key][i]))
        keypair_dicts.append(pairs)

    enum_result = list(itertools.product(*keypair_dicts))
    
    jobs = []
    for i in range(len(enum_result)):
        job_dict = {}
        for j in range(len(enum_result[i])):
            key = enum_result[i][j][0]
            value = enum_result[i][j][1]
            job_dict[key] = value
        jobs.append(job_dict)
    
    return jobs

def worker(args: argparse.Namespace, job_queue: multiprocessing.Queue, done_queue: multiprocessing.Queue):
    '''
    Worker thread for each worker. Consumes all jobs and pushes results to done_queue.
    :args - command line args
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(
            launch_experiment(args, params))

def load_data(args: argparse.Namespace) -> ([list, list, list]):
    '''
    Load PLCO data from csv file and split into train validation and testing sets.
    '''
    reader = DictReader(open(args.plco_data_path,"r"))
    rows = [r for r in reader]
    NUM_TRAIN, NUM_VAL = 100000, 25000
    random.seed(0)
    random.shuffle(rows)
    train, val, test = rows[:NUM_TRAIN], rows[NUM_TRAIN:NUM_TRAIN+NUM_VAL], rows[NUM_TRAIN+NUM_VAL:]

    return train, val, test
        
        
def launch_experiment(args: argparse.Namespace, experiment_config: dict) ->  dict:
 print("Initializing model")
    ## TODO: Implement your deep learning methods
    datamodule = NAME_TO_DATASET_CLASS[args.dataset_name](**vars(args[args.dataset_name]))

    if args.checkpoint_path is None:
        model = NAME_TO_MODEL_CLASS[args.model_name](**vars(args[args.model_name]))
    else:
        model = NAME_TO_MODEL_CLASS[args.model_name].load_from_checkpoint(args.checkpoint_path)

    print("Initializing trainer")
    logger = pl.loggers.WandbLogger(project=args.project_name)

    args.trainer.accelerator = 'auto'
    args.trainer.logger = logger
    args.trainer.precision = "bf16-mixed" ## This mixed precision training is highly recommended

    args.trainer.callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=args.monitor_key,
            mode='min' if "loss" in args.monitor_key else "max",
            save_last=True
        )]

    trainer = pl.Trainer(**vars(args.trainer))

    if args.train:
        print("Training model")
        trainer.fit(model, datamodule)

    print("Best model checkpoint path: ", trainer.checkpoint_callback.best_model_path)

    print("Evaluating model on validation set")
    trainer.validate(model, datamodule)

    print("Evaluating model on test set")
    trainer.test(model, datamodule)
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace) -> dict:
    print(args)
    config = json.load(open(args.config_path, "r"))
    print("Starting grid search with the following config:")
    print(config)

    # TODO: From config, generate a list of experiments to run
    experiments = get_experiment_list(config)
    random.shuffle(experiments)

    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for exper in experiments:
        job_queue.put(exper)

    print("Launching dispatcher with {} experiments and {} workers".format(len(experiments), args.num_workers))

    # TODO: Define worker fn to launch an experiment as a separate process.
    for _ in range(args.num_workers):
        multiprocessing.Process(target=worker, args=(args, job_queue, done_queue)).start()

    # Accumualte results into a list of dicts
    grid_search_results = []
    for _ in range(len(experiments)):
        grid_search_results.append(done_queue.get())

    keys = grid_search_results[0].keys()

    print("Saving results to {}".format(args.grid_search_results_path))

    writer = DictWriter(open(args.grid_search_results_path, 'w'), keys)
    writer.writeheader()
    writer.writerows(grid_search_results)

    print("Done")

if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)
