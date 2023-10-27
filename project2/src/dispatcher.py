import argparse
import json
import random
import os, subprocess
from csv import DictWriter
import multiprocessing

from csv import DictReader
from vectorizer import Vectorizer
from logistic_regression import LogisticRegression
import json
from sklearn.metrics import roc_auc_score
import numpy as np
import time
import random

def add_main_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--plco_data_path",
        default="lung_prsn.csv",
        help="Location of PLCO csv",
    )

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

    return parser

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
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    :configs: flags to use for this model run. Will be fed into
    scripts/main.py

    returns: flags for this experiment as well as result metrics
    '''

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    # TODO: Launch the experiment
    train, val, test = load_data(args)

    feature_config = {
        'age': 0
    }
    
    plco_vectorizer = Vectorizer(feature_config)
    plco_vectorizer.fit(train)

    train_X = plco_vectorizer.transform(train)
    val_X = plco_vectorizer.transform(val)
    test_X = plco_vectorizer.transform(test)

    train_Y = np.array([int(r["lung_cancer"]) for r in train])
    val_Y = np.array([int(r["lung_cancer"]) for r in val])
    test_Y = np.array([int(r["lung_cancer"]) for r in test])

    model = LogisticRegression(num_epochs=experiment_config['num_epochs'], learning_rate=experiment_config['learning_rate'], 
                                batch_size=experiment_config['batch_size'], regularization_lambda=experiment_config['regularization_lambda'], verbose=True)

    model.fit(train_X, train_Y, val_X, val_Y)
    # TODO: Parse the results from the experiment and return them as a dict

    pred_train_Y = model.predict_proba(train_X)#[:,-1]
    pred_val_Y = model.predict_proba(val_X)#[:,-1]

    results = {
        "num_epochs": experiment_config['num_epochs'], 
        "learning_rate": experiment_config['learning_rate'], 
        "batch_size": experiment_config['batch_size'], 
        "regularization_lambda": experiment_config['regularization_lambda'],
        "train_auc": roc_auc_score(train_Y, pred_train_Y),
        "val_auc": roc_auc_score(val_Y, pred_val_Y)
    }
    
    results_path = os.path.join(args.log_dir, str(time.time()) + '_' + 'result.json')


    json.dump(results, open(results_path, "w"), indent=True, sort_keys=True)

    return results


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
