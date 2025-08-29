"""
This is the main script for running experiments.
It parses command-line arguments to set up the experiment parameters and then runs the experiment.
E.g. to run a test:
> python3 main.py "seed=0|n_seeds_per_job=1|recalibrate=True|experiment=GP-UCB-testRun|
> surrogate=GP|acquisition=UCB|data_name=Benchmark|d=2|problem=SixHumpCamel|n_initial=5|n_evals=30|n_test=5000|n_pool=5000|
> snr=100.0|quantile_level=0.95|noisify=True|track_surrogate_state=False|bo=True|test=False|scale_kernel=True|n_calibration_bins=20|
> recalibration_method=isotonic|recalibrator_type=UNIBOv2|eta=0.5"
"""

from src.parameters import Parameters
from src.dataset import Dataset
from src.experiment import Experiment
from surrogates.gaussian_process import GaussianProcess
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from botorch.acquisition.analytic import ExpectedImprovement
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import random
from torch.utils.data.sampler import SubsetRandomSampler
import json
from sklearn.preprocessing import StandardScaler
import itertools
from datetime import datetime
import time
import sys
import warnings
from tqdm import tqdm

#python3 -c "from main import *; run()" $args
#"seed=0|n_seeds_per_job=1|surrogate=GP|acquisition=EI|data_name=mnist|std_change=1.0|bo=True|experiment=experiment-GP--0|test=False|extensive_metrics=True|recalibrate=False"
#"seed=0|n_seeds_per_job=1|surrogate=GP|acquisition=EI|data_name=benchmark|problem_idx=11|snr=100|bo=True|d=3|experiment=experiment-TEST--seed-TEST"|test=False|recalibrate=False"
def run():
    start = time.time()
    try:
        args = sys.argv[-1].split("|")
    except:
        args = []
    print("------------------------------------")
    print("Arguments:", args)
    print("RUNNING EXPERIMENT...")
    warnings.filterwarnings("ignore", message="A not p.d.")
    kwargs = {}
    parameters_temp = Parameters(mkdir=False)
    if args[0] != "main.py":
        for arg in args:
            var = arg.split("=")[0]
            val = arg.split("=")[1]
            par_val = getattr(parameters_temp, var)

            if isinstance(par_val, bool):
                val = val.lower() == "true"
            elif isinstance(par_val, int):
                val = int(val)
            elif isinstance(par_val, float):
                val = float(val)
            elif isinstance(par_val, str):
                pass
            else:
                var = None
                print("COULD NOT FIND VARIABLE:", var)
            kwargs.update({var: val})

    for i in range(kwargs['n_seeds_per_job']):
        parameters = Parameters(kwargs, mkdir=True)
        print("Running with:", parameters)
        print(f"Using device: {parameters.device}")
        experiment = Experiment(parameters)
        experiment.run()
        print("FINISHED EXPERIMENT")
        print("------------------------------------")
        kwargs['seed'] += 1
if __name__ == "__main__":
    run()
