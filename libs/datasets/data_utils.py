import os
import random

import torch
import numpy as np


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing.
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker.
    """
    seed = torch.initial_seed() % 2 ** 31
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)