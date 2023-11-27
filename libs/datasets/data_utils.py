import os
import random

import torch
import numpy as np
from torch.utils.data import Subset


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

def subset_dataset(dataset, num_samples = 500):
    
    # Set the random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    # Generate random indices
    indices = torch.randperm(len(dataset))[:num_samples]
    # Create a subset
    subset = Subset(dataset, indices)
    return subset