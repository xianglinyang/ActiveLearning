import gc
import numpy as np


def get_unlabeled_idx(pool_num, lb_idx):
    tot_idx = np.arange(pool_num)
    ulb_idx = np.setdiff1d(tot_idx, lb_idx)
    return ulb_idx


class QueryMethod:
    """
    A general class for query strategies, with a general method for querying examples to be labeled.
    """

    def __init__(self, task_model, task_model_type, pool_num):
        """
        init Query Method
        :param task_model: task_model
        :param task_model_type: "tensorflow" or "pytorch"
        :param pool_num: the number of samples in the pool
        """
        self.task_model = task_model
        self.task_model_type = task_model_type
        self.pool = pool_num

    def query(self, *args, **kwargs):
        return NotImplemented

    def update_model(self, new_model):
        del self.task_model
        gc.collect()
        self.task_model = new_model

