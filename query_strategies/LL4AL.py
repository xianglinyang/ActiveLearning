import numpy as np
import torch
import torch.nn
import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from query_strategies.query_strategy import QueryMethod
from query_strategies.query_strategy import get_unlabeled_idx


class LL4ALSampling(QueryMethod):
    """
    The basic uncertainty sampling query strategy, querying the examples with the minimal top confidence.
        adopt from LL4AL repo
        https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
    """

    def __init__(self, model, model_type, n_pool, init_lb, num_classes, dataset_name, model_name, gpu=True, **kwargs):

        super(LL4ALSampling, self).__init__(model, model_type, n_pool)
        self.strategy_name = "LL4AL"
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_classes = num_classes
        self.lb_idxs = init_lb
        self.device = torch.device("cuda:0" if gpu else "cpu")
        self.kwargs = kwargs

    def query(self, budget):
        # TODO
        pass
        # return np.hstack((self.lb_idxs, new_indices))

    def update_lb_idxs(self, new_indices):
        self.lb_idxs = new_indices

    # def train(self, total_epoch, complete_dataset):
