import numpy as np

from query_strategies.query_strategy import QueryMethod
from query_strategies.query_strategy import get_unlabeled_idx

class LeastConfidenceg(QueryMethod):
    """
    The basic uncertainty sampling query strategy, querying the examples with the minimal top confidence.
        adopt from discriminative active learning repo
        https://github.com/dsgissin/DiscriminativeActiveLearning/blob/master/query_methods.py
    """

    def __init__(self, task_model, task_model_type, n_pool):
        super().__init__(task_model, task_model_type, n_pool)

    def query(self, complete_dataset, labeled_idx, budget):
        unlabeled_idx = get_unlabeled_idx(self.n_pool, labeled_idx)
        # SUBSET
        # get prediction
        # TODO

        unlabeled_predictions = np.amax(predictions, axis=1)
        selected_indices = np.argpartition(unlabeled_predictions, budget)[:budget]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))