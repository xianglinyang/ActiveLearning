import torch.nn
from scipy.spatial import distance_matrix
import numpy as np
import torch
from keras.models import Model

from query_strategy import QueryMethod as QueryMethod
from query_strategy import get_unlabeled_idx as get_unlabeled_idx



class CoreSetSampling(QueryMethod):
    """
    An implementation of the greedy core set query strategy.
    """

    def __init__(self, model, model_type, pool_num, embedding_shape):
        """
        :param model:
        :param model_type:
        :param pool_num:
        :param embedding_shape: the number of dimension of embedding, int
        """
        super(CoreSetSampling, self).__init__(model, model_type, pool_num)
        self.embeding_shape = embedding_shape

    def greedy_k_center(self, labeled, unlabeled, amount):

        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 1000):
            if j + 1000 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+1000, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount-1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices)

    # def get_embedding_model(self):
    #     # tensorflow
    #     if self.task_model_type == "tensorflow":
    #         embedding_model = Model(input=self.task_model.input, output=self.task_model.get_layer('softmax').input)
    #     else:
    #         # pytorch version
    #         embedding_model = torch.nn.Sequential(*list(self.task_model.children())[:-1])
    #     return embedding_model
    #
    # def get_embedding(self, X_train):
    #     embedding_model = self.get_embedding_model()
    #     if self.task_model_type == "tensorflow":
    #         embedding = embedding_model.predict(X_train, verbose=0)
    #     else:
    #         embedding = np.zeros((X_train.shape[0], self.embeding_shape))
    #         # TODO fill up

    def query(self, embedding, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(embedding.shape[0], labeled_idx)

        # use the learned representation for the k-greedy-center algorithm:
        new_indices = self.greedy_k_center(embedding[labeled_idx, :], embedding[unlabeled_idx, :], amount)
        return np.hstack((labeled_idx, unlabeled_idx[new_indices]))


if __name__ == "__main__":
    a = np.random.rand(200, 10)
    b = np.random.rand(300, 10)
    tot = np.concatenate((a, b), axis=0)
    model = None
    strategy = CoreSetSampling(model, "pytorch", 500)
    # query 20 new samples from unlabeled data
    new_idx = strategy.query(tot, np.arange(200), 20)

