import torch.nn
from scipy.spatial import distance_matrix
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import tqdm
from keras.models import Model

from query_strategy import QueryMethod as QueryMethod
from query_strategy import get_unlabeled_idx as get_unlabeled_idx


class CoreSetSampling(QueryMethod):
    """
    An implementation of the greedy core set query strategy.
    """

    def __init__(self, model, model_type, n_pool, embedding_shape, init_lb, dataset_name, model_name, gpu=True, **kwargs):

        super(CoreSetSampling, self).__init__(model, model_type, n_pool)
        self.strategy = "coreset"
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.embeding_shape = embedding_shape
        self.lb_idxs = init_lb
        self.device = torch.device("cuda:" if gpu else "cpu")
        self.kwargs = kwargs

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

    def get_embedding_model(self):
        # tensorflow
        if self.task_model_type == "tensorflow":
            embedding_model = Model(input=self.task_model.input, output=self.task_model.get_layer('softmax').input)
        else:
            # pytorch version
            embedding_model = torch.nn.Sequential(*list(self.task_model.children())[:-1])
        return embedding_model

    def get_embedding(self, trainset):
        embedding_model = self.get_embedding_model()
        loader = DataLoader(trainset, shuffle=False, **self.kwargs['loader_te_args'])
        embedding_model.eval()

        train_num = trainset.targets.shape[0]
        batch_size = self.kwargs['loader_te_args']['batch_size']
        embedding = np.zeros(train_num, dtype=np.long)
        with torch.no_grad():
            for idx, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                out = embedding_model(x)
                p = out.view(out.shape[0], -1)
                embedding[idx*batch_size: (idx+1)*batch_size] = p.cpu().numpy()
        return embedding

    def query(self, embedding, amount):
        labeled_idx = self.lb_idxs

        unlabeled_idx = get_unlabeled_idx(embedding.shape[0], labeled_idx)

        # use the learned representation for the k-greedy-center algorithm:
        new_indices = self.greedy_k_center(embedding[labeled_idx, :], embedding[unlabeled_idx, :], amount)
        return np.hstack((labeled_idx, unlabeled_idx[new_indices]))

    def update_lb_idxs(self, new_indices):
        self.lb_idxs = new_indices

    def train(self, total_epoch, complete_dataset):

        """
        Only train samples from labeled dataset
        :return:
        """
        print("[Training] labeled and unlabeled data")

        # setting idx_lb
        idx_lb_train = self.lb_idxs
        train_dataset = Subset(complete_dataset, idx_lb_train)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
        optimizer = optim.SGD(
            self.task_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        for epoch in range(total_epoch):
            if epoch == total_epoch * 4 // 5:
                optimizer = optim.SGD(
                    self.task_model.parameters(), **self.kwargs['transform_tr_args']
                )

            self.task_model.train()

            total_loss = 0
            n_batch = 0
            acc = 0

            progress = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
            for batch_idx, (inputs, targets) in progress:
                n_batch += 1
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.task_model(inputs)
                loss = criterion(outputs, targets)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predicted = outputs.max(1)
                b_acc = 1.0 * (targets == predicted).sum().item() / targets.shape[0]
                acc += b_acc

                progress.set_description('Loss: %.3f | Acc: %.3f' % (
                    total_loss / (batch_idx + 1), 100 * b_acc))

            total_loss /= n_batch
            acc /= n_batch

            print('==========Inner epoch {:d} ========'.format(epoch))
            print('Training Loss {:.3f}'.format(total_loss))
            print('Training accuracy {:.3f}'.format(acc*100))

    def predict(self, testset):

        loader_te = DataLoader(testset, shuffle=False, **self.kwargs['loader_te_args'])
        self.task_model.eval()

        test_num = testset.targets.shape[0]
        batch_size = self.kwargs['loader_te_args']['batch_size']
        pred = np.zeros(test_num, dtype=np.long)
        with torch.no_grad():
            for idx, (x, y) in enumerate(loader_te):
                x, y = x.to(self.device), y.to(self.device)
                out = self.task_model(x)
                p = out.max(1)
                pred[idx*batch_size:(idx+1)*batch_size] = p.cpu().numpy()
        return pred

    def test_accu(self, testset):
        pred = self.predict(testset)
        label = np.array(testset.targets)
        return np.sum(pred == label) / float(label.shape[0])


if __name__ == "__main__":
    a = np.random.rand(200, 10)
    b = np.random.rand(300, 10)
    tot = np.concatenate((a, b), axis=0)
    model = None
    strategy = CoreSetSampling(model, "pytorch", 500)
    # query 20 new samples from unlabeled data
    new_idx = strategy.query(tot, np.arange(200), 20)

