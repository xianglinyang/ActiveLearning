import numpy as np
import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from query_strategies.query_strategy import QueryMethod
from query_strategies.query_strategy import get_unlabeled_idx


class RandomSampling(QueryMethod):
    """
    The basic uncertainty sampling query strategy, querying the examples with the minimal top confidence.
        adopt from discriminative active learning repo
        https://github.com/dsgissin/DiscriminativeActiveLearning/blob/master/query_methods.py
    """

    def __init__(self, model, model_type, n_pool, init_lb, num_classes, dataset_name, model_name, gpu=True, **kwargs):

        super(RandomSampling, self).__init__(model, model_type, n_pool)
        self.strategy_name = "random"
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_classes = num_classes
        self.lb_idxs = init_lb
        self.device = torch.device("cuda:0" if gpu else "cpu")
        self.kwargs = kwargs

    def query(self, budget):
        unlabeled_idx = get_unlabeled_idx(self.n_pool, self.lb_idxs)
        new_indices = np.random.choice(unlabeled_idx, budget, replace=False)

        return np.hstack((self.lb_idxs, new_indices))

    def update_lb_idxs(self, new_indices):
        self.lb_idxs = new_indices

    def train(self, total_epoch, complete_dataset):

        """
        Only train samples from labeled dataset
        :return:
        """
        print("[Training] labeled and unlabeled data")

        self.task_model.to(self.device)
        # setting idx_lb
        idx_lb_train = self.lb_idxs
        train_dataset = Subset(complete_dataset, idx_lb_train)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=1)
        optimizer = optim.SGD(
            self.task_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        for epoch in range(total_epoch):
            if epoch == total_epoch * 4 // 5:
                optimizer = optim.SGD(
                    self.task_model.parameters(), **self.kwargs['optimizer_args']
                )

            self.task_model.train()

            total_loss = 0
            n_batch = 0
            acc = 0

            for inputs, targets in train_loader:
                n_batch += 1
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.task_model(inputs)
                loss = criterion(outputs, targets)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predicted = outputs.argmax(1)
                b_acc = 1.0 * (targets == predicted).sum().item() / targets.shape[0]
                acc += b_acc

            total_loss /= n_batch
            acc /= n_batch

            if epoch % 50 == 0 or epoch == total_epoch-1:
                print('==========Inner epoch {:d} ========'.format(epoch))
                print('Training Loss {:.3f}'.format(total_loss))
                print('Training accuracy {:.3f}'.format(acc*100))

    def predict(self, testset):
        loader_te = DataLoader(testset, shuffle=False, **self.kwargs['loader_te_args'])
        self.task_model.to(self.device)
        self.task_model.eval()

        test_num = len(testset.targets)
        batch_size = self.kwargs['loader_te_args']['batch_size']
        pred = np.zeros(test_num, dtype=np.long)
        with torch.no_grad():
            for idx, (x, y) in enumerate(loader_te):
                x, y = x.to(self.device), y.to(self.device)
                out = self.task_model(x)
                p = out.argmax(1)
                pred[idx*batch_size:(idx+1)*batch_size] = p.cpu().numpy()
        return pred

    def test_accu(self, testset):
        pred = self.predict(testset)
        label = np.array(testset.targets)
        return np.sum(pred == label) / float(label.shape[0])


