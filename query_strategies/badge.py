import numpy as np
from scipy import stats
import pdb
import gc
import numpy as np
import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
from keras.models import Model
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import pairwise_distances

from query_strategies.query_strategy import QueryMethod
# from query_strategy import QueryMethod
from query_strategies.query_strategy import get_unlabeled_idx


# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

class BadgeSampling(QueryMethod):
    """
    The badge sampling query strategy, querying the examples considering uncertainty and gradient direcion diversity.
        adopt from badge offical repo
        https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py
    """
    def __init__(self, model, model_type, n_pool, embedding_shape, init_lb, num_classes, dataset_name, model_name, gpu=None, **kwargs):
        super(BadgeSampling, self).__init__(model, model_type, n_pool)
        self.strategy_name = "badge"
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_classes = num_classes
        self.lb_idxs = init_lb
        self.embedding_shape = embedding_shape
        if gpu is None:
            self.device = torch.device("cpu")
        elif type(gpu) == str:
            self.device = torch.device("cuda:{}".format(gpu))
        self.kwargs = kwargs
    
    def get_embedding_model(self):
        # tensorflow
        if self.task_model_type == "tensorflow":
            embedding_model = Model(input=self.task_model.input, output=self.task_model.get_layer('softmax').input)
        else:
            # pytorch version
            embedding_model = torch.nn.Sequential(*list(self.task_model.children())[:-1])
        return embedding_model

    def get_embedding(self, trainset):
        """get the gap embedding of input

        Args:
            trainset (torch dataset): a partial dataset of query set

        Returns:
            embedding: ndarray
        """
        embedding_model = self.get_embedding_model()
        loader = DataLoader(trainset, shuffle=False, **self.kwargs['loader_te_args'])
        embedding_model.to(self.device)
        embedding_model.eval()

        train_num = len(trainset.targets)
        batch_size = self.kwargs['loader_te_args']['batch_size']
        embedding = np.zeros((train_num, self.embedding_shape))
        with torch.no_grad():
            for idx, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                out = embedding_model(x)
                p = out.view(out.shape[0], -1)
                embedding[idx*batch_size:(idx+1)*batch_size] = p.cpu().numpy()
        return embedding

    # gradient embedding for badge (assumes cross-entropy loss)
    def get_grad_embedding(self, trainset):
        embedding = self.get_embedding(trainset=trainset)
        loader = DataLoader(trainset, shuffle=False, **self.kwargs['loader_te_args'])
        
        self.task_model.to(self.device)
        self.task_model.eval()

        train_num = len(trainset.targets)
        batch_size = self.kwargs['loader_te_args']['batch_size']
        grad_embedding = np.zeros(train_num)
            
        with torch.no_grad():
            for idx, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                out = self.task_model(x)
                batchProbs = F.softmax(out, dim=1).data.cpu().numpy()
                pseudo = out.argmax(1)
                one_hot = F.one_hot(pseudo).data.cpu().numpy()
                # times -1
                p = (one_hot - batchProbs).sum(axis=1)
                grad_embedding[idx*batch_size:(idx+1)*batch_size] = p
        
        return embedding* grad_embedding[:, np.newaxis]

    def query(self, complete_dataset, budget):
        unlabeled_idx = get_unlabeled_idx(self.n_pool, self.lb_idxs)
        # query_set = Subset(complete_dataset, unlabeled_idx)
        gradEmbedding = self.get_grad_embedding(complete_dataset)[unlabeled_idx]
        new_indices = init_centers(gradEmbedding, budget)
        scores = np.ones_like(new_indices)
        return unlabeled_idx[new_indices], scores


    def update_lb_idxs(self, new_indices):
        self.lb_idxs = new_indices

    def train(self, total_epoch, task_model, complete_dataset):

        """
        Only train samples from labeled dataset
        :return:
        """
        print("[Training] labeled and unlabeled data")
        np.random.seed()
        seed = np.random.random_integers(1)
        torch.manual_seed(seed)

        self.task_model.to(self.device)
        # task_model.to(self.device)
        # setting idx_lb
        idx_lb_train = self.lb_idxs
        # !Note, two methods here, subset or SubsetRandomSampler inside Dataloader
        train_dataset = Subset(complete_dataset, idx_lb_train)
        train_loader = DataLoader(train_dataset, batch_size=self.kwargs['loader_tr_args']['batch_size'], shuffle=True, num_workers=self.kwargs['loader_tr_args']['num_workers'])
        optimizer = optim.SGD(
            self.task_model.parameters(), lr=self.kwargs['optimizer_args']['lr'], momentum=self.kwargs['optimizer_args']['momentum'], weight_decay=self.kwargs['optimizer_args']['weight_decay']
        )
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch)

        # retrain at each iteration
        for epoch in range(total_epoch):

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
            scheduler.step()
        # del self.task_model
        # self.task_model = task_model

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

if __name__ == "__main__":
    X=np.random.rand(100,2)
    K=10
    id_all = init_centers(X, K)

