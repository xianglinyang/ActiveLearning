import gc
import numpy as np
import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
'''
https://github.com/google/active-learning/tree/efedd8f1c45421ee13af2b9ff593ad31f3835942
'''

# idxs or true/false list
def get_unlabeled_idx(pool_num, lb_idx):
    tot_idx = np.arange(pool_num)
    ulb_idx = np.setdiff1d(tot_idx, lb_idx)
    return ulb_idx


class QueryMethod:
    """
    A general class for query strategies, with a general method for querying examples to be labeled.
    """

    def __init__(self, task_model, task_model_type, n_pool):
        """
        init Query Method
        :param task_model: task_model
        :param task_model_type: "tensorflow" or "pytorch"
        :param n_pool: the number of samples in the pool
        """
        self.task_model = task_model
        self.task_model_type = task_model_type
        self.n_pool = n_pool

    def query(self, *args, **kwargs):
        return NotImplemented

    def update_model(self, new_model):
        del self.task_model
        gc.collect()
        self.task_model = new_model
    
    # def train(self, total_epoch, task_model, complete_dataset):
    #     """
    #     Only train samples from labeled dataset
    #     :return:
    #     """
    #     print("[Training] labeled and unlabeled data")

    #     task_model.to(self.device)
    #     # setting idx_lb
    #     idx_lb_train = self.lb_idxs
    #     train_dataset = Subset(complete_dataset, idx_lb_train)
    #     train_loader = DataLoader(train_dataset, batch_size=self.kwargs['loader_tr_args']['batch_size'], shuffle=True, num_workers=self.kwargs['loader_tr_args']['num_workers'])
    #     optimizer = optim.SGD(
    #         task_model.parameters(), lr=self.kwargs['optimizer_args']['lr'], momentum=self.kwargs['optimizer_args']['momentum'], weight_decay=self.kwargs['optimizer_args']['weight_decay']
    #     )
    #     criterion = torch.nn.CrossEntropyLoss(reduction='none')
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch)

    #     for epoch in range(total_epoch):
    #         task_model.train()

    #         total_loss = 0
    #         n_batch = 0
    #         acc = 0

    #         for inputs, targets in train_loader:
    #             n_batch += 1
    #             inputs, targets = inputs.to(self.device), targets.to(self.device)

    #             optimizer.zero_grad()
    #             outputs = task_model(inputs)
    #             loss = criterion(outputs, targets)
    #             loss = torch.mean(loss)
    #             loss.backward()
    #             optimizer.step()

    #             total_loss += loss.item()
    #             predicted = outputs.argmax(1)
    #             b_acc = 1.0 * (targets == predicted).sum().item() / targets.shape[0]
    #             acc += b_acc

    #         total_loss /= n_batch
    #         acc /= n_batch

    #         if epoch % 50 == 0 or epoch == total_epoch-1:
    #             print('==========Inner epoch {:d} ========'.format(epoch))
    #             print('Training Loss {:.3f}'.format(total_loss))
    #             print('Training accuracy {:.3f}'.format(acc*100))
    #         scheduler.step()
    #     self.update_model(task_model)

