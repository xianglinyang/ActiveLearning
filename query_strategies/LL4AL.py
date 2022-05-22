import numpy as np
import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from query_strategies.query_strategy import QueryMethod
from query_strategies.query_strategy import get_unlabeled_idx
from models.LL4ALnet import LossPredLoss


class LL4ALSampling(QueryMethod):
    """
    The basic uncertainty sampling query strategy, querying the examples with the minimal top confidence.
        adopt from LL4AL repo
        https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
    """

    def __init__(self, model, model_type, n_pool, loss_pred_model, init_lb, num_classes, dataset_name, model_name, gpu=None, **kwargs):

        super(LL4ALSampling, self).__init__(model, model_type, n_pool)
        self.strategy_name = "LL4AL"
        self.loss_pred_model = loss_pred_model
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_classes = num_classes
        self.lb_idxs = init_lb
        if gpu is None:
            self.device = torch.device("cpu")
        elif type(gpu) == str:
            self.device = torch.device("cuda:{}".format(gpu))
        self.kwargs = kwargs

    def query(self, complete_dataset, budget):
        self.loss_pred_model.to(self.device)
        self.loss_pred_model.eval()
        self.task_model.to(self.device)
        self.task_model.eval()

        unlabeled_idx = get_unlabeled_idx(self.n_pool, self.lb_idxs)

        query_set = Subset(complete_dataset, unlabeled_idx)
        query_loader = DataLoader(query_set, shuffle=False, **self.kwargs['loader_te_args'])

        query_num = len(query_set)
        batch_size = self.kwargs['loader_te_args']['batch_size']
        pred = np.zeros((query_num), dtype=np.long)
        with torch.no_grad():
            for idx, (x, y) in enumerate(query_loader):
                x, y = x.to(self.device), y.to(self.device)
                out, features = self.task_model(x)
                pred_loss = self.loss_pred_model(features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                pred[idx*batch_size:(idx+1)*batch_size] = pred_loss.cpu().numpy()

        selected_indices = np.argpartition(pred, query_num - budget)[-budget:]

        # return np.hstack((self.lb_idxs, unlabeled_idx[selected_indices]))
        return unlabeled_idx[selected_indices]

    def update_lb_idxs(self, new_indices):
        self.lb_idxs = new_indices

    def train(self, total_epoch, task_model, loss_pred_model, complete_dataset):

        """
        Only train samples from labeled dataset
        :return:
        """
        print("[Training] labeled and unlabeled data")

        task_model.to(self.device)
        loss_pred_model.to(self.device)

        # setting idx_lb
        idx_lb_train = self.lb_idxs
        train_dataset = Subset(complete_dataset, idx_lb_train)
        train_loader = DataLoader(train_dataset, batch_size=self.kwargs['loader_tr_args']['batch_size'], shuffle=True, num_workers=self.kwargs['loader_tr_args']['num_workers'])
        optimizer_task = optim.SGD(
            task_model.parameters(), lr=self.kwargs['optimizer_args']['lr'], momentum=self.kwargs['optimizer_args']['momentum'], weight_decay=self.kwargs['optimizer_args']['weight_decay']
        )
        optimizer_losspred = optim.SGD(
            loss_pred_model.parameters(), lr=self.kwargs['optimizer_args']['lr'], momentum=self.kwargs['optimizer_args']['momentum'], weight_decay=self.kwargs['optimizer_args']['weight_decay']
        )
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_task, T_max=total_epoch)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_task, milestones=[160]) # official implementation
        sched_module   = torch.optim.lr_scheduler.MultiStepLR(optimizer_losspred, milestones=self.kwargs['milestone'])
    

        for epoch in range(total_epoch):

            task_model.train()
            loss_pred_model.train()

            total_loss = 0
            n_batch = 0
            acc = 0

            for inputs, targets in train_loader:
                n_batch += 1
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer_task.zero_grad()
                optimizer_losspred.zero_grad()

                outputs, features = self.task_model(inputs)
                target_loss = criterion(outputs, targets)

                if epoch > 120:
                    # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
                    features[0] = features[0].detach()
                    features[1] = features[1].detach()
                    features[2] = features[2].detach()
                    features[3] = features[3].detach()
                pred_loss = self.loss_pred_model(features)
                pred_loss = pred_loss.view(pred_loss.size(0))

                m_backbone_loss = torch.mean(target_loss)
                m_module_loss = LossPredLoss(pred_loss, target_loss, margin=1.0)
                loss = m_backbone_loss + 1.0 * m_module_loss

                loss.backward()
                optimizer_task.step()
                optimizer_losspred.step()

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
            sched_module.step()
        del self.task_model
        del self.loss_pred_model
        self.task_model = task_model
        self.loss_pred_model = loss_pred_model
            

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
                out, _ = self.task_model(x)
                p = out.argmax(1)
                pred[idx*batch_size:(idx+1)*batch_size] = p.cpu().numpy()
        return pred

    def test_accu(self, testset):
        pred = self.predict(testset)
        label = np.array(testset.targets)
        return np.sum(pred == label) / float(label.shape[0])
