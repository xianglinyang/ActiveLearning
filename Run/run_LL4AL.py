'''Active Learning Procedure in PyTorch.
Reference:
- [Yoo et al. 2019] Learning Loss for Active Learning(https://arxiv.org/abs/1905.03677)
- https://github.com/Mephisto405/Learning-Loss-for-Active-Learning/blob/3c11ff7cf96d8bb4596209fe56265630fda19da6/main.py
'''

# Python
import os
import random
import json

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision

# Utils
from tqdm import tqdm

# Custom
from models.LL4ALnet import ResNet18
from models.LL4ALnet import LossNet
from utils import save_datasets, save_task_model

from query_strategies.LL4AL import LL4ALSampling
from args_pool import args_pool
from arguments import get_arguments

if __name__ == "__main__":
    hyperparameters = get_arguments()

    NUM_INIT_LB = hyperparameters.init_num   # 1000
    NUM_QUERY = hyperparameters.query_num    # 1000
    NUM_ROUND = hyperparameters.cycle_num    # 10
    DATA_NAME = hyperparameters.dataset  # CIFAR10
    SAVE = hyperparameters.save  # True
    TOTAL_EPOCH = hyperparameters.epoch_num  # 200
    METHOD = hyperparameters.method
    RESUME = hyperparameters.resume

    # for reproduce purpose
    torch.manual_seed(1331)

    args = args_pool[DATA_NAME]

    if SAVE:
        save_datasets(METHOD, "resnet18", DATA_NAME, **args)

    # start experiment
    n_pool = args['train_num']  # 50000
    n_test = args['test_num']   # 10000

    # loading neural network
    task_model = ResNet18()
    task_model_type = "pytorch"
    if RESUME:
        print('==> Resuming from checkpoint...')
        resume_path = hyperparameters.resume_path
        idxs_lb = json.load(os.path.join(resume_path, "index.json"))
        state_dict = torch.load(os.path.join(resume_path, "subject_model.pth"))
        task_model.load_state_dict(state_dict)
        NUM_INIT_LB = len(idxs_lb)
    else:
        # Generate the initial labeled pool
        idxs_tot = np.arange(n_pool)
        idxs_lb = np.random.choice(n_pool, NUM_INIT_LB, replace=False)
        # np.random.permutation(idxs_tot)[:NUM_INIT_LB]#
    print('number of labeled pool: {}'.format(NUM_INIT_LB))
    print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
    print('number of testing pool: {}'.format(n_test))

    # here the training handlers and testing handlers are different
    train_dataset = torchvision.datasets.CIFAR10(root="..//data//CIFAR10", download=True, train=True, transform=args['transform_tr'])
    test_dataset = torchvision.datasets.CIFAR10(root="..//data//CIFAR10", download=True, train=False, transform=args['transform_te'])
    complete_dataset = torchvision.datasets.CIFAR10(root="..//data//CIFAR10", download=True, train=True, transform=args['transform_te'])

    iters = 0

    #
    def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
        models['backbone'].train()
        models['module'].train()
        global iters

        for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
            inputs = data[0].cuda()
            labels = data[1].cuda()
            iters += 1

            optimizers['backbone'].zero_grad()
            optimizers['module'].zero_grad()

            scores, features = models['backbone'](inputs)
            target_loss = criterion(scores, labels)

            if epoch > epoch_loss:
                # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()
            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            loss            = m_backbone_loss + WEIGHT * m_module_loss

            loss.backward()
            optimizers['backbone'].step()
            optimizers['module'].step()

            # Visualize
            if (iters % 100 == 0) and (vis != None) and (plot_data != None):
                plot_data['X'].append(iters)
                plot_data['Y'].append([
                    m_backbone_loss.item(),
                    m_module_loss.item(),
                    loss.item()
                ])
                vis.line(
                    X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                    Y=np.array(plot_data['Y']),
                    opts={
                        'title': 'Loss over Time',
                        'legend': plot_data['legend'],
                        'xlabel': 'Iterations',
                        'ylabel': 'Loss',
                        'width': 1200,
                        'height': 390,
                    },
                    win=1
                )

    #
    def test(models, dataloaders, mode='val'):
        assert mode == 'val' or mode == 'test'
        models['backbone'].eval()
        models['module'].eval()

        total = 0
        correct = 0
        with torch.no_grad():
            for (inputs, labels) in dataloaders[mode]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                scores, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        return 100 * correct / total

    #
    def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, vis, plot_data):
        print('>> Train a Model.')
        best_acc = 0.
        checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        for epoch in range(num_epochs):
            schedulers['backbone'].step()
            schedulers['module'].step()

            train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis, plot_data)

            # Save a checkpoint
            if False and epoch % 5 == 4:
                acc = test(models, dataloaders, 'test')
                if best_acc < acc:
                    best_acc = acc
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict_backbone': models['backbone'].state_dict(),
                        'state_dict_module': models['module'].state_dict()
                    },
                        '%s/active_resnet18_cifar10.pth' % (checkpoint_dir))
                print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
        print('>> Finished.')

    #
    def get_uncertainty(models, unlabeled_loader):
        models['backbone'].eval()
        models['module'].eval()
        uncertainty = torch.tensor([]).cuda()

        with torch.no_grad():
            for (inputs, labels) in unlabeled_loader:
                inputs = inputs.cuda()
                # labels = labels.cuda()

                scores, features = models['backbone'](inputs)
                pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
                pred_loss = pred_loss.view(pred_loss.size(0))

                uncertainty = torch.cat((uncertainty, pred_loss), 0)

        return uncertainty.cpu()


    ##
    # Main
    if __name__ == '__main__':
        vis = visdom.Visdom(server='http://localhost', port=9000)
        plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}

        for trial in range(TRIALS):
            # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
            indices = list(range(NUM_TRAIN))
            random.shuffle(indices)
            labeled_set = indices[:ADDENDUM]
            unlabeled_set = indices[ADDENDUM:]

            train_loader = DataLoader(cifar10_train, batch_size=BATCH,
                                      sampler=SubsetRandomSampler(labeled_set),
                                      pin_memory=True)
            test_loader  = DataLoader(cifar10_test, batch_size=BATCH)
            dataloaders  = {'train': train_loader, 'test': test_loader}

            # Model
            resnet18    = resnet.ResNet18(num_classes=10).cuda()
            loss_module = lossnet.LossNet().cuda()
            models      = {'backbone': resnet18, 'module': loss_module}
            torch.backends.cudnn.benchmark = False

            # Active learning cycles
            for cycle in range(CYCLES):
                # Loss, criterion and scheduler (re)initialization
                criterion      = nn.CrossEntropyLoss(reduction='none')
                optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                                           momentum=MOMENTUM, weight_decay=WDECAY)
                optim_module   = optim.SGD(models['module'].parameters(), lr=LR,
                                           momentum=MOMENTUM, weight_decay=WDECAY)
                sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
                sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)

                optimizers = {'backbone': optim_backbone, 'module': optim_module}
                schedulers = {'backbone': sched_backbone, 'module': sched_module}

                # Training and test
                train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, vis, plot_data)
                acc = test(models, dataloaders, mode='test')
                print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))

                ##
                #  Update the labeled dataset via loss prediction-based uncertainty measurement

                # Randomly sample 10000 unlabeled data points
                random.shuffle(unlabeled_set)
                subset = unlabeled_set[:SUBSET]

                # Create unlabeled dataloader for the unlabeled subset
                unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=BATCH,
                                              sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
                                              pin_memory=True)

                # Measure uncertainty of each data points in the subset
                uncertainty = get_uncertainty(models, unlabeled_loader)

                # Index in ascending order
                arg = np.argsort(uncertainty)

                # Update the labeled dataset and the unlabeled dataset, respectively
                labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
                unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

                # Create a new dataloader for the updated labeled dataset
                dataloaders['train'] = DataLoader(cifar10_train, batch_size=BATCH,
                                                  sampler=SubsetRandomSampler(labeled_set),
                                                  pin_memory=True)

            # Save a checkpoint
            torch.save({
                'trial': trial + 1,
                'state_dict_backbone': models['backbone'].state_dict(),
                'state_dict_module': models['module'].state_dict()
            },
                './cifar10/train/weights/active_resnet18_cifar10_trial{}.pth'.format(trial))