import os
import sys
import time
import json

import numpy as np
import torch
import torchvision
from torchvision import transforms

active_learning_path = "/home/xianglin/projects/git_space/ActiveLearning"
sys.path.append(active_learning_path)

from utils import save_datasets, save_task_model, save_new_select
from models.resnet import ResNet18
from query_strategies.bayesianEntropy import bayesianEntropySampling
from args_pool import args_pool
from arguments import get_arguments

if __name__ == "__main__":
    hyperparameters = get_arguments()

    NUM_INIT_LB = hyperparameters.init_num   # 1000
    NUM_QUERY = hyperparameters.query_num    # 1000
    NUM_ROUND = hyperparameters.cycle_num    # 10
    DATA_NAME = hyperparameters.dataset  # "CIFAR10
    SAVE = hyperparameters.save  # True
    TOTAL_EPOCH = hyperparameters.epoch_num  # 200
    METHOD = hyperparameters.method
    RESUME = hyperparameters.resume
    GPU = hyperparameters.gpu

    # record output information
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
    file_path = os.path.join("..", "..", "..", "DVI_data", "active_learning", "bayesianEntropy", "resnet18", DATA_NAME)
    os.system("mkdir -p {}".format(file_path))
    sys.stdout = open(os.path.join(file_path, now+".txt"), "w")

    # for reproduce purpose
    torch.manual_seed(1331)
    np.random.seed(1131)

    args = args_pool[DATA_NAME]

    if SAVE:
        save_datasets(METHOD, "resnet18", DATA_NAME, GPU, **args)

    # loading neural network
    task_model = ResNet18()
    task_model_type = "pytorch"
    # start experiment
    n_pool = args['train_num']  # 50000
    n_test = args['test_num']   # 10000

    if RESUME:
        resume_path = hyperparameters.resume_path
        idxs_lb = np.array(json.load(open(os.path.join(resume_path, "index.json"), "r")))
        state_dict = torch.load(os.path.join(resume_path, "subject_model.pth"))
        task_model.load_state_dict(state_dict)
        NUM_INIT_LB = len(idxs_lb)
    else:
        # Generate the initial labeled pool
        idxs_tot = np.arange(n_pool)
        idxs_lb = np.random.choice(n_pool, NUM_INIT_LB, replace=False)
        # np.random.permutation(idxs_tot)[:NUM_INIT_LB]

    print('number of labeled pool: {}'.format(NUM_INIT_LB))
    print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
    print('number of testing pool: {}'.format(n_test))

    # here the training handlers and testing handlers are different
    train_dataset = torchvision.datasets.CIFAR10(root="..//data//CIFAR10", download=True, train=True, transform=args['transform_tr'])
    test_dataset = torchvision.datasets.CIFAR10(root="..//data//CIFAR10", download=True, train=False, transform=args['transform_te'])
    complete_dataset = torchvision.datasets.CIFAR10(root="..//data//CIFAR10", download=True, train=True, transform=args['transform_te'])

    strategy = bayesianEntropySampling(task_model, task_model_type, n_pool, idxs_lb, 10, DATA_NAME, "resnet18", gpu=GPU, **args)

    # print information
    print(DATA_NAME)
    print(type(strategy).__name__)

    if not RESUME:
        # round 0
        task_m = ResNet18()
        strategy.train(total_epoch=TOTAL_EPOCH, task_model=task_m, complete_dataset=train_dataset)
    accu = strategy.test_accu(test_dataset)
    acc = np.zeros(NUM_ROUND+1)
    acc[0] = accu
    print('Round 0\ntesting accuracy {:.3f}'.format(100*acc[0]))
    if SAVE:
        save_task_model(0, strategy)

    q_time = np.zeros(NUM_ROUND)
    t_time = np.zeros(NUM_ROUND)

    for rd in range(1, NUM_ROUND+1):

        print('================Round {:d}==============='.format(rd))

        # query new samples
        t0 = time.time()
        new_indices = strategy.query(complete_dataset, NUM_QUERY)
        save_new_select(rd-1, strategy, new_indices)
        t1 = time.time()
        print("Query time is {:.2f}".format(t1-t0))
        q_time[rd-1] = t1-t0

        # update
        new_indices = np.hstack((strategy.lb_idxs, new_indices))
        strategy.update_lb_idxs(new_indices)
        resnet_model = ResNet18()
        strategy.train(total_epoch=TOTAL_EPOCH, task_model=resnet_model, complete_dataset=train_dataset)
        t2 = time.time()
        print("Training time is {:.2f}".format(t2-t1))
        t_time[rd-1] = t2-t1

        # compute accuracy at each round
        accu = strategy.test_accu(test_dataset)
        acc[rd] = accu
        print('Accuracy {:.3f}'.format(100*acc[rd]))

        if SAVE:
            save_task_model(rd, strategy)

    # print final results for each round
    print(type(strategy).__name__)
    print(acc)
    print("Query time", q_time)
    print("Training time", t_time)