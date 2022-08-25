import os
from sklearn.utils import shuffle
import torch
import torchvision
import json


def save_datasets(strategy_n, model_n, dataset_n, gpu=None, **kwargs):
    # strategy path
    strategy_n_path = os.path.join("..", "..", "..", "DVI_data", "active_learning", strategy_n)
    # strategy path
    os.system("mkdir -p {}".format(strategy_n_path))
    # task model path
    model_n_path = os.path.join(strategy_n_path, model_n)
    os.system("mkdir -p {}".format(model_n_path))
    # dataset path
    dataset_n_path = os.path.join(model_n_path, dataset_n)
    training_path = os.path.join(dataset_n_path, "Training_data")
    testing_path = os.path.join(dataset_n_path, "Testing_data")
    os.system("mkdir -p {}".format(training_path))
    os.system("mkdir -p {}".format(testing_path))
    # Model dir
    model_path = os.path.join(dataset_n_path, "Model")
    os.system("mkdir -p {}".format(model_path))

    # save dataset
    # device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    if gpu is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(gpu))
    if dataset_n == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(root='../data/CIFAR10', download=True,
                                                     transform=kwargs['transform_te'], train=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=500)
        test_dataset = torchvision.datasets.CIFAR10(root='../data/CIFAR10', download=True,
                                                    transform=kwargs['transform_te'], train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=500)
    elif dataset_n == "SVHN":
        # TODO
        pass
    elif dataset_n == "CIFAR100":
        # TODO
        pass
    elif dataset_n == "MNIST":
        train_dataset = torchvision.datasets.MNIST("../data/mnist", train=True, download=True, transform=kwargs['transform_te'])
        train_dataloader = torch.utils.data.DataLoader(train_dataset,shuffle=False, batch_size=500)
        test_dataset = torchvision.datasets.MNIST("../data/mnist", train=False, download=True, transform=kwargs['transform_te'])
        test_dataloader = torch.utils.data.DataLoader(test_dataset,shuffle=False, batch_size=500)
    elif dataset_n == "FMNIST":
        train_dataset = torchvision.datasets.MNIST("../data/fmnist", train=True, download=True, transform=kwargs['transform_te'])
        train_dataloader = torch.utils.data.DataLoader(train_dataset,shuffle=False, batch_size=500)
        test_dataset = torchvision.datasets.MNIST("../data/fmnist", train=False, download=True, transform=kwargs['transform_te'])
        test_dataloader = torch.utils.data.DataLoader(test_dataset,shuffle=False, batch_size=500)
    else:
        raise NotImplementedError

    training_data = torch.Tensor().to(device)
    training_labels = torch.Tensor().to(device)
    for data, target in train_dataloader:
        data = data.to(device)
        target = target.to(device)
        training_data = torch.cat((training_data, data), 0)
        training_labels = torch.cat((training_labels, target), 0)
    training_data_path = os.path.join(training_path, "training_dataset_data.pth")
    training_labels_path = os.path.join(training_path, "training_dataset_label.pth")
    torch.save(training_data, training_data_path)
    torch.save(training_labels, training_labels_path)


    testing_data = torch.Tensor().to(device)
    testing_labels = torch.Tensor().to(device)
    for data, target in test_dataloader:
        data = data.to(device)
        target = target.to(device)
        testing_data = torch.cat((testing_data, data), 0)
        testing_labels = torch.cat((testing_labels, target), 0)
    testing_data_path = os.path.join(testing_path, "testing_dataset_data.pth")
    testing_labels_path = os.path.join(testing_path, "testing_dataset_label.pth")
    torch.save(testing_data, testing_data_path)
    torch.save(testing_labels, testing_labels_path)


def save_task_model(n_epoch, strategy):
    # save subject model and index
    working_path = os.path.join("..", "..", "..", "DVI_data", "active_learning", strategy.strategy_name, strategy.model_name, strategy.dataset_name, "Model")
    os.system("mkdir -p {}".format(working_path))
    working_path = os.path.join(working_path, "Iteration_{}".format(n_epoch))
    os.system("mkdir -p {}".format(working_path))
    task_model_path = os.path.join(working_path, "subject_model.pth")
    torch.save(strategy.task_model.state_dict(), task_model_path)

    current_indices = strategy.lb_idxs.tolist()
    with open(os.path.join(working_path, "index.json"), "w") as f:
        json.dump(current_indices, f)


def save_model(n_epoch, strategy, model, name):
    # save subject model and index
    working_path = os.path.join("..", "..", "..", "DVI_data", "active_learning", strategy.strategy_name, strategy.model_name, strategy.dataset_name, "Model")
    os.system("mkdir -p {}".format(working_path))
    working_path = os.path.join(working_path, "Iteration_{}".format(n_epoch))
    os.system("mkdir -p {}".format(working_path))
    model_path = os.path.join(working_path, name+".pth")
    torch.save(model.state_dict(), model_path)


def save_new_select(iteration, strategy, new_selected):
    working_path = os.path.join("..", "..", "..", "DVI_data", "active_learning", strategy.strategy_name, strategy.model_name, strategy.dataset_name, "Model", "Iteration_{}".format(iteration))
    os.system("mkdir -p {}".format(working_path))
    new_selection = new_selected.tolist()
    with open(os.path.join(working_path, "new_selected.json"), "w") as f:
        json.dump(new_selection, f)

