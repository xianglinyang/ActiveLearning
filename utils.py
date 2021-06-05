import os
import torch
import torchvision
import json


def save_datasets(strategy_n, model_n, dataset_n, **kwargs):
    # output log
    output_path = os.path.join("..", "results")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # strategy path
    strategy_n_path = os.path.join(output_path, strategy_n)
    if not os.path.exists(strategy_n_path):
        os.mkdir(strategy_n_path)

    # task model path
    model_n_path = os.path.join(strategy_n_path, model_n)
    if not os.path.exists(model_n_path):
        os.mkdir(model_n_path)

    # dataset path
    dataset_n_path = os.path.join(model_n_path, dataset_n)
    if not os.path.exists(dataset_n_path):
        os.mkdir(dataset_n_path)

    # dir
    model_path = os.path.join(dataset_n_path, "Model")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    training_path = os.path.join(dataset_n_path, "Training_data")
    testing_path = os.path.join(dataset_n_path, "Testing_data")
    if not os.path.exists(training_path):
        os.mkdir(training_path)
    if not os.path.exists(testing_path):
        os.mkdir(testing_path)

    # save dataset
    # device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    if dataset_n == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(root='..//data//CIFAR10', download=True,
                                                     transform=kwargs['transform_te'], train=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=500)
        test_dataset = torchvision.datasets.CIFAR10(root='..//data//CIFAR10', download=True,
                                                    transform=kwargs['transform_te'], train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=500)
    elif dataset_n == "SVHN":
        # TODO
        pass
    elif dataset_n == "CIFAR100":
        # TODO
        pass
    elif dataset_n == "MNIST":
        # TODO
        pass
    training_data = torch.Tensor().to(device)
    training_labels = torch.Tensor().to(device)
    for data, target in train_dataloader:
        training_data = torch.cat((training_data, data), 0)
        training_labels = torch.cat((training_labels, target), 0)
    training_data_path = os.path.join(training_path, "training_dataset_data.pth")
    training_labels_path = os.path.join(training_path, "training_dataset_label.pth")
    torch.save(training_data, training_data_path)
    torch.save(training_labels, training_labels_path)


    testing_data = torch.Tensor().to(device)
    testing_labels = torch.Tensor().to(device)
    for data, target in test_dataloader:
        testing_data = torch.cat((testing_data, data), 0)
        testing_labels = torch.cat((testing_labels, target), 0)
    testing_data_path = os.path.join(testing_path, "testing_dataset_data.pth")
    testing_labels_path = os.path.join(testing_path, "testing_dataset_label.pth")
    torch.save(testing_data, testing_data_path)
    torch.save(testing_labels, testing_labels_path)


def save_task_model(n_epoch, strategy):
    # save subject model and index
    working_path = os.path.join("..", "results", strategy.strategy_name, strategy.model_name, strategy.dataset_name, "Model")
    if not os.path.exists(working_path):
        os.mkdir(working_path)
    working_path = os.path.join(working_path, "Epoch_{}".format(n_epoch))
    if not os.path.exists(working_path):
        os.mkdir(working_path)
    task_model_path = os.path.join(working_path, "subject_model.pth")
    torch.save(strategy.task_model.state_dict(), task_model_path)

    current_indices = strategy.lb_idxs.tolist()
    with open(os.path.join(working_path, "index.json"), "w") as f:
        json.dump(current_indices, f)
