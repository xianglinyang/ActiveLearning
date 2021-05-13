import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler as sampler

import numpy as np
import argparse
import random
import os

from custom_datasets import *
import model
import vgg
import resnet
from solver import Solver
from utils import *
import arguments
import json


def cifar_transformer():
    return transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5,],
            #                     std=[0.5, 0.5, 0.5]),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

def main(args):
    # manual seed for reproduce purpose
    torch.manual_seed(1331)
    if args.dataset == 'cifar10':
        test_dataloader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR10(args.data_path)

        args.num_images = 50000
        args.num_val = 5000
        args.budget = 10000
        args.initial_budget = 1000
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        test_dataloader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_path, download=True, transform=cifar_transformer(), train=False),
             batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR100(args.data_path)

        args.num_val = 5000
        args.num_images = 50000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 100

    elif args.dataset == 'imagenet':
        test_dataloader = torch.utils.data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        train_dataset = ImageNet(args.data_path)

        args.num_val = 128120
        args.num_images = 1281167
        args.budget = 64060
        args.initial_budget = 128120
        args.num_classes = 1000
    else:
        raise NotImplementedError

    if args.save:
        # dir
        model_path = os.path.join(".", "Model")
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        training_path = os.path.join(".", "Training_data")
        testing_path = os.path.join(".", "Testing_data")
        if not os.path.exists(training_path):
            os.mkdir(training_path)
        if not os.path.exists(testing_path):
            os.mkdir(testing_path)

        # save dataset
        # device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=True), batch_size=500)
        training_data = torch.Tensor().to(device)
        training_labels = torch.Tensor().to(device)
        for data, target in dataloader:
            training_data = torch.cat((training_data, data), 0)
            training_labels = torch.cat((training_labels, target), 0)
        training_data_path = os.path.join(training_path, "training_dataset_data.pth")
        training_labels_path = os.path.join(training_path, "trainin_dataset_label.pth")
        torch.save(training_data, training_data_path)
        torch.save(training_labels, training_labels_path)

        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=False), batch_size=500)
        testing_data = torch.Tensor().to(device)
        testing_labels = torch.Tensor().to(device)
        for data, target in dataloader:
            testing_data = torch.cat((testing_data, data), 0)
            testing_labels = torch.cat((testing_labels, target), 0)
        testing_data_path = os.path.join(testing_path, "testing_dataset_data.pth")
        testing_labels_path = os.path.join(testing_path, "testing_dataset_label.pth")
        torch.save(testing_data, testing_data_path)
        torch.save(testing_labels, testing_labels_path)

    all_indices = set(np.arange(args.num_images))
    val_indices = random.sample(all_indices, args.num_val)
    all_indices = np.setdiff1d(list(all_indices), val_indices)

    initial_indices = random.sample(list(all_indices), args.initial_budget)
    sampler = torch.utils.data.sampler.SubsetRandomSampler(initial_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    # dataset with labels available
    querry_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=sampler,
            batch_size=args.batch_size, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=val_sampler,
            batch_size=args.batch_size, drop_last=False)
            
    args.cuda = args.cuda and torch.cuda.is_available()
    solver = Solver(args, test_dataloader)

    splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    current_indices = list(initial_indices)

    accuracies = []
    
    for split in range(1, 11, 1):
        # need to retrain all the models on the new images
        # re initialize and retrain the models
        # task_model = vgg.vgg16_bn(num_classes=args.num_classes)
        task_model = resnet.ResNet18()
        vae = model.VAE(args.latent_dim)
        discriminator = model.Discriminator(args.latent_dim)

        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = torch.utils.data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = torch.utils.data.DataLoader(train_dataset,
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)

        # train the models on the current data
        acc, vae, discriminator = solver.train(querry_dataloader,
                                               val_dataloader,
                                               task_model, 
                                               vae, 
                                               discriminator,
                                               unlabeled_dataloader)

        print('Final accuracy with {}% of data is: {:.2f}'.format(int(1000*(split+1)//500), acc))
        accuracies.append(acc)

        sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader)
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = torch.utils.data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=sampler,
                                            batch_size=args.batch_size, drop_last=True)

        if args.save:
            # save subject model and index
            path = os.path.join(".", "Model", "Epoch_{}".format(split))
            if not os.path.exists(path):
                os.mkdir(path)
            subject_model_path = os.path.join(path, "subject_model.pth")
            torch.save(task_model.state_dict(), subject_model_path)

            with open(os.path.join(path, "index.json"), "w") as f:
                json.dump(current_indices, f)

    torch.save(accuracies, os.path.join(args.out_path, args.log_name))

if __name__ == '__main__':
    args = arguments.get_args()
    main(args)

