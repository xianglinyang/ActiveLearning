import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image

## to sample label and unlabel data at the same time
# 1. modified dataset
# 2. yield

# only subset of dataset use Subset from torch.utils.data

def get_dataset(DATA_NAME):
    if DATA_NAME == "CIFAR10":
        return get_CIFAR10()


def get_handler(DATA_NAME):
    if DATA_NAME == 'CIFAR10':
        return DataHandler


def get_CIFAR10():
    data_tr = datasets.CIFAR10('data/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10('data/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te


class DataHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)