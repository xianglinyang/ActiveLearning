from torchvision import transforms as transforms

args_pool = {
    'CIFAR10':
        {'transform_tr': transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
            'transform_te': transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
            'loader_tr_args': {'batch_size': 128, 'num_workers': 1},
            'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
            'optimizer_args': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4},
            'num_class': 10,
            'train_num': 50000,
            'test_num': 10000,
            'milestone':[160],
        },
    "MNIST": {
        "NET": "resnet18",
        # "transform_tr": transforms.Compose([transforms.ToTensor(),
        #                        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]),
        # "transform_te": transforms.Compose([transforms.ToTensor(),
        #                        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]),
        "transform_tr": transforms.ToTensor(),
        "transform_te":transforms.ToTensor(),
        "loader_tr_args": {"batch_size": 128, "num_workers": 1},
        "loader_te_args": {"batch_size": 1000, "num_workers": 1},
        "optimizer_args": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
        "num_class": 10,
        "train_num": 60000,
        "test_num": 10000,
        "milestone":[10]
    },
}