from logging import *
import random
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from app import config

MNIST_NORMALIZE_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

MNIST_TRAIN_DATASET = None
MNIST_TEST_DATASET = None

def init_dataset(transform=MNIST_NORMALIZE_TF):
    global MNIST_TRAIN_DATASET, MNIST_TEST_DATASET

    dir = config.MNIST_DATASET_DIR
    MNIST_TRAIN_DATASET = datasets.MNIST(root=dir, download=True, train=True, transform=transform)
    MNIST_TEST_DATASET = datasets.MNIST(root=dir, download=True, train=False, transform=transform)

def seed_dataloader(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_data_loader(
        train: bool,
        batch_size: int,
        shuffle: bool = False,
        generator = None):
    init_dataset()
    
    dataset = MNIST_TRAIN_DATASET if train else MNIST_TEST_DATASET
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=seed_dataloader,
        generator=generator)
    
    return loader