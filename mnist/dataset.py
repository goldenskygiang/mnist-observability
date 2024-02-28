from logging import *
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

MNIST_NORMALIZE_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

MNIST_TRAIN_DATASET = None
MNIST_TEST_DATASET = None

__downloaded__ = False

def init_dataset(dir: str, transform=MNIST_NORMALIZE_TF):
    if __downloaded__:
        log(INFO, "Dataset already downloaded")

    global MNIST_TRAIN_DATASET, MNIST_TEST_DATASET
    MNIST_TRAIN_DATASET = datasets.MNIST(root=dir, download=True, train=True, transform=transform)
    MNIST_TEST_DATASET = datasets.MNIST(root=dir, download=True, train=False, transform=transform)
    __downloaded__ = True

def get_data_loader(train: bool, batch_size: int, shuffle: bool = False):
    if not(__downloaded__):
        raise ValueError("Dataset not downloaded")

    dataset = MNIST_TRAIN_DATASET if train else MNIST_TEST_DATASET

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return loader