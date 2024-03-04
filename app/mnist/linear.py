from logging import *
import logging
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.models.enums import ActivationFunction, Optimizer
from app.models.experiment import ExperimentModel, Hyperparam

from app.mnist.dataset import get_data_loader, init_dataset
from app.models.metrics import Metrics

NUM_CLASSES = 10

class MNISTLinearModel(nn.Module):
    def __init__(self, activ_func: ActivationFunction):
        super(MNISTLinearModel, self).__init__()
        self.linear = nn.Linear(28 * 28, 512)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(512, NUM_CLASSES)

        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        self.activ_func = activ_func

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.linear(x)
        x = self.sigmoid(x)
        x = self.fc(x)

        if self.activ_func == ActivationFunction.softmax:
            return self.softmax(x)
        else:
            return self.tanh(x)
    
def init_linear_model(activ_func: ActivationFunction):
    model = MNISTLinearModel(activ_func)
    return model

# https://stackoverflow.com/questions/63645357/using-pytorch-with-celery
def run_model_with_dataloader(
        model: MNISTLinearModel,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        dataloader: DataLoader,
        device: torch.device):
    mean_loss = 0.0

    true_positives = torch.zeros(NUM_CLASSES)
    false_positives = torch.zeros(NUM_CLASSES)
    false_negatives = torch.zeros(NUM_CLASSES)
    correct = 0
    total = 0
    
    start_time = time.time()

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs.to(device)
        labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            truth = labels[i].item()
            pred = predicted[i].item()

            if truth == pred:
                true_positives[truth] += 1
            else:
                false_positives[pred] += 1
                false_negatives[truth] += 1

    end_time = time.time()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Average metrics
    mean_loss /= len(dataloader)
    accuracy = correct / total
    avg_precision = torch.nanmean(precision).cpu().numpy()
    avg_recall = torch.nanmean(recall).cpu().numpy()
    avg_f1_score = torch.nanmean(f1_score).cpu().numpy()
    runtime = end_time - start_time

    results = {
        'loss': mean_loss,
        'accuracy': accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1_score,
        'runtime': runtime
    }

    results = Metrics.model_validate(results)

    return model, results

def train_model(
        model: MNISTLinearModel,
        experiment: ExperimentModel,
        lr: float,
        batch_sz: int):
    logger = logging.getLogger(str(experiment.id))

    args = experiment.hyperparam

    if experiment.seed:
        torch.manual_seed(experiment.seed)

    device = torch.device('cpu')
    if experiment.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')

    logger.info("Start TRAINING phase")
    logger.info(f"Using device: {device}")

    model.to(device)
    model.train()

    train_dataloader = get_data_loader(
        train=True,
        batch_size=batch_sz,
        shuffle=experiment.seed is None)
    
    # TOOD: grid search hyperparams
    optimizer = None
    if args.optimizer == Optimizer.Adam:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer == Optimizer.RMSprop:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    results = []

    for ep in range(args.epochs):
        model, metrics = run_model_with_dataloader(
            model, optimizer, criterion, train_dataloader, device)
        results.append(metrics)
        logger.info(f'Epoch {ep + 1}/{args.epochs}: loss={metrics.loss:.5f}, acc={metrics.accuracy:.5f}, pre={metrics.precision:.5f}, rec={metrics.recall:.5f}, f1={metrics.f1_score:.5f}')

    return model, results

def test_model(
        model: MNISTLinearModel,
        experiment: ExperimentModel,
        lr: float,
        batch_sz: int):
    logger = logging.getLogger(str(experiment.id))

    args = experiment.hyperparam

    if experiment.seed:
        torch.manual_seed(experiment.seed)

    device = torch.device('cpu')
    if experiment.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')

    logger.info("Start TESTING phase")
    logger.info(f"Using device: {device}")

    model.to(device)
    model.eval()

    test_dataloader = get_data_loader(
        train=False,
        batch_size=batch_sz,
        shuffle=experiment.seed is None)
    
    optimizer = None
    if args.optimizer == Optimizer.Adam:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer == Optimizer.RMSprop:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    model, metrics = run_model_with_dataloader(
        model, optimizer, criterion, test_dataloader, device)
    
    logger.info(f'Test result: loss={metrics.loss:.5f}, acc={metrics.accuracy:.5f}, pre={metrics.precision:.5f}, rec={metrics.recall:.5f}, f1={metrics.f1_score:.5f}')

    return model, metrics
