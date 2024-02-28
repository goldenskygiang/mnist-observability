from logging import *

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from app.models.enums import ActivationFunction, LossFunction, Optimizer
from app.models.experiment import Experiment

from app.mnist.dataset import get_data_loader, init_dataset
from app.models.metrics import Metrics

class LinearModel(nn.Module):
    def __init__(self, activ_func: ActivationFunction):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(28 * 28, 512)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(512, 10)

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
    
def init_model(activ_func: ActivationFunction):
    model = LinearModel(activ_func)
    return model

def train(model: LinearModel, experiment: Experiment):
    args = experiment.hyperparam

    if experiment.seed:
        torch.manual_seed(experiment.seed)

    device = torch.device('cpu')
    if experiment.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')

    log(INFO, f"Using device: {device}")

    model.to(device)
    model.train()

    train_dataloader = get_data_loader(
        train=True,
        batch_size=args.batch_size,
        shuffle=experiment.seed is None)
    
    optimizer = None
    if args.optimizer == Optimizer.Adam:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == Optimizer.RMSprop:
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    criterion = None
    if args.loss_func == LossFunction.MAE:
        criterion = nn.L1Loss()
    elif args.loss_func == LossFunction.MSE:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    num_classes = 10
    true_positives = torch.zeros(num_classes)
    false_positives = torch.zeros(num_classes)
    false_negatives = torch.zeros(num_classes)
    correct = 0
    total = 0
    
    for ep in range(args.epochs):
        epoch_loss = 0.0

        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs.to(device)
            labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            if args.loss_func == LossFunction.CrossEntropy:
                truths = labels
            else:
                truths = F.one_hot(labels, num_classes=num_classes).float()
            
            loss = criterion(outputs, truths)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

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

        epoch_loss /= len(train_dataloader)
        acc = correct / total
        pre = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (pre * recall) / (pre + recall)

        # Average metrics
        avg_precision = torch.mean(pre)
        avg_recall = torch.mean(recall)
        avg_f1_score = torch.mean(f1_score)
        
        log(INFO, f'Epoch {ep + 1}/{args.epochs}: loss={epoch_loss:.5f}, acc={acc:.5f}, pre={avg_precision:.5f}, rec={avg_recall:.5f}, f1={avg_f1_score:.5f}')

def test(model: LinearModel, experiment: Experiment):
    pass