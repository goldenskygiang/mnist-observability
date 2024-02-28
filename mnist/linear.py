from logging import *

import torch
import torch.optim as optim
import torch.nn as nn

from app.models.enums import ActivationFunction, LossFunction, Optimizer
from app.models.experiment import Experiment

from mnist.dataset import get_data_loader

class LinearModel(nn.Module):
    def __init__(self, activ_func: ActivationFunction):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(28 * 28, 512)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(512, 10)

        self.softmax = nn.Softmax(10)
        self.tanh = nn.Tanh()

        self.activ_func = activ_func

    def forward(self, x):
        x = x.view(-1)
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
    if args.loss_func == LossFunction.CrossEntropy:
        criterion = nn.CrossEntropyLoss()
    elif args.loss_func == LossFunction.MSE:
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
    
    for ep in range(args.epochs):
        epoch_loss = 0.0

        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_dataloader)
        log(INFO, f'Epoch {ep + 1}/{args.epoch}: loss={epoch_loss:.5f}')

def test(model: LinearModel, experiment: Experiment):
    pass