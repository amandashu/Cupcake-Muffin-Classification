"""
Code modified from machine learning course assignment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os

torch.manual_seed(0)
np.random.seed(0)

batch_size = 30

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        A multilayer perceptron model with one hidden layer
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, X):
        """
        Uses logistic activation function after hidden layer.
        """
        out = self.fc1(X)
        out = torch.sigmoid(out)
        out = self.fc2(out)
        return out

def convert_tensor(pd_df):
    """
    Froms pandas dataframe/series to torch tensor
    """
    return torch.tensor(pd_df.to_numpy().astype(np.float32))

def read_data(X_train, y_train, X_test, y_test):
    """
    Takes in training/testing data and returns data loaders their respective
    data loaders.
    """
    testing = utils.TensorDataset(convert_tensor(X_test), convert_tensor(y_test).long())
    training = utils.TensorDataset(convert_tensor(X_train), convert_tensor(y_train).long())
    train_dataloader = utils.DataLoader(training, batch_size=batch_size, shuffle=True)
    test_dataloader = utils.DataLoader(testing, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def train_mlp(train_loader, input_size, hidden_size, output_size, learning_rate, epochs):
    """
    Trains the mlp model, calculates the average training at each epochs, and
    returns the final model and training losses
    """
    # create results folder if not yet exists
    dirname = '../../results'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # list of training losses
    avg_train_loss = []

    # initialize MLP, loss function, and optimizer
    net = MLP(input_size, hidden_size, output_size)
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate)

    i = 0
    while i < epochs:
        # for each training batch, update parameters
        for k, (recipes, type) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(recipes)
            loss = loss_function(outputs, type)

            loss.backward()
            optimizer.step()

        # calculate training losses
        train_loss_list = []
        with torch.no_grad():
            for k, (recipes, type) in enumerate(train_loader):
                outputs = net(recipes)
                loss = loss_function(outputs, type)
                train_loss_list.append(loss.item())

        # calculate average training loss
        avg_train = sum(train_loss_list)/len(train_loss_list)
        avg_train_loss.append(avg_train)

        i += 1
    return net, avg_train_loss

def get_accuracy(test_loader, net):
    """
    Loops through inputted dataloader and calculates accuracy between true values
    and inputted net model's predictions
    """
    total = 0
    correct = 0
    for ingredients, type in test_loader:
        outputs = net(ingredients)
        prediction = torch.argmax(outputs.data, 1)
        total += type.shape[0]
        correct += torch.sum(prediction == type)
    return float(correct) / float(total)

def run_mlp(X_train, y_train, X_test, y_test, input_size, hidden_size, output_size, learning_rate, epochs):
    """
    Takes in training/testing data, trains mlp, prints accuracy score, plot losses
    during training.
    """
    # train model
    train_loader, test_loader = read_data(X_train, y_train, X_test, y_test)
    net, t_losses = train_mlp(train_loader, input_size, hidden_size, output_size, learning_rate, epochs)

    # accuracy
    accuracy = get_accuracy(test_loader, net)
    print("Test accuracy: {}".format(accuracy))

    # plot losses
    plt.plot(t_losses, color='teal')
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("Training Loss plot")
    plt.show()
