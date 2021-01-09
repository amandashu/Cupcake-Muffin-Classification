"""
Code modified from CSE151A course assignment.
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

mlp_results_path = '../../results/model.pt'
mlp_input_size = 6
mlp_output_size = 2
mlp_hidden_size1 = 8
num_iter = 700
learning_rate = 0.0005
batch_size = 30

class MLP(nn.Module):
    def __init__(self):
        """
        A fully connection multilayer perceptron model with one hidden layer
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(mlp_input_size, mlp_hidden_size1)
        self.fc2 = nn.Linear(mlp_hidden_size1, mlp_output_size)

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
    Takes in training/testing data. Training is further split to training and
    validation. Returns data loaders for training, validation, and testing.
    """
    X_smalltrain, X_val, y_smalltrain, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=234)
    training = utils.TensorDataset(convert_tensor(X_smalltrain), convert_tensor(y_smalltrain).long())
    validation = utils.TensorDataset(convert_tensor(X_val), convert_tensor(y_val).long())
    testing = utils.TensorDataset(convert_tensor(X_test), convert_tensor(y_test).long())
    train_dataloader = utils.DataLoader(training, batch_size=batch_size, shuffle=True)
    val_dataloader = utils.DataLoader(validation, batch_size=batch_size, shuffle=True)
    test_dataloader = utils.DataLoader(testing, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader


def train_mlp(train_loader, val_loader):
    dirname = '../../results'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    avg_train_loss = []
    avg_val_loss = []
    best_val_score = float('inf')
    net = MLP()

    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate)

    i = 0
    while i < num_iter:
        train_loss_list = []
        val_loss_list = []

        for k, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for k, (images, labels) in enumerate(val_loader):
                outputs = net(images)
                loss = loss_function(outputs, labels)

                val_loss_list.append(loss.item())

            for k, (images, labels) in enumerate(train_loader):
                outputs = net(images)
                loss = loss_function(outputs, labels)

                train_loss_list.append(loss.item())

        avg_train = sum(train_loss_list)/len(train_loss_list)
        avg_val = sum(val_loss_list)/len(val_loss_list)

        avg_train_loss.append(avg_train)
        avg_val_loss.append(avg_val)

        if avg_val < best_val_score:
            best_val_score = avg_val
            torch.save(net.state_dict(), mlp_results_path)
        i += 1

    net = MLP()
    net.load_state_dict(torch.load(mlp_results_path))

    return net, avg_train_loss, avg_val_loss

def get_accuracy(loader, net):
    """
    Loops through inputted dataloader and calculates accuracy between true values
    and inputted net model's predictions
    """
    total = 0
    correct = 0
    for X, y in loader:
        outputs = net(X)
        predictions = torch.argmax(outputs.data, 1)
        total += y.shape[0]
        correct += torch.sum(predictions == y)
    return float(correct) / float(total)

def run_mlp(X_train, y_train, X_test, y_test):
    """
    Takes in training/testing data, trains mlp, prints accuracy score, plot t_losses
    during training.
    """
    train_loader, val_loader, test_loader = read_data(X_train, y_train, X_test, y_test)
    net, t_losses, v_losses = train_mlp(train_loader,val_loader)
    accuracy = get_accuracy(test_loader, net)
    print("Test accuracy: {}".format(accuracy))

    # plot losses
    plt.plot(t_losses)
    plt.plot(v_losses)
    plt.legend(["training_loss","validation_loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss plot")
    plt.show()
