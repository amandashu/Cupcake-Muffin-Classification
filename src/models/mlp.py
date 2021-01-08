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

# 0.7647
# mlp_hidden_size1 = 10
# mlp_hidden_size2 = 6
# num_iter = 200
# learning_rate = 0.002
# batch_size = 20

# 0.794
mlp_hidden_size1 = 8
num_iter = 700
learning_rate = 0.0005
batch_size = 30


class BaselineMLP(nn.Module):
    def __init__(self):
        """
        A multilayer perceptron model
        Consists of one hidden layer and 1 output layer (all fully connected)
        """
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(mlp_input_size, mlp_hidden_size1)
        self.fc2 = nn.Linear(mlp_hidden_size1, mlp_output_size)
        # self.fc3 = nn.Linear(mlp_hidden_size2, mlp_output_size)

    def forward(self, X):
        """
        Pass the batch of images through each layer of the network, applying
        logistic activation function after hidden layer.
        """
        ### two hidden layers ###
        out = self.fc1(X)
        out = torch.sigmoid(out)
        out = self.fc2(out)
        # out = torch.sigmoid(out)
        # out = self.fc3(out)
        return out

def read_data(X_train, y_train, X_test, y_test):
    X_smalltrain, X_val, y_smalltrain, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=234)
    training = utils.TensorDataset(torch.tensor(X_smalltrain.to_numpy().astype(np.float32)),
                                                torch.tensor(y_smalltrain.to_numpy().astype(np.float32)).long())
    validation = utils.TensorDataset(torch.tensor(X_val.to_numpy().astype(np.float32)),
                                                torch.tensor(y_val.to_numpy().astype(np.float32)).long())
    testing = utils.TensorDataset(torch.tensor(X_test.to_numpy().astype(np.float32)),
                                                torch.tensor(y_test.to_numpy().astype(np.float32)).long())
    train_dataloader = utils.DataLoader(training, batch_size=batch_size, shuffle=True)
    val_dataloader = utils.DataLoader(validation, batch_size=batch_size, shuffle=True)
    test_dataloader = utils.DataLoader(testing, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader


def trainMLP(train_loader, val_loader):
    dirname = '../../results'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    avg_train_loss = []
    avg_val_loss = []
    best_val_score = float('inf')
    net = BaselineMLP()

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

    net = BaselineMLP()
    net.load_state_dict(torch.load(mlp_results_path))

    return net, avg_train_loss, avg_val_loss

def evaluate(loader, net):
    total = 0
    correct = 0
    # use model to get predictions
    for X, y in loader:
        outputs = net(X)
        predictions = torch.argmax(outputs.data, 1)

        # total number of items in dataset
        total += y.shape[0]

        # number of correctly labeled items in dataset
        correct += torch.sum(predictions == y)

    # return fraction of correctly labeled items in dataset
    return float(correct) / float(total)

# import pandas as pd
# recipes = pd.read_csv('../../data/recipes_clean.csv')
# ingredients = recipes.columns[3:]
# X = recipes[ingredients]
# y = recipes['type'].apply(lambda x:1 if x=='muffin' else 0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=234)
def run_mlp(X_train, y_train, X_test, y_test):
    train_loader, val_loader, test_loader = read_data(X_train, y_train, X_test, y_test)
    net, t_losses, v_losses = trainMLP(train_loader,val_loader)
    accuracy = evaluate(test_loader, net)
    print("Test accuracy: {}".format(accuracy))

    # plot losses
    plt.plot(t_losses)
    plt.plot(v_losses)
    plt.legend(["training_loss","validation_loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss plot")
    plt.show()

# run_mlp(X_train, y_train, X_test, y_test)
