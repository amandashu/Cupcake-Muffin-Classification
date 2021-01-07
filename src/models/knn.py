import pandas as pd
import numpy as np

def helper_knn(series, training, k=5):
    """
    Takes in series of testing data point and training data, calculates the
    distance between testing and every training data point, finds the majority
    training class label for the kth smallest distances
    """
    indexes = training.iloc[:,:-1].apply(lambda x:np.sqrt(sum((x-series)**2)),axis=1).sort_values(ascending=True)[:k].index
    mode = training.iloc[indexes,-1].mode()
    return mode

def knn(training, testing):
    """
    Runs KNN and returns predictions
    """
    predictions = testing.apply(helper_knn, axis=1, args=(training,))
    return predictions
