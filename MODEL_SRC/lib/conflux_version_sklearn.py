#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Replacement from some Sklearn functionality for conflux."""

import numpy as np
import sys
sys.dont_write_bytecode = True


def train_test_split(x,y,test_size,random_state):
    """Training and testing random split on whole data (feature, target) pair.

    Args:

        x (:obj:`numpy.ndarray`): feature array.

        y (:obj:`numpy.ndarray`): target array.

        test_size (:obj:`float`): ratio of testing of total size.

        random_state (:obj:`int`): random seed.

    Returns:
        :obj:`tuple` : train feature, valid feature, train target, valid target, random choice index, number of training data.

    """

    # We set up the number of total data and training data.

    size = x.shape[0]
    train_size = int((1.0 - test_size)*size)

    # We set up random shuffle.

    if test_size == 0:
        
        print 'no validation!'
        x_s = x
        y_s = y

        choice = np.arange(size)
        train_size = size

    else:

        # set up random seed

        np.random.seed(random_state)
        choice = np.random.permutation(size)
        x_s = x[choice,:]
        y_s = y[choice,:]

    x_train = x_s[:train_size, :]
    x_valid = x_s[train_size:, :]
    y_train = y_s[:train_size, :]
    y_valid = y_s[train_size:, :]
    
    return x_train, x_valid, y_train, y_valid, choice, train_size


class StandardScaler(object):
    """Class for standard scaler (Z-normalization) in Sklearn.

    Attributes:

        mean (:obj:`numpy.ndarray`): mean of the data.

        var_ (:obj:`numpy.ndarray`): variance of the data.

        std_ (:obj:`numpy.ndarray`): standard deviation of the data.

    """
    def __init__(self):
        self.mean_ = 0. 
        self.var_ = 0.
        self.std_ = 0.
        
    def fit(self, X):
        """Compute mean, std, var for data X along its first axis.

        It assumes `X` is in ``(n_samples, n_vars)``.

        Args:
            X (:obj:`numpy.ndarray`): input data matrix for fitting the statistics.

        """
        self.mean_ = np.mean(X,axis=0)
        self.std_ = np.std(X,axis=0)
        self.var_ = self.std_**2        

    def fit_transform(self, X):
        """Perform Z normalization on X with the given mean and std.

        Args:
            X (:obj:`numpy.ndarray`): input data matrix to be fitted and transformed.

        Returns:
            :obj:`numpy.ndarray` :  Z-normalization array.

        """
        self.fit(X)        
        return (X-self.mean_)/self.std_

    def transform(self, X):
        """Transform on input matrix `X`

        Args:
            X (:obj:`numpy.ndarray`): input data matrix to be transformed.

        Returns:
            :obj:`numpy.ndarray` : Z-normalization array.

        """
        return (X-self.mean_)/self.std_


