import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
import autograd.numpy as npa 
from autograd import grad 


class LinearRegression():

    def MSE(self,current):                 
        m = (self.X.T).dot((self.X.dot(current)) - self.y)
        return (np.sum(np.square(m))/len(m))

    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods

        pass

    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        n = len(X) 
        batch_size =len(X)
        temp = lr
        if (self.fit_intercept):
            bias = pd.DataFrame(pd.Series([1.0 for i in range(n)]))
            X = pd.concat([bias,X],axis=1) 
        column_length = len(X.columns)
        theta = np.zeros(column_length)
        for i in range(1,n_iter):
            if (lr_type == 'inverse'):
                lr = temp/i
            current =  theta.copy()
            for j in range(column_length):
                DMSE = 0
                for x in range(batch_size):
                    y_hat = 0
                    for k in range(column_length):
                        y_hat += current[k]*X.iloc[x,k]
                    DMSE += (y_hat - y.iloc[x])*X.iloc[x,j]
                theta[j] = current[j] - DMSE/batch_size
        self.coef_ = theta
        return
        pass

    def fit_vectorised(self, X, y,batch_size = None, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        if batch_size==None:
            batch_size=len(X)
        self.batch_size=batch_size
        self.n_iter = n_iter
        n = len(X)
        batch_size = len(X)
        temp = lr 
        if (self.fit_intercept):
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X = pd.concat([bias,X],axis=1)
        column_length = len(X.columns)
        theta = np.zeros(column_length)
        for i in range(1,n_iter):
            if (lr_type == 'inverse'):
                lr = temp/i
            current =  theta.copy()
            theta = current - (2/batch_size)*lr*(X.T).dot((X.dot(current)) - y)        
        self.coef_ = theta
        pass

    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.n_iter = n_iter
        n = len(X)
        batch_size = len(X)
        temp = lr
        if (self.fit_intercept):
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X = pd.concat([bias,X],axis=1)
        column_length = len(X.columns)
        theta = np.zeros(column_length)
        grad_MSE = grad(self.MSE)
        self.X = X
        self.y = y
        for i in range(1,n_iter):
            if (lr_type == 'inverse'):
                lr = temp/i
            # X_train_values,Y_train_values = batches_of_X[i%total_batches],batches_of_Y[i*total_batches]
            current =  theta.copy()
            theta = current - (2/batch_size)*lr*grad_MSE(current)
        self.coef_ = theta
        pass

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''
        if (self.fit_intercept):
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X = pd.concat([bias,X],axis=1)
        self.coef_ = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        pass

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        X_hat = X.copy()
        if (self.fit_intercept):
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X_hat))]))
            X_hat = pd.concat([bias,X_hat],axis=1) 

        return pd.Series(np.dot(X_hat,self.coef_))
        pass

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """

        pass

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """

        pass

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        pass
