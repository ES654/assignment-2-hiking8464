
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)
batch_size = 5
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    for i in range(2):
        if i == 0:
            LR.fit_non_vectorised(X, y,batch_size)
            y_hat = LR.predict(X)
            print('NON_VECTORISED RMSE: ', rmse(y_hat, y))
            print('NON_VECTORISED MAE: ', mae(y_hat, y))
        if i == 1:
            LR.fit_vectorised(X, y,batch_size)
            y_hat = LR.predict(X)
            print('VECTORISED RMSE: ', rmse(y_hat, y))
            print('VECTORISED MAE: ', mae(y_hat, y))
        else:
            LR.fit_autograd(X, y, batch_size)
            y_hat = LR.predict(X)
            y_hat = LR.predict(X)
            print('AUTOGRAD RMSE: ', rmse(y_hat, y))
            print('AUTOGRAD MAE: ', mae(y_hat, y))   





    
     
    
