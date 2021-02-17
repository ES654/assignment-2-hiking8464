import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

N = 50
M = 3
X = pd.DataFrame(np.random.randn(N,M))
y = pd.Series(np.random.randn(N))
X = pd.concat([X,X.iloc[:,0]],axis=1)
for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_vectorised(X, y)
    y_hat = LR.predict(X)
    print('VECTORISED RMSE: ', rmse(y_hat, y))
    print('VECTORISED MAE: ', mae(y_hat, y))

