import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression

d = [1,3,5,7,9]
N = list(range(10,15))
for j in N:
    x = np.array([i*np.pi/180 for i in range(60,8*j,4)])
    np.random.seed(10)  #Setting seed for reproducibility
    y = 4*x + 7 + np.random.normal(0,3,len(x))
    t = []
    for i in d:
        fit_intercept = False
        poly = PolynomialFeatures(degree=i)
        X=pd.DataFrame(poly.transform(x))
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit_vectorised(X,y)
        theta = LR.coef_ 
        t.append(max(theta))
    plt.plot(d,t)
plt.title("Varying theta vs degree for varying N")
plt.xlabel("degree")
plt.ylabel("theta")
plt.legend(['theta vs degree'])
plt.yscale("log")
plt.savefig("/mnt/c/Users/Hiking/OneDrive/Documents/codes/ML/assignment-2-hiking8464-main/Plot_Q_6/Varying_theta_vs_degree_for_varying_N.png")
plt.show()