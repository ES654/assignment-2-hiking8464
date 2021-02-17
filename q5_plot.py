import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))
t = []
d = list(range(1,11))
for i in range(1,11):
    fit_intercept = False
    poly = PolynomialFeatures(degree=i)
    X=pd.DataFrame(poly.transform(x))
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_vectorised(X,y)
    theta = LR.coef_ 
    t.append(max(theta))
fig = plt.figure()
plt.title("Varying theta vs degree")
plt.plot(d,t)
plt.xlabel("degree")
plt.ylabel("theta")
plt.yscale("log")
plt.legend(['theta vs degree'])
plt.savefig("/mnt/c/Users/Hiking/OneDrive/Documents/codes/ML/assignment-2-hiking8464-main/Plot_Q_5/Varying_theta_vs_degree.png")
plt.show()

