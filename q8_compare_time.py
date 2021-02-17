import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time

N = 50
M = 3
X = pd.DataFrame(np.random.randn(N,M))
y = pd.Series(np.random.randn(N))

fit_intercept = False
LR = LinearRegression(fit_intercept=fit_intercept)
start_time = time.time()
LR.fit_normal(X, y)
end_time = time.time()
Normal_time = end_time - start_time

start_time = time.time()
LR.fit_vectorised(X, y)
end_time = time.time()
Gradien_Time = end_time - start_time

print("Normal_time : ",Normal_time)
print("Gradient_time : ",Gradien_Time)


N = 10
M = list(range(2,500))
normal_time = []
gradient_time = []

for i in M[::5]:
    X = pd.DataFrame(np.random.randn(N,i))
    y = pd.Series(np.random.randn(N))
    fit_intercept = False
    LR = LinearRegression(fit_intercept=fit_intercept)
    start_time = time.time()
    LR.fit_normal(X, y)
    end_time = time.time()
    t = end_time - start_time
    normal_time.append(t)

    start_time = time.time()
    LR.fit_vectorised(X, y)
    end_time = time.time()
    t = end_time - start_time
    gradient_time.append(t)
    

fig_M = plt.figure()
plt.title("Varying M vs Time")
plt.plot(M[::5],normal_time)
plt.plot(M[::5],gradient_time)
plt.xlabel("M")
plt.legend(['Features'])
plt.savefig("/mnt/c/Users/Hiking/OneDrive/Documents/codes/ML/assignment-2-hiking8464-main/Plots_Q_8/Varying_M.png")


M = 5
N = list(range(10,5000))
normal_time = []
gradient_time = []

for i in N[::10]:
    X = pd.DataFrame(np.random.randn(i,M))
    y = pd.Series(np.random.randn(i))
    fit_intercept = False
    LR = LinearRegression(fit_intercept=fit_intercept)
    start_time = time.time()
    LR.fit_normal(X, y)
    end_time = time.time()
    t = end_time - start_time
    normal_time.append(t)

    start_time = time.time()
    LR.fit_vectorised(X, y)
    end_time = time.time()
    t = end_time - start_time
    gradient_time.append(t)
    

fig_N = plt.figure()
plt.title("Varying N vs Time")
plt.plot(N[::10],normal_time)
plt.plot(N[::10],gradient_time)
plt.xlabel("N")
plt.legend(['Samples'])
plt.savefig("/mnt/c/Users/Hiking/OneDrive/Documents/codes/ML/assignment-2-hiking8464-main/Plots_Q_8/Varying_N.png")






