import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def simulate_gbm(M, N, T, S0, mu, sigma):
    # calc each time step
    dt = T/N
    # simulation using numpy arrays
    dW = np.random.normal(0, np.sqrt(dt),size=(M,N)).T
    St = np.exp((mu-sigma ** 2 / 2) * dt + sigma * dW)
    # include array of 1's
    St = np.vstack([np.ones(M),St])
    St[0] = S0
    # multiply through by s0 and return the cumulative product of elements along a given simulation
    St = St.cumprod(axis=0)
    return St

# Parameters
# number of excersie dates
M = 4
# total number of paths
N = 30
# number of iterations
n = 10
# time in years
T = 1
# initial stock price
S0 = 36
# volatility
sigma = 0.2
# strike
strike = 40
# let annual risk free interest rate be 0.06
# option price
Option_price = 0
# discounted price for all iterations - a list
Price = [0]*n
# price info from iteration 1 to current iteration - a dictionary with k as the key
X = dict()
Y = dict()
# model info for each time step - a dictionary
models = dict()


# initializaiton:
paths =simulate_gbm(int(N/n), M, T, S, 0.06, sigma)
euro_discounted_prices = np.ones_like(paths) * np.exp(-0.06)
final_payoff = strike - paths[-1]
euro_discounted_prices[0] = np.where(final_payoff > 0, final_payoff, 0)
euro_discounted_prices = euro_discounted_prices.cumprod(axis=0)
euro_discounted_prices = euro_discounted_prices[::-1]
euro_discounted_prices

for k in range(1,M+1):
    # update x[k] by including the current paths info
    x= paths[k]
    x2 = x*x
    xs = np.column_stack([x,x2])
    X[k] = xs
    if k==M:
        Y[k] = euro_discounted_prices[k]
    else:
        Y[k]= euro_discounted_prices[k+1] * np.exp(-0.06)
        model_sklearn = LinearRegression()
        model = model_sklearn.fit(x[k], Y[k])
    models[k] = model



# for one interation
i=1
# step 1:paths simulation
# stimulate paths in this set
paths =simulate_gbm(int(N/n),M,T,S0,0.06, sigma)
#3 paths and 4 time steps
paths

# step 2.1:payoff value calculation
# assume put option with strike
payoff_values = strike - paths
payoff_values[payoff_values<0] = 0
payoff_values
# step 2.2:continuation value estimate:
# assume a simple continuation value function of y ~ x + x^2 first
continuation_values = np.zeros_like(paths)
for k in range(1,M+1):
    # in each time step, calculate the continuation values for different path
    # using corresponding regression model(alpha_k)
    x = paths[k]
    x2 = x*x
    xs = np.column_stack([x,x2])
    local_model = models[k]
    continuation_values[k]= local_model.predict(xs)


#step 3:comparison between continuation value and exrecise value
discounted_price = payoff_values

for k in range(M-1,0,-1):
    no_exercise = discounted_price[k]<continuation_values[k]
    discounted_price[k][no_exercise]= discounted_price[k+1][no_exercise] * np.exp(-0.06)
discounted_price
# step 4:sum of t=1 prices
Price[i-1]= np.exp(-0.06) * discounted_price[11].sum()
# step 5:update the regression models
for k in range(1,M+1):
    x = paths[k]
    x2 = x*x
    xs = np.column_stack([x,x2])
    x[k]= np.vstack([X[k],xs])
    if k==M:
        Y[k]= np.hstack([Y[k],discounted_price[k]])
    else:
        Y[k]= np.hstack([Y[k],discounted_price[k+1] * np.exp(-0.06)])
    model_sklearn = LinearRegression()
    mode1 = model_sklearn.fit(X[k], Y[k])
    models[k] = model
