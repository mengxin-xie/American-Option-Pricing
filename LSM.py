import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

def black_scholes_euro(S0,K,r,T, sigma, option_type):
    d1 =(np.log(S0 /K)+(r +0.5 *sigma ** 2)*T)/(sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "cal1":
        price = S0 *norm.cdf(d1) - K * np.exp(-r *T) * norm.cdf(d2)
    elif option_type == "put":
        price =K * np.exp(-r*T) * norm.cdf(-d2)-S0 * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Please use 'call' or 'put'.")
    return price

def simulate_gbm(M, N, T, S0, mu, sigma):
    np.random.seed(1234)
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


# Least Square Monte Carlo
def longstaff_schwartz(paths, strike, r, option_type):
    # prices at T is recorded in the first row
    paths = paths[::-1]
    # initialize the cash flow matrix
    if option_type == "cal1":
        cash_flows = paths - strike
        cash_flows[cash_flows<0] = 0
    else:
        cash_flows = strike - paths
        cash_flows[cash_flows < 0] = 0
    # number of time steps
    T = cash_flows.shape[0] - 1
    # number of path
    M = cash_flows.shape[1]
    for t in range(1,T):
        # Look at time T-t
        # Create index to only look at in the money paths at time T-t
        in_the_money = paths[t,:] < strike
        # Run Regression to obtain the conditional expectation function
        X = paths[t,in_the_money]
        X2 = X*X
        Xs = np.column_stack([X,X2])
        Y = np.zeros_like(X)
        for j in range(t,0,-1):
            Y += cash_flows[t-j,in_the_money] * np.exp(-r*j)
        model_sklearn = LinearRegression()
        model = model_sklearn.fit(Xs,Y)
        # continuation value
        conditional_exp = model.predict(Xs)
        continuations = np.zeros_like(paths[t,:])
        continuations[in_the_money] = conditional_exp
        # compare the continuation value and excericse value
        ## notice that as the values for OTM and ATM paths in both continuations and cas
        cash_flows[t,:] = np.where(continuations > cash_flows[t,:], 0, cash_flows[t,:])
        # If early exercise is performed, subsequent cashflows = 0
        exercised_early = continuations < cash_flows[t, :]
        cash_flows[0:t,:][:,exercised_early] = 0
    # Return final option price
    discounted_cash_flows = np.zeros((T,M))
    for t in range(T):
        discounted_cash_flows[t] = cash_flows[t] * np.exp(-r * (T-t))
    option_price = discounted_cash_flows.sum(axis=0).sum()/M

    return option_price

if __name__=="__main__":

    # Parameters
    # number of paths
    M = 100000
    # number of steps per year
    N = 50
    # time in years
    T = 1
    # initial stock price
    S0 = 36
    # volatility
    sigma = 0.2
    # strike
    strike = 40
    # annual risk-free interest rate
    r = 0.06

    BS_Price = black_scholes_euro(S0, strike, r, T, sigma, 'put')
    print('BS_Price: ', BS_Price)

    paths = simulate_gbm(M, N, T, S0, r, sigma)

    LSM_Price = longstaff_schwartz(paths, strike, (1+r)**(1/N)-1, 'put')
    print('LSM_Price: ', LSM_Price)


