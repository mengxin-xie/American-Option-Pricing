import numpy as np
from sklearn.linear_model import LinearRegression

def simulate_gbm(M, N, T, S0, mu, sigma):
    np.random.seed(1234)
    dt = T/N
    dW = np.random.normal(0, np.sqrt(dt),size=(M,N)).T
    St = np.exp((mu-sigma ** 2 / 2) * dt + sigma * dW)
    St = np.vstack([np.ones(M),St])
    St[0] = S0
    St = St.cumprod(axis=0)
    return St

def Parallel(M, N, n, T, S0, sigma, strike, r):

    # discounted price for all iterations - a list
    Price = [0]*n
    # price info from iteration 1 to current iteration - a dictionary with k as the key
    X = dict()
    Y = dict()
    # model info for each time step - a dictionary
    models = dict()

    # initializaiton:
    paths = simulate_gbm(int(N/n), M, T, S0, r, sigma)
    euro_discounted_prices = np.ones_like(paths) * np.exp(-r/M)
    final_payoff = strike - paths[-1]
    euro_discounted_prices[0] = np.where(final_payoff > 0, final_payoff, 0)
    euro_discounted_prices = euro_discounted_prices.cumprod(axis=0)
    euro_discounted_prices = euro_discounted_prices[::-1]
    for k in range(1,M+1):
        x = paths[k]
        x2 = x*x
        x3 = k*np.ones_like(x)
        x4 = k*x
        x5 = k*x2
        xs = np.column_stack([x,x2,x3,x4,x5])
        X[k] = xs
        if k==M:
            Y[k] = euro_discounted_prices[k]
        else:
            Y[k]= euro_discounted_prices[k+1] * np.exp(-r/M)
        model_sklearn = LinearRegression()
        model = model_sklearn.fit(X[k], Y[k])
        models[k] = model

    # iterations
    for i in range(1,n+1):
        # for one interation
        # step 1:paths simulation
        paths =simulate_gbm(int(N/n), M, T, S0, r, sigma)

        # step 2.1: payoff value calculation
        payoff_values = strike - paths
        payoff_values[payoff_values<0] = 0
        # step 2.2: continuation value estimate:
        continuation_values = np.zeros_like(paths)
        for k in range(1,M+1):
            # in each time step, calculate the continuation values for different paths
            # using corresponding regression model(alpha_k)
            x = paths[k]
            x2 = x*x
            x3 = k * np.ones_like(x)
            x4 = k * x
            x5 = k * x2
            xs = np.column_stack([x, x2, x3, x4, x5])
            local_model = models[k]
            continuation_values[k]= local_model.predict(xs)

        # step 3: comparison between continuation value and exrecise value
        discounted_price = payoff_values
        for k in range(M-1,0,-1):
            no_exercise = discounted_price[k] < continuation_values[k]
            discounted_price[k][no_exercise]= discounted_price[k+1][no_exercise] * np.exp(-r/M)

        # step 4: sum of t=1 prices
        Price[i-1]= np.exp(-r/M) * discounted_price[1].sum()

        # step 5:update the regression models
        for k in range(1,M+1):
            x = paths[k]
            x2 = x * x
            x3 = k * np.ones_like(x)
            x4 = k * x
            x5 = k * x2
            xs = np.column_stack([x, x2, x3, x4, x5])
            X[k]= np.vstack([X[k],xs])
            if k==M:
                Y[k] = np.hstack([Y[k],discounted_price[k]])
            else:
                Y[k] = np.hstack([Y[k],discounted_price[k+1] * np.exp(-r/M)])
            model_sklearn = LinearRegression()
            model = model_sklearn.fit(X[k], Y[k])
            models[k] = model

    # weights given to different iterations
    W = np.array([0.99*(i-1) for i in range(1,n+1)])
    W = np.tanh(W)
    W = 1 - 0.5 * (1 - W)
    Price = np.array(Price) / (N/n)
    Option_price = sum(Price * W) / W.sum()

    return Option_price, Price





def Parallel_DeleteInitialPath(M, N, n, T, S0, sigma, strike, r):

    # discounted price for all iterations - a list
    Price = [0]*n
    # price info from iteration 1 to current iteration - a dictionary with k as the key
    X = dict()
    Y = dict()
    # model info for each time step - a dictionary
    models = dict()

    # initializaiton:
    paths = simulate_gbm(int(N/n), M, T, S0, r, sigma)
    euro_discounted_prices = np.ones_like(paths) * np.exp(-r/M)
    final_payoff = strike - paths[-1]
    euro_discounted_prices[0] = np.where(final_payoff > 0, final_payoff, 0)
    euro_discounted_prices = euro_discounted_prices.cumprod(axis=0)
    euro_discounted_prices = euro_discounted_prices[::-1]
    for k in range(1,M+1):
        x = paths[k]
        x2 = x*x
        x3 = k*np.ones_like(x)
        x4 = k*x
        x5 = k*x2
        xs = np.column_stack([x,x2,x3,x4,x5])

        if k==M:
            y = euro_discounted_prices[k]
        else:
            y = euro_discounted_prices[k+1] * np.exp(-r/M)
        model_sklearn = LinearRegression()
        model = model_sklearn.fit(xs, y)
        models[k] = model

    # iterations
    for i in range(1,n+1):
        # for one interation
        # step 1:paths simulation
        paths =simulate_gbm(int(N/n), M, T, S0, r, sigma)

        # step 2.1: payoff value calculation
        payoff_values = strike - paths
        payoff_values[payoff_values<0] = 0
        # step 2.2: continuation value estimate:
        continuation_values = np.zeros_like(paths)
        for k in range(1,M+1):
            # in each time step, calculate the continuation values for different paths
            # using corresponding regression model(alpha_k)
            x = paths[k]
            x2 = x*x
            x3 = k * np.ones_like(x)
            x4 = k * x
            x5 = k * x2
            xs = np.column_stack([x, x2, x3, x4, x5])
            local_model = models[k]
            continuation_values[k]= local_model.predict(xs)

        # step 3: comparison between continuation value and exrecise value
        discounted_price = payoff_values
        for k in range(M-1,0,-1):
            no_exercise = discounted_price[k] < continuation_values[k]
            discounted_price[k][no_exercise]= discounted_price[k+1][no_exercise] * np.exp(-r/M)

        # step 4: sum of t=1 prices
        Price[i-1]= np.exp(-r/M) * discounted_price[1].sum()

        # step 5:update the regression models
        for k in range(1,M+1):
            x = paths[k]
            x2 = x * x
            x3 = k * np.ones_like(x)
            x4 = k * x
            x5 = k * x2
            xs = np.column_stack([x, x2, x3, x4, x5])

            if k not in X:
                X[k] = xs
            else:
                X[k] = np.vstack([X[k], xs])
            if k not in Y:
                if k == M:
                    Y[k] = discounted_price[k]
                else:
                    Y[k] = discounted_price[k + 1] * np.exp(-r / M)
            else:
                if k == M:
                    Y[k] = np.hstack([Y[k], discounted_price[k]])
                else:
                    Y[k] = np.hstack([Y[k], discounted_price[k + 1] * np.exp(-r / M)])

            model_sklearn = LinearRegression()
            model = model_sklearn.fit(X[k], Y[k])
            models[k] = model

    # weights given to different iterations
    W = np.array([0.99*(i-1) for i in range(1,n+1)])
    W = np.tanh(W)
    W = 1 - 0.5 * (1 - W)
    Price = np.array(Price) / (N/n)
    Option_price = sum(Price * W) / W.sum()

    return Option_price, Price


if __name__=="__main__":
    # Parameters
    # number of excersie dates per year
    M = 50
    # total number of paths
    N = 100000
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
    # let annual risk-free interest rate
    r = 0.06


    Parallel_Price, Price = Parallel_DeleteInitialPath(M, N, n, T, S0, sigma, strike, r)
    print('Parallel_Price: ', Parallel_Price, Price)

