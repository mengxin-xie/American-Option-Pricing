import numpy as np
from scipy.stats import norm

def price_European_Option(option_type, S, X, T, r, b, v):

    d1 = (np.log(S/X) + (b + v ** 2 / 2) * T) / (v * (T) ** 0.5)
    d2 = d1 - v * (T) ** 0.5

    if option_type == 'Call':
        bsp = S * np.exp((b - r) * T) * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    else:
        bsp = X * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((b - r) * T) * norm.cdf(-d1)

    return bsp


def price_American_Option(option_type_flag, S, X, T, r, b, v):
    '''
    Barone-Adesi-Whaley
    '''

    if option_type_flag == 'Call':
        return approximate_American_Call(S, X, T, r, b, v)
    elif option_type_flag == 'Put':
        return approximate_American_Put(S, X, T, r, b, v)

def approximate_American_Put(S, X, T, r, b, v):
    Sk = Critical_Price_Put(X, T, r, b, v)
    N = 2 * b / v**2
    k = 2 * r / (v**2 * (1 - np.exp(-1 * r * T)))
    d1 = (np.log(Sk / X) + (b + (v**2) / 2) * T) / (v * (T)**0.5)
    Q1 = (-1 * (N - 1) - (((N - 1)**2 + 4 * k))**0.5) / 2
    a1 = -1 * (Sk / Q1) * (1 - np.exp((b - r) * T) * norm.cdf(-1 * d1))

    if S > Sk:
        return price_European_Option('Put', S, X, T, r, b, v) + a1 * (S / Sk)**Q1
    else:
        return X - S


def Critical_Price_Put(X, T, r, b, v):
    N = 2 * b / v ** 2
    m = 2 * r / v ** 2
    q1u = (-1 * (N - 1) - ((N - 1) ** 2 + 4 * m) ** 0.5) / 2
    su = X / (1 - 1 / q1u)
    h1 = (b * T - 2 * v * (T) ** 0.5) * X / (X - su)
    Si = su + (X - su) * np.exp(h1)

    k = 2 * r / (v ** 2 * (1 - np.exp(-1 * r * T)))
    d1 = (np.log(Si / X) + (b + v ** 2 / 2) * T) / (v * (T) ** 0.5)
    Q1 = (-1 * (N - 1) - ((N - 1) ** 2 + 4 * k) ** 0.5) / 2
    LHS = X - Si
    RHS = price_European_Option('Put', Si, X, T, r, b, v) - (
                1 - np.exp((b - r) * T) * norm.cdf(-1 * d1)) * Si / Q1
    bi = -1 * np.exp((b - r) * T) * norm.cdf(-1 * d1) * (1 - 1 / Q1) - (
            1 + np.exp((b - r) * T) * norm.pdf(-d1) / (v * (T) ** 0.5)) / Q1

    E = ITERATION_MAX_ERROR

    while np.abs(LHS - RHS) / X > E:
        Si = (X - RHS + bi * Si) / (1 + bi)
        d1 = (np.log(Si / X) + (b + v ** 2 / 2) * T) / (v * (T) ** 0.5)
        LHS = X - Si
        RHS = price_European_Option('Put', Si, X, T, r, b, v) - (
                    1 - np.exp((b - r) * T) * norm.cdf(-1 * d1)) * Si / Q1
        bi = -np.exp((b - r) * T) * norm.cdf(-1 * d1) * (1 - 1 / Q1) - (
                    1 + np.exp((b - r) * T) * norm.cdf(-1 * d1) / (v * (T) ** 0.5)) / Q1

    return Si


def approximate_American_Call(S, X, T, r, b, v):
    if b >= r:
        return price_European_Option('Call', S, X, T, r, b, v)
    else:
        Sk = Critical_Price_Call(X, T, r, b, v)
        N = 2 * b / v**2
        k = 2 * r / (v**2 * (1 - np.exp(-1 * r * T)))
        d1 = (np.log(Sk / X) + (b + (v**2) / 2) * T) / (v * (T**0.5))
        Q2 = (-1 * (N - 1) + ((N - 1)**2 + 4 * k))**0.5 / 2
        a2 = (Sk / Q2) * (1 - np.exp((b - r) * T) * norm.cdf(d1))
        if S < Sk:
            return price_European_Option('Call', S, X, T, r, b, v) + a2 * (S / Sk)**Q2
        else:
            return S - X


def Critical_Price_Call(X, T, r, b, v):
    N = 2 * b / v ** 2
    m = 2 * r / v ** 2
    q2u = (-1 * (N - 1) + ((N - 1) ** 2 + 4 * m) ** 0.5) / 2
    su = X / (1 - 1 / q2u)
    h2 = -1 * (b * T + 2 * v * (T) ** 0.5) * X / (su - X)
    Si = X + (su - X) * (1 - np.exp(h2))

    k = 2 * r / (v ** 2 * (1 - np.exp(-1 * r * T)))
    d1 = (np.log(Si / X) + (b + v ** 2 / 2) * T) / (v * (T) ** 0.5)
    Q2 = (-1 * (N - 1) + ((N - 1) ** 2 + 4 * k) ** 0.5) / 2
    LHS = Si - X
    RHS = price_European_Option('Call', Si, X, T, r, b, v) + (
                1 - np.exp((b - r) * T) * norm.cdf(d1)) * Si / Q2
    bi = np.exp((b - r) * T) * norm.cdf(d1) * (1 - 1 / Q2) + (
                1 - np.exp((b - r) * T) * norm.pdf(d1) / (v * (T) ** 0.5)) / Q2

    E = ITERATION_MAX_ERROR

    while np.abs(LHS - RHS) / X > E:
        Si = (X + RHS - bi * Si) / (1 - bi)
        d1 = (np.log(Si / X) + (b + v ** 2 / 2) * T) / (v * (T) ** 0.5)
        LHS = Si - X
        RHS = price_European_Option('Call', Si, X, T, r, b, v) + (
                    1 - np.exp((b - r) * T) * norm.cdf(d1)) * Si / Q2
        bi = np.exp((b - r) * T) * norm.cdf(d1) * (1 - 1 / Q2) + (
                    1 - np.exp((b - r) * T) * norm.cdf(d1) / (v * (T) ** 0.5)) / Q2

    return Si

if __name__ == "__main__":
    # spot_price
    S = 36
    # strike_price
    X = 40
    # expiration_time_in_years
    T = 1
    # annual interest rate
    r = 0.06
    # annual carry rate (to price options on stocks b = r)
    b = r
    # annual volatility
    v = 0.2

    ITERATION_MAX_ERROR = 0.001

    BAW_Price = price_American_Option('Put', S, X, T, r, b, v)
    print('BAW_Price: ', BAW_Price)