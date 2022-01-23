import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance
from scipy.optimize import minimize

yfinance.pdr_override()
TOLERANCE = 1e-10


def _allocation_risk(weights, covariances):

    # We calculate the risk of the weights distribution
    portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]

    # It returns the risk of the weights distribution
    return portfolio_risk


def _assets_risk_contribution_to_allocation_risk(weights, covariances):

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = np.multiply(
        weights.T, covariances * weights.T) / portfolio_risk

    # It returns the contribution of each asset to the risk of the weights
    # distribution
    return assets_risk_contribution


def _risk_budget_objective_error(weights, args):

    # The covariance matrix occupies the first position in the variable
    covariances = args[0]

    # The desired contribution of each asset to the portfolio risk occupies the
    # second position
    assets_risk_budget = args[1]

    # We convert the weights to a matrix
    weights = np.matrix(weights)

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = _assets_risk_contribution_to_allocation_risk(
        weights, covariances)

    # We calculate the desired contribution of each asset to the risk of the
    # weights distribution
    assets_risk_target = np.asmatrix(
        np.multiply(portfolio_risk, assets_risk_budget))

    # Error between the desired contribution and the calculated contribution of
    # each asset
    error = sum(np.square(assets_risk_contribution -
                          assets_risk_target.T))[0, 0]

    # It returns the calculated error
    return error


def _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):

    # Restrictions to consider in the optimisation: only long positions whose
    # sum equals 100%
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})

    # Optimisation process in scipy
    optimize_result = minimize(fun=_risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances, assets_risk_budget],
                               method='SLSQP',
                               constraints=constraints,
                               tol=TOLERANCE,
                               options={'disp': False})

    # Recover the weights from the optimised object
    weights = optimize_result.x

    # It returns the optimised weights
    return weights


def get_weights(prices):

    # We calculate the covariance matrix
    log_changes = (np.log(prices) -
                   np.log(prices.shift(1))).iloc[1:, :]
    covariances = 365.0 * log_changes.cov().values

    # The desired contribution of each asset to the portfolio risk: we want all
    # asset to contribute equally
    assets_risk_budget = [1 / prices.shape[1]] * prices.shape[1]

    # Initial weights: equally weighted
    init_weights = [1 / prices.shape[1]] * prices.shape[1]

    # Optimisation process of weights
    weights = _get_risk_parity_weights(
        covariances, assets_risk_budget, init_weights)

    # Convert the weights to a pandas Series
    weights = pd.Series(weights, index=prices.columns, name='weight')

    # It returns the optimised weights
    return weights


def get_prices(yahoo_tickers, start_date, end_date):
    prices = (web.DataReader(yahoo_tickers,
                             start_date,
                             end_date
                             )
              .loc[:, 'Adj Close']
              .asfreq('B')  # align time series to business days
              .ffill()      # forward fill missing (NaN) data
              )

    # fix some of yfinance price loading results caveats
    if len(yahoo_tickers) > 1:
        values = {}
        for ticker in yahoo_tickers:
            values[ticker] = prices[ticker].values

        return pd.DataFrame(values, index=prices.index)
    else:
        return pd.DataFrame({yahoo_tickers[0]: prices.values}, index=prices.index)


def find_tickers_with_missing_data(prices):
    result = []
    if prices.isnull().values.any():
        for column in prices.columns:
            if prices[column].isnull().values.any():
                result.append(column)
    return result
