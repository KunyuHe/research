from datetime import datetime

import numpy as np
import pandas as pd
import pandas_datareader as pdr


def sample_snp_clean(start: datetime, end: datetime,
                     n: int = 50, random_state: int = 123) -> pd.DataFrame:
    """
    Sample S&P components and get their adjusted closing prices over the
    specified period. Drop those with missing values.

    Parameters
    ----------
        start: (datetime) : start date of the interval
        end: (string) : end date of the interval
        n: (int) : number of securities
        random_state: (int) : random generator seed

    Return
    -------
        df: (pd.DataFrame) : Adj Close prices of the sampled S&P components
    """
    # Sample S&P 500 components symbols
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    symbols = pd.read_html(url)[0].Symbol.sample(n=n,
                                                 random_state=random_state)

    # Get adjusted close prices of the sampled components over the specified
    # time interval and drop those with missing entries
    df = pdr.get_data_yahoo(symbols, start, end)['Adj Close'].dropna(axis=1)
    df = df.sample(n=min(df.shape[1], n), axis=1)

    return df


def get_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log return from prices.

    Parameters
    ----------
        prices: (pd.DataFrame) : stock prices

    Return
    -------
        log_ret: (pd.DataFrame) : stock log returns
    """
    log_prices = np.log(prices)
    log_ret = (log_prices - log_prices.shift(1)).iloc[1:]

    return log_ret


def get_dist(log_ret: pd.DataFrame) -> np.ndarray:
    """
    Compute distances from correlations of log returns, defined as
    d_{ij} = \sqrt{2 * (1-\rho_{ij}}.

    Parameters
    ----------
        log_ret: (pd.DataFrame) : stock log returns

    Return
    -------
        dist: (np.ndarray) : distances derived from log return correlations

    References
    ----------
    [1] https://arxiv.org/pdf/1703.00485.pdf, page 2
    """
    corr = log_ret.corr().values
    dist = np.sqrt(2 * (1 - corr))

    return dist
