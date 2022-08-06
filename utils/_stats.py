import pandas as pd
import math
from scipy import stats, optimize
import numpy as np

import time
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import warnings



def cum_returns(returns,
                scale='percentage', starting_value=0, out=None):
    
    """
    Compute cumulative returns from simple returns.
    Parameters
    ----------
    returns : pd.Series, np.ndarray, or pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.
         - Time series with decimal returns.
         - Example::
            2015-07-16   -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902
         - Also accepts two dimensional data. In this case, each column is
           cumulated.
    starting_value : float, optional
       The starting returns.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.
    Returns
    -------
    cumulative_returns : array-like
        Series of cumulative returns.
    """
    if len(returns) == 0:
        return returns.copy()

    nanmask = np.isnan(returns)
    if np.any(nanmask):
        returns = returns.copy()
        returns[nanmask] = 0

    allocated_output = out is None
    if allocated_output:
        out = np.empty_like(returns)
    
    assert scale.upper() in {'PERCENTAGE', 'DOLLAR'},\
    f'input parameter `scale = {scale}` is not recognized'
    
    if scale.upper() == 'PERCENTAGE':
        np.add(returns, 1, out=out)
        out.cumprod(axis=0, out=out)
        if starting_value != 0:
            np.multiply(out, starting_value, out=out)
    else: # dollar space
        np.add(returns, 0, out=out)
        out.cumsum(axis=0, out=out)
        if starting_value != 0:
            np.add(out, starting_value, out=out)

    if allocated_output:
        if returns.ndim == 1 and isinstance(returns, pd.Series):
            out = pd.Series(out, index=returns.index)
        elif isinstance(returns, pd.DataFrame):
            out = pd.DataFrame(
                out, index=returns.index, columns=returns.columns,
            )

    return out



def cum_returns_final(returns,
                      scale='percentage', starting_value=0):
    """
    Compute total returns from simple returns.
    Parameters
    ----------
    returns : pd.DataFrame, pd.Series, or np.ndarray
       Noncumulative simple returns of one or more timeseries.
    starting_value : float, optional
       The starting returns.
    Returns
    -------
    total_returns : pd.Series, np.ndarray, or float
        If input is 1-dimensional (a Series or 1D numpy array), the result is a
        scalar.
        If input is 2-dimensional (a DataFrame or 2D numpy array), the result
        is a 1D array containing cumulative returns for each column of input.
    """
    if len(returns) == 0:
        return np.nan
    
    assert scale.upper() in {'PERCENTAGE', 'DOLLAR'},\
    f'input parameter `scale = {scale}` is not recognized'
    
    if scale.upper() == 'PERCENTAGE':
        result = np.nanprod(returns + 1, axis=0)
        if starting_value != 0:
            result *= starting_value
    else:
        result = np.nansum(returns, axis=0)
        if starting_value != 0:
            result += starting_value
    return result



def aggregate_returns(returns,
                      cum = False, convert_to='yearly', scale='percentage', starting_value=0):
    """
    Aggregates returns by week, month, or year.
    Parameters
    ----------
    returns : pd.Series
       Daily returns of the strategy, *noncumulative*.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    convert_to : str
        Can be 'weekly', 'monthly', or 'yearly'.
    Returns
    -------
    aggregated_returns : pd.Series
    """
        
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    YEARLY = 'yearly'

    def cumulate_returns(x, scale=scale):
        return cum_returns(x, scale=scale).iloc[-1]
    
    if not isinstance(returns.index.get_level_values(0), pd.DatetimeIndex):
        raise TypeError('`returns.index.get_level_values(0)` must be a pd.DatetimeIndex type')
    
    if convert_to == WEEKLY:
        grouping = [lambda x: x[0].year, lambda x: x[0].isocalendar()[1], lambda x: x[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda x: x[0].year, lambda x: x[0].month, lambda x: x[1]]
    elif convert_to == QUARTERLY:
        grouping = [lambda x: x[0].year, lambda x: int(math.ceil(x[0].month/3.)), lambda x: x[1]]
    elif convert_to == YEARLY:
        grouping = [lambda x: x[0].year, lambda x: x[1]]
    else:
        raise ValueError(
            'convert_to must be {}, {} or {}'.format(WEEKLY, MONTHLY, YEARLY, QUARTERLY)
        )
        
    if cum:
        if scale.upper() == 'PERCENTAGE':
            return starting_value * returns.groupby(grouping).apply(cumulate_returns).cumprod()
        else:
            return starting_value + returns.groupby(grouping).apply(cumulate_returns).cumsum()
    else:
        if scale.upper() == 'PERCENTAGE':
            return returns.groupby(grouping).apply(cumulate_returns)
        else:
            return returns.groupby(grouping).apply(cumulate_returns)






def drawdown(returns,
             scale='percentage',out=None):
    
    """
    Determines the maximum drawdown of a strategy.
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.
    Returns
    -------
    max_drawdown : float
    Note
    -----
    See https://en.wikipedia.org/wiki/Drawdown_(economics) for more details.
    """
    
    allocated_output = out is None
    if allocated_output:
        out = np.empty_like(returns)
    
    assert scale.upper() in {'PERCENTAGE', 'DOLLAR'},\
    f'input parameter `scale = {scale}` is not recognized'
    
    returns_1d = returns.ndim == 1

    if len(returns) == 0:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns_array = np.asanyarray(returns)
    
    cumulative = np.empty(
        (returns.shape[0] + 1,) + returns.shape[1:],
        dtype='float64',
    )
    cumulative[0] = start = 100
    
    cum_returns(returns_array, scale=scale, starting_value=start, out=cumulative[1:])
    max_return = np.fmax.accumulate(cumulative, axis=0)
    
    if scale.upper() == 'PERCENTAGE':
        _drawdown = ((cumulative - max_return) / max_return) * 100
    else:
        _drawdown = cumulative - max_return
    
    if allocated_output:
        if returns.ndim == 1 and isinstance(returns, pd.Series):
            out = pd.Series(_drawdown[1:], index=returns.index)
        elif isinstance(returns, pd.DataFrame):
            out = pd.DataFrame(
                _drawdown[1:], index=returns.index, columns=returns.columns,
            )
    
    return out


def max_drawdown(returns, scale='percentage', out=None):
    return drawdown(returns, scale=scale).min()

def average_drawdown(returns, scale='percentage', out=None):
    return drawdown(returns, scale=scale).mean()




APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
QTRS_PER_YEAR = 4

DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
QUARTERLY = 'quarterly'
YEARLY = 'yearly'

ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR,
    QUARTERLY: QTRS_PER_YEAR,
    YEARLY: 1
}

def annualization_factor(period, annualization=None):
    """
    Return annualization factor from period entered or if a custom
    value is passed in.
    Parameters
    ----------
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    Returns
    -------
    annualization_factor : float
    """
    if annualization is None:
        try:
            factor = ANNUALIZATION_FACTORS[period]
        except KeyError:
            raise ValueError(
                "Period cannot be '{}'. "
                "Can be '{}'.".format(
                    period, "', '".join(ANNUALIZATION_FACTORS.keys())
                )
            )
    else:
        factor = annualization
    return factor


def annual_return(returns,
                  scale='PERCENTAGE', period=DAILY, annualization=None):
    """
    Determines the mean annual growth rate of returns. This is equivilent
    to the compound annual growth rate.
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    Returns
    -------
    annual_return : float
        Annual Return as CAGR (Compounded Annual Growth Rate).
    """

    if len(returns) < 1:
        return np.nan
    
    assert scale.upper() in {'PERCENTAGE', 'DOLLAR'},\
    f'input parameter `scale = {scale}` is not recognized'
    
    ann_factor = annualization_factor(period, annualization)
    num_years = len(returns) / ann_factor
    
    if scale.upper() == 'PERCENTAGE':
        # Pass array to ensure index -1 looks up successfully.
        ending_value = cum_returns_final(returns, scale = scale, starting_value=1)
        return ending_value ** (1 / num_years) - 1
    else:
        ending_value = cum_returns_final(returns, scale=scale, starting_value=1)
        return ending_value / num_years



def annual_volatility(returns,
                      period=DAILY,
                      alpha=2.0,
                      annualization=None,
                      out=None):
    """
    Determines the annual volatility of a strategy.
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::
            'monthly':12
            'weekly': 52
            'daily': 252
    alpha : float, optional
        Scaling relation (Levy stability exponent).
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.
    Returns
    -------
    annual_volatility : float
    """
    
    if len(returns) < 2:
        return np.nan
    ann_factor = annualization_factor(period, annualization)
    
    return (ann_factor ** (1.0 / alpha)) * np.std(returns, axis=0)
    
    
    
def calmar_ratio(returns,
                 scale='PERCENTAGE', period=DAILY, annualization=None):
    """
    Determines the Calmar ratio, or drawdown ratio, of a strategy.
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    Returns
    -------
    calmar_ratio : float
        Calmar ratio (drawdown ratio) as float. Returns np.nan if there is no
        calmar ratio.
    Note
    -----
    See https://en.wikipedia.org/wiki/Calmar_ratio for more details.
    """

    max_dd = max_drawdown(returns=returns, scale=scale)
    if max_dd < 0:
        temp = annual_return(
            returns=returns,
            scale=scale,
            period=period,
            annualization=annualization
        ) / abs(max_dd)
    else:
        return np.nan

    if np.isinf(temp):
        return np.nan

    return temp * 100.0




def _adjust_returns(returns, adjustment_factor):
    """
    Returns the returns series adjusted by adjustment_factor. Optimizes for the
    case of adjustment_factor being 0 by returning returns itself, not a copy!
    Parameters
    ----------
    returns : pd.Series or np.ndarray
    adjustment_factor : pd.Series or np.ndarray or float or int
    Returns
    -------
    adjusted_returns : array-like
    """
    if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
        return returns
    return returns - adjustment_factor

def sharpe_ratio(returns,
                 risk_free=0,
                 period=DAILY,
                 annualization=None,
                 out=None):
    """
    Determines the Sharpe ratio of a strategy.
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    risk_free : int, float
        Constant daily risk-free return throughout the period.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.
    Returns
    -------
    sharpe_ratio : float
        nan if insufficient length of returns or if if adjusted returns are 0.
    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    return_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if return_1d:
            out = out.item()
        return out

    returns_risk_adj = np.asanyarray(_adjust_returns(returns, risk_free))
    ann_factor = annualization_factor(period, annualization)

    np.multiply(
        np.divide(
            np.mean(returns_risk_adj, axis=0),
            np.std(returns_risk_adj, ddof=1, axis=0),
            out=out,
        ),
        np.sqrt(ann_factor),
        out=out,
    )
    if return_1d:
        out = out.item()

    return out
    
    
    
    
    
def stability_of_timeseries(returns,
                           scale='PERCENTAGE'):
    """Determines R-squared of a linear fit to the cumulative
    log returns. Computes an ordinary least squares linear fit,
    and returns R-squared.
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    Returns
    -------
    float
        R-squared.
    """
    if len(returns) < 2:
        return np.nan
    
    returns = np.asanyarray(returns)
    returns = returns[~np.isnan(returns)]
    
    assert scale.upper() in {'PERCENTAGE', 'DOLLAR'},\
    f'input parameter `scale = {scale}` is not recognized'
    
    if scale.upper() == 'PERCENTAGE':
        # log-cum return fit
        cum_log_returns = np.log1p(returns).cumsum()
        rhat = stats.linregress(np.arange(len(cum_log_returns)),
                                cum_log_returns)[2]
    else:
        # cum return fit
        cum_returns = returns.cumsum()
        rhat = stats.linregress(np.arange(len(cum_returns)),
                                cum_returns)[2]
    
    return rhat ** 2





def tail_ratio(returns):
    """Determines the ratio between the right (95%) and left tail (5%).
    For example, a ratio of 0.25 means that losses are four times
    as bad as profits.
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
         - See full explanation in :func:`~empyrical.stats.cum_returns`.
    Returns
    -------
    tail_ratio : float
    """

    if len(returns) < 1:
        return np.nan

    returns = np.asanyarray(returns)
    # Be tolerant of nan's
    returns = returns[~np.isnan(returns)]
    if len(returns) < 1:
        return np.nan

    return np.abs(np.percentile(returns, 95)), np.abs(np.percentile(returns, 5))


def value_at_risk(returns, cutoff=0.05):
    """
    Value at risk (VaR) of a returns stream.
    Parameters
    ----------
    returns : pandas.Series or 1-D numpy.array
        Non-cumulative daily returns.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the bottom percentile of
        returns. Defaults to 0.05.
    Returns
    -------
    VaR : float
        The VaR value.
    """
    return np.percentile(returns, 100 * cutoff)


def conditional_value_at_risk(returns, cutoff=0.05):
    """
    Conditional value at risk (CVaR) of a returns stream.
    CVaR measures the expected single-day returns of an asset on that asset's
    worst performing days, where "worst-performing" is defined as falling below
    ``cutoff`` as a percentile of all daily returns.
    Parameters
    ----------
    returns : pandas.Series or 1-D numpy.array
        Non-cumulative daily returns.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the bottom percentile of
        returns. Defaults to 0.05.
    Returns
    -------
    CVaR : float
        The CVaR value.
    """
    # PERF: Instead of using the 'value_at_risk' function to find the cutoff
    # value, which requires a call to numpy.percentile, determine the cutoff
    # index manually and partition out the lowest returns values. The value at
    # the cutoff index should be included in the partition.
    cutoff_index = int((len(returns) - 1) * cutoff)
    return np.mean(np.partition(returns, cutoff_index)[:cutoff_index + 1])





class StatsOfDayPnLBook:
    
    def __init__(self, returns,
                 scale = 'PERCENTAGE', starting_value = 0.):
        
        if not (isinstance(returns, pd.Series) or isinstance(returns, pd.DataFrame)):
            raise TypeError("`returns data` must be a pandas.DataFrame or pd.Series")
            
        assert scale.upper() in {'PERCENTAGE', 'DOLLAR'},\
        f'input parameter `scale = {scale}` is not recognized'
        
        self._returns = returns.copy()
        self._scale = scale
        self._init_val = starting_value
        
    # return metrics
    @property
    def _cumulative_returns_final(self):
        return cum_returns_final(self._returns, self._scale, self._init_val)
    
    @property
    def _annual_return(self):
        return annual_return(self._returns, self._scale)
    
    # drawdown metrics
    @property
    def _max_drawdown(self):
        return max_drawdown(self._returns, self._scale)
    
    @property
    def _average_drawdown(self):
        return average_drawdown(self._returns, self._scale)
    
    # risk metrics
    @property
    def _tail_ratio(self):
        return tail_ratio(self._returns)
    
    @property
    def _value_at_risk(self, cutoff=0.05):
        return value_at_risk(self._returns, cutoff)
    
    @property
    def _conditional_value_at_risk(self, cutoff=0.05):
        return conditional_value_at_risk(self._returns, cutoff)

    # comprehensive metrics
    @property
    def _calmar_ratio(self):
        return calmar_ratio(self._returns, self._scale)
    
    @property
    def _sharpe_ratio(self):
        return sharpe_ratio(self._returns)
    
    
    def run_stats(self):
        if self._scale == 'PERCENTAGE':
            stats_dict = {
                'cumulative final return [%]': (self._cumulative_returns_final - 1.0) * 100.,
                'annual return [%]': self._annual_return * 100,
                'max drawdown [%]': self._max_drawdown,
                'average_drawdown [%]': self._average_drawdown,
                'sharpe ratio': self._sharpe_ratio,
                'calmar ratio': self._calmar_ratio,
                'value at risk [%]': self._value_at_risk * 100
            }
        else: # dollar term
           stats_dict = {
                'cumulative final return [$]': self._cumulative_returns_final - self._init_val,
                'annual return [$]': self._annual_return - self._init_val,
                'max drawdown [%]': self._max_drawdown,
                'average_drawdown [%]': self._average_drawdown,
                'sharpe ratio': self._sharpe_ratio,
                'calmar ratio': self._calmar_ratio,
                'value at risk [$]': self._value_at_risk
            }
            
        # print dictionary
        for k, v in stats_dict.items(): print(k, ' : ', round(v,2))
        return pd.DataFrame.from_dict(stats_dict, orient='index', columns = ['metrics'])
