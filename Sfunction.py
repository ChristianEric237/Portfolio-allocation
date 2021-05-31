# All functions below have been defined for the study of the Statistical Methods for Financial Time Series data !!!

import numpy as np                              #library used to operate mathematical functions multi-dimensional arrays and matrices
import pandas as pd                             #library used for data manipulation and analysis
import yfinance as yf                           #library to import Yahoo finance Tickers data (Close, Volume, ...)
import datetime as dt                           #library for manipulating dates and times
from datetime import timedelta                  #for time shift

import scipy.optimize as sco                    #library for minimizing (or maximizing) objective functions, possibly subject to constraints

import matplotlib.pyplot as plt                 #library used for creating static, animated, and interactive visualizations
import seaborn as sns                           #Data visualization library used to provide a high-level interface for drawing attractive                                                          and informative statistical graphics: (histograms, KDE, box plot, ...)

from scipy import stats                         #library used for statistical analysis
from arch import arch_model                     #library to construct a GARCH model

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf   #Classes used to plot Autocorrelation and Partial autocorrelation function
from statsmodels.tsa.api import adfuller, kpss, acf             #Classes used for data testing: non-Normality and autocorrelation
from statsmodels.tsa import stattools                           #Classes used for Statistical learning and inference algorithms
from statsmodels.stats.diagnostic import het_arch               #Classes used for Engle test

#from statsmodels.tsa.stattools import adfuller

    

def get(tickers_list, tickers_to_names, interval, frequency, start_date, end_date):
    """
    Import data from yahoo finance
    Rename the ticker column with each company name
    Set the frequency as weekly monday
    Extract the Adjusted closed price
    Drop row if all prices at specific date are NA 
    Parameters:
        ------------------------------------------
        tickers_list: list of tickers
        tickers_to_names: list of companies names corresponding to tickers
        start_date: firt day of data needed
        end_date: last day of data needed
        interval: lenght interval between data
        frequency: frequency of data
        ------------------------------------------
    """
    df = yf.download(tickers = tickers_list,
                         start = start_date, end = end_date, interval = interval)

    df.rename(columns = tickers_to_names, level = 1, inplace = True)
    df = df.asfreq(frequency)
    df = df.loc[:,'Adj Close'].copy()
    df.dropna(inplace = True)
    
    return df


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    """
    Compute the annualized performance of the portfolio (returns and volatility)
    Assuming no correlation between returns
    """
    returns = np.sum(mean_returns * weights ) * 52                               
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(52)
    return std, returns


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    Compute the negative sharpe ratio of portfolio of the portfolio in order to minimizes it 
    """
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var


def tangent_portfolio(returns, sep_date, risk_free_rate):
    
    """
    Compute the tangent portfolio with negative sharpe ratio and annualized performance functions as parameters,
    Our portfolio is rebalanced every quarter starting from a fixed date and setup asset allocation
    The parameters are:
        returns: Weekly returns matrix from close price
        sep_date : This date separates in-sample vs out-of-sample, the starting point of trading strategy
        risk_free_rate : interest rate of US Treasury Bond
    """
    idx = pd.IndexSlice                              #Used to slice dataframe
    
    portf_weights = pd.DataFrame(index = returns.index, columns = returns.columns) #Create empty dataframe to store weights
    P_return = pd.DataFrame(index = returns.index, columns = ['P_return']) #Create empty dataframe to store portfolio returns
    
    cov_matrix = returns[:sep_date].cov()                 #Covariance matrix of returns
    mean_returns = returns[:sep_date].mean()              #Mean assets's return
    num_assets = len(mean_returns)                        #Numbers of assets traded
    args = (mean_returns, cov_matrix, risk_free_rate)     #Constraints sum of assets weights == 0
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 0.15)              #Set maximum allocation of each asset 15% to acheive a certain level of diversification
    bounds = tuple(bound for asset in range(num_assets))  
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,    
                          method='SLSQP', bounds=bounds, constraints=constraints)
    
    #Start trading with equally weighted portfolio then switch to a Markovitz (max sharpe ratio)            
    portf_weights.iloc[0] = 1/num_assets            
    P_return.iloc[0] = (portf_weights.iloc[0] * returns.iloc[0]).sum()
    
    nr, nc = returns.loc[idx[:sep_date]].shape      #Numbers of rows and columns of dataframe at separation date
                                                    # nr used as index reference for the separation date
    for t in range(0, nr - 1):                      #Create a loop to store weights on dataframe up to separation date
        portf_weights.iloc[t + 1] = result.x        #Call the results of the minimization
        P_return.iloc[t + 1] = (portf_weights.iloc[t] * returns.iloc[t + 1]).sum() 
                                                    #At each date, we compute the portfolio return
    
    #Create iterative loop with jumps every quarter
    sep_date = dt.datetime.strptime(sep_date, "%Y-%m-%d")   
    for i in range(nr , len(returns)):
        if (i - nr) % 12 == 0: #Check if 3 months has passed since previous allocation (quaterly rebalancing)              
                
            set_quarter = sep_date + timedelta (days = (i - nr)*7)         #Set the next rebalancing date for the portfolio allocation
            cov_matrix = returns.loc[idx[:set_quarter]].cov()              #Covariance matrix of returns each quarter
            mean_returns = returns.loc[idx[:set_quarter]].mean()           #Mean asset return each quarter
            num_assets = len(mean_returns)                                 #Numbers of assets traded
            args = (mean_returns, cov_matrix, risk_free_rate)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #Constraints sum of assets weights == 0
            bound = (0.0, 0.15)              #Set maximum allocation of each asset 15% to acheive a certain level of diversification
            bounds = tuple(bound for asset in range(num_assets))
            result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                                    method='SLSQP', bounds=bounds, constraints=constraints)
                
            #define date filling method from sep_date to each quarter
            for k in range(i, i + 12):                                 #Store weights from separation date up to today after each  rebalancing
                if k <= len(returns) -1 :                              #Check i the loop reached the last line of the dataframe
                    portf_weights.iloc[k] = result.x                   #Call the results of the minimization
                    P_return.iloc[k] = (portf_weights.iloc[k - 1] * returns.iloc[k]).sum() 
                                                                       #Compute portfolio returns with realized assets returns

        i += 1    
    
    return portf_weights, P_return                                     #Return assets's weights and portfolio return evolution across time


def acf_pacf(x, lags):
    """
    Plot the autocorrelation function,
    and the partial autocorrelation function of x,
    Variables:
        x    : data,
        title: to be printed on the graph (optional)
    """
    fig, ax = plt.subplots(1, 2, figsize = (14, 5))
    plot_acf(x, lags = lags, ax = ax[0])
    plot_pacf(x, lags = lags, ax = ax[1])
    
    # add the 95% confidence interval using variance = 1/N
    stderr = stats.norm.ppf(.975) / np.sqrt(len(x))
    ax[0].hlines([stderr, -stderr], 0, lags, colors = 'r', linestyles = 'dotted', label = '95% confidence interval')
    ax[1].hlines([stderr, -stderr], 0, lags, colors = 'r', linestyles = 'dotted', label = '95% confidence interval')
    
    ax[0].legend()
    ax[1].legend()
    
    return fig


def AIC(nll, nparams):
    """
    Compute the Akaike Information Criterion
    Parameters:
      nll : negative maximized loglikelihood of the sample
      nparams : number of free parameters in the model
    """
    aic = 2 * nll + 2 * nparams
    return aic


def BIC(nll, nparams, n):
    """
    Compute the Bayesian Information Criterion
    Parameters:
      nll : negative maximized loglikelihood of the sample
      nparams : number of free parameters in the model
      n : sample size
    """
    bic = 2 * nll + np.log(n) * nparams
    return bic


def Criteria(data):
    """
    Present conclusion of Akaike Information Criterion and 
    Bayesian Information Criterion as dataframe to select 
    the best distribution which suits the data
    
    Parameters:
        data: the dataframe
    """
    mu, sigma = stats.norm.fit(data)
    nu, loc, scale = stats.t.fit(data)

    nll_norm = stats.norm.nnlf([mu, sigma], data)
    nll_t = stats.t.nnlf([nu, loc, scale], data)
    n = len(data)
    nparams_norm = 2
    nparams_t = 3   
    
    norm_IC_values = [AIC(nll_norm, nparams_norm), BIC(nll_norm, nparams_norm, n)]
    t_IC_values = [AIC(nll_t, nparams_t), BIC(nll_t, nparams_t, n)]
    
    criteria = pd.DataFrame({'norm': norm_IC_values, 't': t_IC_values}, index=['AIC', 'BIC'])
    criteria = criteria.round(1)
    
    return criteria


def print_acf(x, lags):
    """
    Ljung-Box Q test for autocorrelation
    """
    val, qstat, pval = acf(x, fft = False, qstat = True, nlags = lags.max())
    df = pd.DataFrame(np.array([qstat[lags-1], 100 * pval[lags-1]]).T,
                      columns = ['Q Statistic', 'p-value (%)'],
                      index = [f'up to lag {lag}' for lag in lags]).round(2)
    display(df)
    return df


def print_engle_test(x):
    """
    Engle test:
        H0: There is no ARCH effect (α1 = . . . = αm = 0)
        H1: There is an ARCH effect (at least one αi <> 0)
    """
    res = het_arch(x, nlags = min(10, len(x) // 5))
    df = pd.DataFrame(data = np.array(res).reshape(2, 2),
                      columns = ['test statistic', 'p-value'],
                      index = ['Lagrange Multiplier', 'F test'])
    print("Engle's ARCH Test")
    display(df.round(4))
    
    return df


def print_adf_test(x):
    """
    This statistical test is testing the data for stationarity.
    The Augmented Dickey-Fuller unit root test:
        H0: There is a unit root 
        H1: There's no unit root, data are stationarity or trend stationarity 
            (set by the regression parameter).
    """
    regressions = {'constant only': 'c',
                   'constant and trend': 'ct',
                   'constant, and linear and quadratic trend': 'ctt',
                   'no constant, no trend': 'nc'}
    df = pd.DataFrame(data = np.zeros((len(regressions),2)),
                      index = regressions,
                      columns = ['test statistic', 'p-value'])
    for reg in regressions:
        res = adfuller(x, regression = regressions[reg])
        df.loc[reg, 'test statistic'] = res[0]
        df.loc[reg, 'p-value'] = res[1]
    display(df.round(4))
    return df


def print_kpss_test(x, regression):
    """
    This statistical test is testing the data for stationarity.
    The Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test:
        H0: either
            regression='c': the data is stationary around a constant (default), or
            regression='ct': the data is stationary around a trend.
        H1: There is a unit root 
    """
    regressions = {'constant only': 'c',
                   'constant and trend': 'ct'}
    df = pd.DataFrame(data = np.zeros((len(regressions),2)),
                      index = regressions,
                      columns = ['test statistic', 'p-value'])
    for reg in regressions:
        res = kpss(x, regression = regressions[reg], nlags = 'auto')
        df.loc[reg, 'test statistic'] = res[0]
        df.loc[reg, 'p-value'] = res[1]
    display(df.round(4))
    return df


def select_garch(x, max_arch, max_garch, mean = 'Zero'):
    """
    Compute and store the AIC and BIC values for all GARCH models
    with 1 <= ARCH lags  <= max_arch
    and  0 <= GARCH lags <= max_garch
    Returns two DataFrames, first with AIC values, second with BIC values
    """
    # create the row and columns indices for the DataFrame
    row_index = pd.Index(data = np.arange(max_arch) + 1, name = 'ARCH Lags')
    col_index = pd.Index(data = np.arange(max_garch + 1), name = 'GARCH Lags')
    # create the DataFrames with zeros everywhere
    aic = pd.DataFrame(data = np.zeros((max_arch, max_garch + 1)),
                       index = row_index,
                       columns = col_index)
    bic = pd.DataFrame(data = np.zeros((max_arch, max_garch + 1)),
                       index = row_index,
                       columns = col_index)
    
    # loop on ARCH lags and GARCH lags
    for i in range(max_arch):
        for j in range(max_garch + 1):
            # estimate the model and save AIC and BIC
            model = arch_model(x, p = i + 1, q = j, mean = mean)
            res = model.fit(update_freq = 0, disp = 'off')
            aic.iloc[i, j] = res.aic
            bic.iloc[i, j] = res.bic
    
    # find the row and column index of the minimum value in the AIC DataFrame
    r_min, c_min = np.unravel_index(np.argmin(aic), aic.shape)
    # store the corresponding order of the GARCH model in a tuple
    aic_min_order = (row_index[r_min], col_index[c_min])
    
    # same for BIC
    r_min, c_min = np.unravel_index(np.argmin(bic), bic.shape)
    bic_min_order = (row_index[r_min], col_index[c_min])
    
    # display the results
    display(aic.round(1))
    print(f'AIC selected order: {aic_min_order}')
    display(bic.round(1))
    print(f'BIC selected order: {bic_min_order}')
    
    # return the two DataFrame
    return aic, bic


def hist_probplot(data, bins = 'auto'):
    """
    The function is build to plot:
        an histogram of the data with the KDE and the best fit by a Normal distribution
        a Normal Probability Plot
    """
    fig, ax = plt.subplots(1, 2, figsize = (13, 6))
    
    sns.histplot(data = data, bins = bins, kde = True, stat = 'density', ax = ax[0])
    stats.probplot(data, plot = ax[1])
    
    xmin, xmax = ax[0].get_xlim()
    x = np.linspace(xmin, xmax, 200)
    y = stats.norm.pdf(x, loc = data.mean(), scale = data.std(ddof = 0))
    ax[0].plot(x, y, 'r')
    ax[0].legend(['KDE', 'best normal fit', 'data'])
    
    return fig


def hist_probplot_t(data, ddof, bins = 'auto', lags = 40):
    """
    The function is build to plot:
        an histogram of the data with the KDE and the best fit by a Normal distribution
        a Q/Q Plot vs. the 𝑡-distribution with ddof degrees of freedom.
    """
    fig, ax = plt.subplots(1, 2, figsize = (13, 6))
    
    sns.histplot(data = data, bins = bins, kde = True, stat = 'density', ax = ax[0])
    stats.probplot(data, plot = ax[1], dist = 't', sparams = ddof)
    
    xmin, xmax = ax[0].get_xlim()
    x = np.linspace(xmin, xmax, 200)
    y = stats.norm.pdf(x, loc = data.mean(), scale = data.std())
    ax[0].plot(x, y, 'r')
    ax[0].legend(['KDE', 'best normal fit', 'data'])
    
    return fig


def garch_cond_vol_forecast(model_forecast):
    """
    For a GARCH model
    Return a pandas Dataframe with forecasts of conditional volatility
    index starts on the next date after forecast_date according to the frequency of the data
    """
    # retrieve frequency of data
    freq = model_forecast.variance.index.freq
    # retrieve forecast horizon
    horizon = model_forecast.variance.shape[1]
    
    # compute conditional volatility
    vol = np.sqrt(model_forecast.variance.iloc[-1].values)
    
    # generates the dates at which the forecasts are computed
    # the first date being the last date of the dataset is ignored 
    d = pd.date_range(model_forecast.variance.index[-1], periods = horizon + 1, freq = freq)[1:]
    
    return pd.Series(data = vol, index = d)