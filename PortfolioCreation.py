from DataGeneration import *
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
import matplotlib.pyplot as plt
from ReturnForecasting import *
from CovarianceForecasting import *

# Ideas: Create a portfolio that equally invests in all assets that
# an svm algo says will have a positive return

def generateRandomReturns(numAssets, numObs):
    # Creates a df of random annual stock returns between -50% and 50%
    # with numObs data points
    returnVec = np.random.rand(numAssets, numObs)
    return returnVec

def createEfficientFrontier(dfDict):
    # Not completed since it isnt helpful in what we're doing
    returnData = pd.DataFrame()
    for asset in dfDict:
        returnData[asset] = dfDict[asset]["Returns"]

    numAssets = len(returnData.columns)
    numPortfolios = 1
    portWeights = []
    portReturns = []
    portVol = []
    erDF = returnPrediction(dfDict)
    sdDF = createSD(dfDict)

    assets = pd.concat([erDF, sdDF], axis = 1)
    assets.columns = ["Returns", "Volatility"]

    covMatrix = returnData.cov()

    for portfolio in range(numPortfolios):
        weights = [.5, .5]
        portWeights.append(weights)
        returns = np.dot(weights, erDF)
        portReturns.append(returns)
        var = covMatrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        sd = np.sqrt(var)
        annSD = sd*np.sqrt(252)
        portVol.append(annSD)

    data = {"Returns": portReturns, "Volatility": portVol}
    portfolios = pd.DataFrame(data)
    opSpace = pd.concat([portfolios, assets])
    print(opSpace)

    return opSpace


def globalMinVarPortfolio(returnVec, annualize = 1, S = None):
    # Creates a portfolio with the minimum possible
    # variance (not highest sharpe necessarily)
    solvers.options['show_progress'] = False
    n = len(returnVec) # Gets how many assets we are dealing with
    returnVec = np.asmatrix(returnVec)
    # Convert to cvxopt matrices:
    if S is None:
        S = opt.matrix(np.cov(returnVec)) # Covariance matrix
    # Create constraint matrices:
    G = -opt.matrix(np.eye(n)) # Negative nxn identity matrix
    h = opt.matrix(0.0, (n, 1)) # nx1 vector with all zero entries
    hp = opt.matrix(np.zeros((n, 1)))
    A = opt.matrix(1.0, (1,n)) # 1xn vector with all one entries
    b = opt.matrix(1.0) # 1x1 with 1.0 as its entry

    sol = solvers.qp(S, h, G, hp, A, b)
    wGMV = sol['x']
    risk = np.sqrt(blas.dot(wGMV, S * wGMV)) * np.sqrt(annualize)

    returnVec = opt.matrix(np.mean(returnVec, axis=1))
    rtn = blas.dot(wGMV.T, returnVec) * annualize
    return wGMV, risk, rtn

def oneN(returnVec, annualize = 1, S = None):
    # Creates the naive 1/N portfolio where each asset
    # has an equal weighting in the portfolio
    if S is None:
        S = opt.matrix(np.cov(returnVec))
    wgt = 1/len(returnVec)
    risk = np.sqrt(blas.dot(wgt, S * wgt)) * np.sqrt(annualize)
    rtn = returnVec.mean() * annualize
    return wgt, risk, rtn

# Check out
# https://investresolve.com/blog/portfolio-optimization-simple-optimal-methods/
# for ideas on types of portfolios to test

# Below is adapted from
# https://www.quantopian.com/posts/the-efficient-frontier-markowitz-portfolio-optimization-in-python

'''
# Creating a random set of data we can mess around with:
nAssets = 2
nObs = 10
returnVec1 = np.random.rand(nAssets, nObs)

plt.plot(returnVec1.T, alpha=.4)
plt.xlabel('time')
plt.ylabel('returns')
plt.show()
'''

def randWeights(n):
    # Produce n random weights that sum to 1
    k = np.random.rand(n)
    return k / sum(k)

def randomPortfolio(returnVec):
    # Returns the mean and standard deviation of returns for a random portfolio

    p = np.asmatrix(np.mean(returnVec, axis=1)) # Get the average returns for each asset
    w = np.asmatrix(randWeights(returnVec.shape[0])) # Get a vector of weights
    C = np.asmatrix(np.cov(returnVec))  # Creates a cov matrix with returns
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    # This recursion reduces outliers in our data
    if sigma > 2:
        return randomPortfolio(returns)
    return mu, sigma

'''
# Plotting the efficient frontier:
nPortfolios = 50
means, stds = np.column_stack([
    randomPortfolio(returnVec1)
    for _ in range(nPortfolios)
])
plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')
plt.show()
'''



def sharpePortfolio(returnVec):
    # Creates a portfolio with the highest sharpe ratio
    # in an efficient frontier (might be wrong?)
    solvers.options['show_progress'] = False
    n = len(returnVec) # Gets how many assets we are dealing with
    returnVec = np.asmatrix(returnVec)
    N = 100
    mus = [10**(5.0 * t / N - 1.0) for t in range(N)]
    # Convert to cvxopt matrices:
    S = opt.matrix(np.cov(returnVec)) # Covariance matrix
    pbar = opt.matrix(np.mean(returnVec, axis=1)) # Average returns for each asset
    # Create constraint matrices:
    G = -opt.matrix(np.eye(n)) # Negative nxn identity matrix
    h = opt.matrix(0.0, (n, 1)) # nx1 vector with all zero entries
    A = opt.matrix(1.0, (1,n)) # 1xn vector with all one entries
    b = opt.matrix(1.0) # 1x1 with 1.0 as its entry
    # Calculate efficient frontier weights using quadratic programming:
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    # Calculate risks and returns for frontier:
    returnVec = [blas.dot(pbar, x) for x in portfolios]
    riskVec = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    # Calculate the 2nd degree polynomial of the frontier cure:
    m1 = np.polyfit(returnVec, riskVec, 2)
    x1 = np.sqrt(abs(m1[2]/m1[0]))
    # Calculate the optimal portfolio
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returnVec, riskVec

'''
weights, returns, risks = globalMinVarPortfolio(returnVec1)

plt.plot(stds, means, 'o')
plt.ylabel('mean')
plt.xlabel('std')
plt.plot(risks, returns, 'y-o')
plt.show()
print(weights)
'''

# Below is burtons code:

"""

# !/usr/bin/env python
# coding: utf-8

# The code computes the mean-variance frontier for a sample of names, and looks at out of sample performance

# In[1]:


# Import modules

import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt.solvers import options
from cvxopt.solvers import qp
from cvxopt.blas import dot

# In[2]:


# functions for MV optimization

plt.style.use('seaborn-colorblind')


# function to compute the global minimum variance portfolio
def global_min_portfolio(asset_means, asset_cov, n):
    sigma = matrix(asset_cov)
    asset_means = matrix(asset_means)
    P = matrix(asset_cov) # Covariance matrix (S)
    q = matrix(np.zeros((n, 1))) # nx1 vector with all zeroes (h)
    Gp = matrix(-np.identity(n))  # Negative nxn identity matrix (G)
    hp = matrix(np.zeros((n, 1))) # nx1 vector with all zeroes
    A = matrix(1.0, (1, n)) # 1xn vector with all ones (A)
    b = matrix(1.0) # 1x1 with 1.0 as its entry (b)
    options['show_progress'] = False
    sol = qp(P, q, Gp, hp, A, b) # solvers.qp(S, h, G, h, A, b)
    w_gmv = sol['x']
    risk = np.sqrt(dot(w_gmv, P * w_gmv)) * np.sqrt(12) * 100
    rtn = dot(w_gmv, asset_means) * 1200
    return w_gmv, risk, rtn


# function compute the entire frontier
def mean_variance_fronter(asset_means, risk_free, asset_cov, n, r_min, r_max, num_points):
    sigma = matrix(asset_cov)
    asset_means = matrix(asset_means)
    P = matrix(asset_cov)
    q = matrix(np.zeros((n, 1)))
    Gp = matrix(-np.identity(n))  # used to keep weights positive
    A = matrix(1.0, (1, n))
    b = matrix(1.0)
    hp = matrix(np.zeros((n, 1)))
    App = matrix(np.concatenate((np.array(A), np.array(asset_means))))
    Step = (r_max - r_min) / np.float(num_points)
    mus = [r_min + t * Step for t in range(num_points)]
    ws = [qp(P, q, Gp, hp, App, matrix(np.concatenate((b, mu * b))))['x'] for mu in mus]
    returns = np.ravel([dot(asset_means, w) * 1200 for w in ws])
    risks = np.ravel([np.sqrt(dot(w, sigma * w)) * np.sqrt(12) * 100 for w in ws])
    Sharpes = (returns - risk_free) / risks
    return returns, risks, Sharpes, ws


# function to compute the risk parity strategy
def risk_parity_portfolio(asset_means, asset_cov, n):
    sigma = matrix(asset_cov)
    asset_means = matrix(asset_means)
    w = 1 / np.sqrt(np.diag(asset_cov))
    w = matrix(w / np.sum(w))
    risk = np.sqrt(dot(w, sigma * w)) * np.sqrt(12) * 100
    rtn = dot(w, asset_means) * 1200
    return w, risk, rtn


# function to compute the equally weighted strategy
def equally_weighted_portfolio(asset_means, asset_cov, n):
    sigma = matrix(asset_cov)
    asset_means = matrix(asset_means)
    w = np.ones(n)
    w = matrix(w / np.sum(w))
    risk = np.sqrt(dot(w, sigma * w)) * np.sqrt(12) * 100
    rtn = dot(w, asset_means) * 1200
    return w, risk, rtn


# In[3]:


# Download data

start = datetime.datetime(2005, 12, 31)
end = datetime.datetime(2020, 12, 31)
return_data = pd.DataFrame()

Tbill = pdr.get_data_fred("DGS1MO", start, end)
riskfree = Tbill.resample('M').last()
names = ['IVV', 'AGG']
for i in names:
    data = pdr.DataReader(i, 'yahoo', start, end)
    monthlydata = data.resample('M').last()
    return_data[i] = monthlydata['Adj Close'] / monthlydata['Adj Close'].shift(1) - 1.
return_data['rf'] = riskfree / 1200
upper_bound = np.max(np.sqrt(np.diag(return_data.cov()))
                     * np.sqrt(12) * 100) * 1
lower = 0.00
return_data = return_data.dropna()

# In[4]:


# mean vector, covariance and standard deviations matrix for the entire sample

E = np.matrix(return_data[names].mean())
Sigma = np.matrix(return_data[names].cov())
Corr_matrix = np.matrix(return_data[names].corr())
risk_matrix = np.matrix(return_data[names].std()) * np.sqrt(12) * 100

rf_table = return_data['rf'].mean() * 1200
print(' ETF |Avg Rtn | Std Dev| SR', end='\n')
print('|:--|:--:|:--:|:--:', end='\n')
print('Avg Risk free |{:.2f}% | |'.format(return_data['rf'].mean() * 1200), end='\n')

for i in range(len(names)):
    print('{:.6} | {:.2f}% | {:.2f}% |{:.2f} '.format(names[i], E[0, i] * 1200, risk_matrix[0, i],
                                                      ((E[0, i] * 1200 - rf_table) / risk_matrix[0, i])))

print('# Correlation Matix')

corr = return_data[names].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
fig = plt.figure()
sns.heatmap(corr, mask=mask, linewidths=0.5)
plt.show()

# In[5]:


# compute frontier
w_gmv, risk_gmv, rtn_gmv = global_min_portfolio(E, Sigma, E.shape[1])
r_max = np.max(E) * 1
r_min = np.min(E) * 1
avg_rf = return_data['rf'].mean()
n = E.shape[1]
rtns, risks, Sharpes, ws = mean_variance_fronter(E, avg_rf, Sigma, n, r_min, r_max, 200)
i_star = np.argmax(Sharpes)
risk_sr = risks[i_star]
rtn_sr = rtns[i_star]
w_rp, risk_rp, rtn_rp = risk_parity_portfolio(E, Sigma, E.shape[1])
w_ew, risk_ew, rtn_ew = equally_weighted_portfolio(E, Sigma, E.shape[1])

# In[6]:


# Plot the frontier over the entire sample period

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
borrowing_spread = 2
borrowing_rate = avg_rf * 1200 + borrowing_spread

ax1.plot(risks, rtns)
ax1.scatter(risk_gmv, rtn_gmv, c='black', marker='.')
ax1.annotate('GMV', (risk_gmv + 0.30, rtn_gmv - 0.25), fontsize=10)
ax1.scatter(risk_rp, rtn_rp, c='black', marker='.')
ax1.annotate('RP', (risk_rp + 0.30, rtn_rp - 0.25), fontsize=10)
ax1.scatter(risk_ew, rtn_ew, c='black', marker='.')
ax1.annotate('EW', (risk_ew + 0.30, rtn_ew - 0.25), fontsize=10)

for i in names:
    m = return_data[i].mean() * 1200
    s = return_data[i].std() * np.sqrt(12) * 100
    ax1.scatter(s, m, c='black', marker='.')
    ax1.annotate(i, (s + 0.30, m - 0.25), fontsize=10)
ax1.set_xlabel('Annualized $\sigma_p$%')
ax1.set_title('No borrowing or lending')
i_star = np.argmax(Sharpes)
risk_values = np.linspace(0, risks[i_star], 100)
Sharpes_leverage = (rtns - avg_rf - borrowing_spread) / risks
ib_star = np.argmax(Sharpes_leverage)
return_values = avg_rf + np.max(Sharpes) * risk_values
risk_values_leverage = np.linspace(risks[ib_star], risks[np.argmax(rtns)] * 1.2, 100)
return_values_leverage = avg_rf + borrowing_spread + np.max(Sharpes_leverage) * risk_values_leverage
ax2.plot(risks, rtns)
ax2.plot(risk_values, return_values, c='black')
ax2.scatter(risks[i_star], rtns[i_star], c='black', marker='.')
ax2.annotate('$SR^*_{lending}$', (risks[i_star] + 0.30, rtns[i_star] - 0.25), fontsize=10)
ax2.plot(risk_values_leverage, return_values_leverage, c='black')
ax2.scatter(risks[ib_star], rtns[ib_star], c='black', marker='.')
ax2.annotate('$SR^*_{borrowing}$', (risks[ib_star] + 0.30, rtns[ib_star] - 0.25), fontsize=10)
ax2.scatter(risk_gmv, rtn_gmv, c='black', marker='.')
ax2.annotate('GMV', (risk_gmv + .30, rtn_gmv - 0.25), fontsize=10)
ax2.set_xlabel('Annualized $\sigma_p$%')
ax1.set_ylabel('Annualized $\hat E[r_p]$%')
temp = ax1.set_xlim([np.floor(np.min(risks * .0)), np.max(risks) * 1.75])
ax2.set_title('$r^b_f=$' + str(round(borrowing_rate, 2)) + '%$\geq r^l_f=$' + str(round(avg_rf * 1200, 2)) + '%')
plt.show()

# In[7]:


# Report porfolios

w_lend_star = ws[i_star]
w_borrow_star = ws[ib_star]

print('Name | $w^*_{lend}$| $w_{GMV}$ |$w_{rp}$ |$w_{ew}$')
print('|:--:|:--:|:--:|:--:|:--:|')
for i in zip(names, w_lend_star, w_gmv, w_rp, w_ew):
    print('{:1} | {:.2f} | {:.2f} | {:.2f}| {:.2f}'.format(i[0], i[1] * 100, i[2] * 100, i[3] * 100, i[4] * 100))
print('Avg Rtn  | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}%'.
      format(rtns[i_star], rtn_gmv, rtn_rp, rtn_ew))
print('Std Dev  | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}%'.
      format(risks[i_star], risk_gmv, risk_rp, risk_ew))
print('Sharpe | {:.2f} | {:.2f} | {:.2f} | {:.2f}'.format(Sharpes[i_star], (rtn_gmv - avg_rf) / risk_gmv,
                                                          (rtn_rp - avg_rf) / risk_rp, (rtn_ew - avg_rf) / risk_ew))

# In[8]:


# computations for the first five years


E = np.matrix(return_data[names].iloc[0:59].mean())
Sigma = np.matrix(return_data[names].iloc[0:59].cov())
Corr_matrix = np.matrix(return_data[names].iloc[0:59].corr())
risk_matrix = np.matrix(return_data[names].iloc[0:59].std()) * np.sqrt(12) * 100

rf_table = return_data['rf'].iloc[0:59].mean() * 1200

print(' ETF |Avg Rtn | Std Dev| SR|', end='\n')
print('|:--|:--:|:--:|:--:|', end='\n')
print('Avg Risk free |{:.2f}% | |'.format(return_data['rf'].iloc[0:59].mean() * 1200, ), end='\n')

for i in range(len(names)):
    print('{:.6} | {:.2f}% | {:.2f}% |{:.2f}'.format(names[i], E[0, i] * 1200, risk_matrix[0, i],
                                                     ((E[0, i] * 1200 - rf_table) / risk_matrix[0, i])))

print('')
print('')
print('Correlation Matrix')
print('     ', end=' ')
for i in range(len(names)):
    print(' |{:6}'.format(names[i]), end=' ')
print('')

for row in range(len(Corr_matrix)):
    print('{:6}'.format(names[row]), end=' ')
    for col in range(len(Corr_matrix)):
        print('|{:.2f} '.format(Corr_matrix[row, col]), end=' ')
    print()

# Heat map for the correlation matrix

print('')
print('')
print('# Correlation Matix')

corr = return_data[names].iloc[:59].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
fig = plt.figure()
sns.heatmap(corr, mask=mask, linewidths=0.5)
plt.show()

# In[9]:


# Calculate frontier

w_gmv, risk_gmv, rtn_gmv = global_min_portfolio(E, Sigma, E.shape[1])
r_max = np.max(E) * 1
r_min = np.min(E) * 1.01
avg_rf = return_data['rf'].iloc[0:59].mean()
n = E.shape[1]
rtns, risks, Sharpes, ws = mean_variance_fronter(E, avg_rf, Sigma, n, r_min, r_max, 50)
i_star = np.argmax(Sharpes)
risk_sr = risks[i_star]
rtn_sr = rtns[i_star]
w_rp, risk_rp, rtn_rp = risk_parity_portfolio(E, Sigma, E.shape[1])
w_ew, risk_ew, rtn_ew = equally_weighted_portfolio(E, Sigma, E.shape[1])

# In[10]:


# Plot the frontier

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
borrowing_spread = 2
borrowing_rate = avg_rf * 1200 + borrowing_spread

ax1.plot(risks, rtns)
ax1.scatter(risk_gmv, rtn_gmv, c='black', marker='.')
ax1.annotate('GMV', (risk_gmv + 0.30, rtn_gmv - 0.25), fontsize=10)
ax1.scatter(risk_rp, rtn_rp, c='black', marker='.')
ax1.annotate('RP', (risk_rp + 0.30, rtn_rp - 0.25), fontsize=10)
ax1.scatter(risk_ew, rtn_ew, c='black', marker='.')
ax1.annotate('EW', (risk_ew + 0.30, rtn_ew - 0.25), fontsize=10)

for i in names:
    m = return_data[i].iloc[0:59].mean() * 1200
    s = return_data[i].iloc[0:59].std() * np.sqrt(12) * 100
    ax1.scatter(s, m, c='black', marker='.')
    ax1.annotate(i, (s + 0.30, m - 0.25), fontsize=10)
ax1.set_xlabel('Annualized $\sigma_p$%')
ax1.set_title('No borrowing or lending')
i_star = np.argmax(Sharpes)
risk_values = np.linspace(0, risks[i_star], 100)
Sharpes_leverage = (rtns - avg_rf - borrowing_spread) / risks
ib_star = np.argmax(Sharpes_leverage)
return_values = avg_rf + np.max(Sharpes) * risk_values
risk_values_leverage = np.linspace(risks[ib_star], risks[np.argmax(rtns)] * 1.2, 100)
return_values_leverage = avg_rf + borrowing_spread + np.max(Sharpes_leverage) * risk_values_leverage
ax2.plot(risks, rtns)
ax2.plot(risk_values, return_values, c='black')
ax2.scatter(risks[i_star], rtns[i_star], c='black', marker='.')
ax2.annotate('$SR^*_{lending}$', (risks[i_star] + 0.30, rtns[i_star] - 0.25), fontsize=10)
ax2.plot(risk_values_leverage, return_values_leverage, c='black')
ax2.scatter(risks[ib_star], rtns[ib_star], c='black', marker='.')
ax2.annotate('$SR^*_{borrowing}$', (risks[ib_star] + 0.30, rtns[ib_star] - 0.25), fontsize=10)
ax2.scatter(risk_gmv, rtn_gmv, c='black', marker='.')
ax2.annotate('GMV', (risk_gmv + .30, rtn_gmv - 0.25), fontsize=10)
ax2.set_xlabel('Annualized $\sigma_p$%')
ax1.set_ylabel('Annualized $\hat E[r_p]$%')
temp = ax1.set_xlim([np.floor(np.min(risks * .0)), np.max(risks) * 1.75])
ax2.set_title('$r^b_f=$' + str(round(borrowing_rate, 2)) + '%$\geq r^l_f=$' + str(round(avg_rf * 1200, 2)) + '%')
plt.show()

# In[11]:


# In sample performance for the first five years


w_lend_star = ws[i_star]
w_borrow_star = ws[ib_star]

print('Name | $w^*_{lend}$| $w_{GMV}$ |$w_{rp}$ |$w_{ew}$')
print('|:--:|:--:|:--:|:--:|:--:|')
for i in zip(names, w_lend_star, w_gmv, w_rp, w_ew):
    print('{:1} | {:.2f} | {:.2f} | {:.2f}| {:.2f}'.format(i[0], i[1] * 100, i[2] * 100, i[3] * 100, i[4] * 100))
print('Avg Rtn  | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}%'.
      format(rtns[i_star], rtn_gmv, rtn_rp, rtn_ew))
print('Std Dev  | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}%'.
      format(risks[i_star], risk_gmv, risk_rp, risk_ew))
print('Sharpe | {:.2f} | {:.2f} | {:.2f} | {:.2f}'.
      format(Sharpes[i_star], (rtn_gmv - avg_rf) / risk_gmv, (rtn_rp - avg_rf) / risk_rp, (rtn_ew - avg_rf) / risk_ew))

# In[12]:


# Compute rolling means and covariances
rolling_cov = return_data[names].rolling(60).cov().dropna()
rolling_mean = return_data[names].rolling(60).mean().dropna()
dates_cov = rolling_cov.index.get_level_values(0)
dates_mean = rolling_mean.index.get_level_values(0)
nobs = len(rolling_mean)

# In[13]:


# Compute the out-of-sample performance, fixed_weights
return_data['gmv_fixed'] = np.NaN
return_data['sr_fixed'] = np.NaN
return_data['sr_bonds_fixed'] = np.NaN
return_data['rp_fixed'] = np.NaN
return_data['rp_bonds_fixed'] = np.NaN
return_data['ew_fixed'] = np.NaN
return_data['ew_bonds_fixed'] = np.NaN

for i in range(1, nobs - 1):
    return_data['gmv_fixed'].loc[dates_mean[i + 1]] = np.dot(w_gmv.T,
                                                             np.matrix(return_data[names].loc[dates_mean[i + 1]]).T)
    return_data['rp_fixed'].loc[dates_mean[i + 1]] = np.dot(w_rp.T,
                                                            np.matrix(return_data[names].loc[dates_mean[i + 1]]).T)
    k_rp = risk_gmv / risk_rp
    return_data['rp_bonds_fixed'] = (1 - k_rp) * return_data['rf'] + k_rp * return_data['rp_fixed']
    return_data['ew_fixed'].loc[dates_mean[i + 1]] = np.dot(w_ew.T,
                                                            np.matrix(return_data[names].loc[dates_mean[i + 1]]).T)
    k_ew = risk_gmv / risk_ew
    return_data['ew_bonds_fixed'] = (1 - k_ew) * return_data['rf'] + k_rp * return_data['ew_fixed']
    return_data['sr_fixed'].loc[dates_mean[i + 1]] = np.dot(ws[i_star].T,
                                                            np.matrix(return_data[names].loc[dates_mean[i + 1]]).T)
    k_sr = risk_gmv / risk_sr
    return_data['sr_bonds_fixed'] = (1 - k_sr) * return_data['rf'] + k_sr * return_data['sr_fixed']

avg_rf = return_data['rf'].loc[dates_mean[1]:].mean() * 1200
strategy_names = ['gmv_fixed', 'sr_fixed', 'rp_fixed', 'ew_fixed']
print('Out of sample performance-fixed weights')
print('Strategy | Avg | Std | SR |$VAR_{{.05}}$| Final', end='\n')
print('|--|--|--|--|--|--|', end='\n')
for i in strategy_names:
    m = return_data[i].dropna().mean() * 1200
    var_05 = return_data[i].dropna().quantile(0.05) * 1200
    s = return_data[i].dropna().std() * np.sqrt(12) * 100
    sr = (m - avg_rf) / s
    final_wealth = (1 + return_data[i].dropna()).cumprod().iloc[-1]
    print(i, '| {:.2f}% |{:.2f}% |{:.2f} | {:.2f}% | ${:.2f}'.format(m, s, sr, var_05, final_wealth))

for i in strategy_names:
    sns.distplot(return_data[i].dropna() * 1200, hist=False, label=i)
plt.title('Out of Sample Performance -fixed weights')

# In[14]:


# Plot wealths

temp = (1 + return_data[strategy_names].dropna()).cumprod().plot()
temp = plt.title('Out of sample - fixed weights')
temp = plt.xlabel('Return')
temp = plt.show()
# temp = plt.clf()


# In[15]:


#  Out of sample -- Levered to GMV

strategy_names = ['gmv_fixed', 'sr_bonds_fixed', 'rp_bonds_fixed', 'ew_bonds_fixed']
print('Strategy | Avg | Std | SR |$VAR_{{.05}}$| Final', end='\n')
print('|--|--|--|--|--|--|', end='\n')
for i in strategy_names:
    m = return_data[i].dropna().mean() * 1200
    var_05 = return_data[i].dropna().quantile(0.05) * 1200
    med = return_data[i].dropna().median() * 1200
    s = return_data[i].dropna().std() * np.sqrt(12) * 100
    sr = (m - avg_rf) / s
    final_wealth = (1 + return_data[i].dropna()).cumprod().iloc[-1]
    print(i, '| {:.2f}% |{:.2f}% |{:.2f} | {:.2f}% | ${:.2f}'.format(m, s, sr, var_05, final_wealth))

for i in strategy_names:
    sns.distplot(return_data[i].dropna() * 1200, hist=False, label=i)
plt.title('Levered Performance - fixed weights')
plt.xlabel('Return')
plt.show()

# In[16]:


# Plot wealth

(1 + return_data[strategy_names].dropna()).cumprod().plot()
plt.title('Wealth - levered fixed weights')
plt.show()

# In[17]:


n = len(names)
rolling_cov = return_data[names].rolling(60).cov().dropna()
rolling_mean = return_data[names].rolling(60).mean().dropna()
dates_cov = rolling_cov.index.get_level_values(0)
dates_mean = rolling_mean.index.get_level_values(0)
nobs = len(rolling_mean)

return_data['r_gmv'] = np.NaN
return_data['r_sr'] = np.NaN
return_data['r_sr_bonds'] = np.NaN
return_data['r_rp'] = np.NaN
return_data['r_rp_bonds'] = np.NaN
return_data['r_ew'] = np.NaN
return_data['r_ew_bonds'] = np.NaN

return_data['k_sr'] = np.NaN
return_data['k_rp'] = np.NaN
return_data['k_ew'] = np.NaN

for i in range(1, nobs - 1):
    Sigma = np.matrix(rolling_cov.loc[dates_cov[(i - 1) * n:i * n]])
    E = np.matrix(rolling_mean.loc[dates_mean[i]])
    w_gmv, risk_gmv, rtn_gmv = global_min_portfolio(E, Sigma, E.shape[1])
    return_data['r_gmv'].loc[dates_mean[i + 1]] = np.dot(w_gmv.T,
                                                         np.matrix(return_data[names].loc[dates_mean[i + 1]]).T)
    w_rp, risk_rp, rtn_rp = risk_parity_portfolio(E, Sigma, E.shape[1])
    return_data['r_rp'].loc[dates_mean[i + 1]] = np.dot(w_rp.T, np.matrix(return_data[names].loc[dates_mean[i + 1]]).T)
    k_rp = risk_gmv / risk_rp
    return_data['k_rp'].loc[dates_mean[i + 1]] = k_rp
    return_data['r_rp_bonds'] = (1 - k_rp) * return_data['rf'] + k_rp * return_data['r_rp']
    w_ew, risk_ew, rtn_ew = equally_weighted_portfolio(E, Sigma, E.shape[1])
    return_data['r_ew'].loc[dates_mean[i + 1]] = np.dot(w_ew.T, np.matrix(return_data[names].loc[dates_mean[i + 1]]).T)
    k_ew = risk_gmv / risk_ew
    return_data['k_ew'].loc[dates_mean[i + 1]] = k_ew
    return_data['r_ew_bonds'] = (1 - k_ew) * return_data['rf'] + k_rp * return_data['r_ew']
    r_max = np.max(E)
    r_min = np.min(E)
    n = E.shape[1]
    rtns, risks, Sharpes, ws = mean_variance_fronter(E, avg_rf, Sigma, n, r_min, r_max, 200)
    i_star = np.argmax(Sharpes)
    risk_sr = risks[i_star]
    return_data['r_sr'].loc[dates_mean[i + 1]] = np.dot(ws[i_star].T,
                                                        np.matrix(return_data[names].loc[dates_mean[i + 1]]).T)
    k_sr = risk_gmv / risk_sr
    return_data['k_sr'].loc[dates_mean[i + 1]] = k_sr
    return_data['r_sr_bonds'] = (1 - k_sr) * return_data['rf'] + k_sr * return_data['r_sr']

avg_rf = return_data['rf'].loc[dates_mean[1]:].mean() * 1200

strategy_names = ['r_gmv', 'r_sr', 'r_rp', 'r_ew']
print('Strategy | Avg | Std | SR |$VAR_{{.05}}$| Final', end='\n')
print('|--|--|--|--|--|--|', end='\n')
for i in strategy_names:
    m = return_data[i].dropna().mean() * 1200
    var_05 = return_data[i].dropna().quantile(0.05) * 1200
    s = return_data[i].dropna().std() * np.sqrt(12) * 100
    sr = (m - avg_rf) / s
    final_wealth = (1 + return_data[i].dropna()).cumprod().iloc[-1]
    print(i, '| {:.2f}% |{:.2f}% |{:.2f} | {:.2f}% | ${:.2f}'.format(m, s, sr, var_05, final_wealth))

for i in strategy_names:
    sns.distplot(return_data[i].dropna() * 1200, hist=False, label=i)
plt.show()

# In[18]:


# Plot wealth


(1 + return_data[strategy_names].dropna()).cumprod().plot()
temp = plt.title('Wealth - allowing for learning')
temp = plt.xlabel('Time')
temp = plt.ylabel('Wealth in $')
plt.show()

# In[19]:


# Performance including bonds

print('')
strategy_names = ['r_gmv', 'r_sr_bonds', 'r_rp_bonds', 'r_ew_bonds']
print('Strategy | Avg | Std | SR |$VAR_{{.05}}$| Final', end='\n')
print('|--|--|--|--|--|--|', end='\n')
for i in strategy_names:
    m = return_data[i].dropna().mean() * 1200
    var_05 = return_data[i].dropna().quantile(0.05) * 1200
    med = return_data[i].dropna().median() * 1200
    s = return_data[i].dropna().std() * np.sqrt(12) * 100
    sr = (m - avg_rf) / s
    final_wealth = (1 + return_data[i].dropna()).cumprod().iloc[-1]

    print(i, '| {:.2f}% |{:.2f}% |{:.2f} | {:.2f}% | ${:.2f}'.format(m, s, sr, var_05, final_wealth))

plt.clf()
for i in strategy_names:
    sns.distplot(return_data[i].dropna() * 1200, hist=False, label=i)
plt.savefig('dist_learning_bonds.pdf', type='pdf')
plt.show()

# In[20]:


# Wealth with bonds
plt.clf()
(1 + return_data[strategy_names].dropna()).cumprod().plot()
plt.savefig('wealth_learning_bonds.pdf', type='pdf')
plt.show()

# In[21]:


plt.clf()
(1 + return_data[strategy_names].dropna()).cumprod().plot()
plt.savefig('wealth_learning_bonds.pdf', type='pdf')
plt.show()

# In[ ]:
"""