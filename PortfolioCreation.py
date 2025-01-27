
import cvxopt as opt
from cvxopt import blas, solvers
import matplotlib.pyplot as plt
import scipy.optimize as sco
from GlobalVariables import *

# Ideas: Create a portfolio that equally invests in all assets that
# an svm algo says will have a positive return


def generateWeights(portFunc, sMatrices, returnVec, n, **kwargs):
    # Generates the asset weights for a given portfolio function over all
    # of the testing periods
    weights = []
    testingLen = kwargs["testingLen"]

    if str(portFunc.__name__).startswith("oneNMixed"):
        return portFunc(sMatrices, returnVec, n, **kwargs)

    for i in range(testingLen):
        predValues = []  # Predicted weights
        S = sMatrices[i]
        for j in range(n):
            predValues.append([returnVec[j][i]])
        w = portFunc(np.asarray(predValues), S, **kwargs)[0]
        weights.append(w)
    return weights


def oneNMixedGMV(sMatrices, returnVec, n, **kwargs):
    weights = generateWeights(globalMinVarPortfolio, sMatrices,
                              returnVec, n, **kwargs)
    return oneNMixedPortfolio(returnVec, n, weights, **kwargs)


def oneNMixedSharpe(sMatrices, returnVec, n, **kwargs):
    weights = generateWeights(cvxoptSharpe, sMatrices,
                              returnVec, n, **kwargs)
    return oneNMixedPortfolio(returnVec, n, weights, **kwargs)


def oneNMixedPortfolio(returnVec, n, weights, **kwargs):
    # Takes a set of weights and trys
    alphas = np.array(range(101))/100
    testingLen = kwargs["testingLen"]
    #alphas = [.7]
    X = []
    Y = []
    bestWeights = None
    bestSharpe = -999

    for alpha in alphas:
        tempWgts = []
        for weight in weights:
            temp = []
            for i in range(len(weight)):
                temp.append((1-alpha)*(1/n) + alpha*weight[i])
            temp = opt.matrix(temp)
            tempWgts.append(temp)

        predicted = []  # predicted returns
        for i in range(testingLen):
            temp = []
            for j in range(n):
                temp.append(returnVec[j][i])
            temp = opt.matrix(temp)
            predicted.append(blas.dot(tempWgts[i].T, temp))

        stdDev = np.std(predicted, ddof=1)
        annualize = kwargs["periodsPerAnnum"]
        stdDev, avgRtn = stdDev*annualize**.5, np.mean(predicted)*annualize
        sharpe = (avgRtn - kwargs["rf"]) / stdDev
        X.append(alpha)
        Y.append(sharpe)
        if sharpe > bestSharpe:
            bestSharpe = sharpe
            bestWeights = tempWgts
    #print(X)
    #print(Y)
    #plt.plot(X, Y)
    #plt.show()
    return bestWeights


def oneN(returnVec, S=None, **kwargs):
    # Creates the naive 1/N portfolio where each asset
    # has an equal weighting in the portfolio
    returnVec = np.asmatrix(returnVec)
    if S is None:
        S = opt.matrix(np.cov(returnVec))
    n = len(returnVec)
    wgt = opt.matrix(1/n, (n, 1))
    risk = np.sqrt(blas.dot(wgt, S * wgt))

    returnVec = opt.matrix(np.mean(returnVec, axis=1))
    rtn = blas.dot(wgt.T, returnVec)
    return wgt, risk, rtn

# Below function is adapted from
# https://www.quantopian.com/posts/the-efficient-frontier-markowitz-portfolio-optimization-in-python


def cvxoptSharpe(returnVec, S=None, **kwargs):
    original = returnVec
    # Creates a portfolio with the highest sharpe ratio
    # in an efficient frontier (might be wrong?)
    solvers.options['show_progress'] = False
    n = len(returnVec)  # Gets how many assets we are dealing with
    returnVec = np.asmatrix(returnVec)
    N = 100
    mus = [10**(5.0 * t / N - 1.0) for t in range(N)]
    # Convert to cvxopt matrices:
    if S is None:
        S = np.cov(returnVec)  # Covariance matrix
    S = opt.matrix(S)
    pbar = opt.matrix(np.mean(returnVec, axis=1))  # Average returns for each asset
    # Create constraint matrices:
    G = -opt.matrix(np.eye(n))   # Negative nxn identity matrix
    h = opt.matrix(0.0, (n, 1))  # nx1 vector with all zero entries
    A = opt.matrix(1.0, (1, n))  # 1xn vector with all one entries
    b = opt.matrix(1.0)          # 1x1 with 1.0 as its entry
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
    sol = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)
    wgt = sol['x']
    '''print("break")
    print(calc_portfolio_perf(np.asarray(wgt), original, S, kwargs["rf"],
                              kwargs["periodsPerAnnum"])[2])
    print(wgt)
    print("break")'''
    return wgt, riskVec, returnVec


def globalMinVarPortfolio(returnVec, S=None, **kwargs):
    # Creates a portfolio with the minimum possible
    # variance (not highest sharpe necessarily)
    solvers.options['show_progress'] = False
    n = len(returnVec)  # Gets how many assets we are dealing with
    returnVec = np.asmatrix(returnVec)
    # Convert to cvxopt matrices:
    if S is None:
        S = np.cov(returnVec)  # Covariance matrix
    S = opt.matrix(S)
    # Create constraint matrices:
    G = -opt.matrix(np.eye(n))  # Negative nxn identity matrix
    h = opt.matrix(0.0, (n, 1))  # nx1 vector with all zero entries
    hp = opt.matrix(np.zeros((n, 1)))
    A = opt.matrix(1.0, (1, n))  # 1xn vector with all one entries
    b = opt.matrix(1.0)  # 1x1 with 1.0 as its entry

    sol = solvers.qp(S, h, G, hp, A, b)
    wgt = sol['x']
    risk = np.sqrt(blas.dot(wgt, S * wgt))

    returnVec = opt.matrix(np.mean(returnVec, axis=1))
    rtn = blas.dot(wgt.T, returnVec)
    return wgt, risk, rtn


# Below function are adapted from:
# https://www.pythonforfinance.net/2019/07/02/investment-portfolio-optimisation-with-python-revisited/


def calc_portfolio_perf(weights, mean_returns, cov, rf, annualize):
    portfolio_return = np.dot(weights.T, mean_returns) * annualize
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(
        annualize)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio


def simulate_random_portfolios(num_portfolios, mean_returns, cov, rf,
                               annualize,  assetNames):
    results_matrix = np.zeros((len(mean_returns) + 3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, sharpe_ratio = calc_portfolio_perf(
            weights, mean_returns, cov, rf, annualize)
        results_matrix[0, i] = portfolio_return
        results_matrix[1, i] = portfolio_std
        results_matrix[2, i] = sharpe_ratio
        # iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results_matrix[j + 3, i] = weights[j]

    results_df = pd.DataFrame(results_matrix.T,
                              columns=['ret', 'stdev', 'sharpe'] + [ticker for
                                                                    ticker in
                                                                    assetNames])

    return results_df


def bruteSharpe(returnVec, S, **kwargs):
    # Wrapper for brute sharpe and GMV
    return bruteSharpeAndGMV(returnVec, S, **kwargs)[0]


def bruteGMV(returnVec, S, **kwargs):
    # Wrapper for brute sharpe and GMV
    return bruteSharpeAndGMV(returnVec, S, **kwargs)[1]


def bruteSharpeAndGMV(returnVec, S, rf, assetNames, **kwargs):
    # Generates a bunch of portfolios that represent 'all' possible portfolios
    # and picks the best of the bunch
    num_portfolios = len(returnVec)*1000
    annualize = kwargs['periodsPerAnnum']
    results_frame = simulate_random_portfolios(num_portfolios, returnVec, S, rf,
                                               annualize, assetNames)
    # locate position of portfolio with highest Sharpe Ratio
    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
    risk, rtn, wgt = max_sharpe_port[1], max_sharpe_port[0], max_sharpe_port[3:]
    wgt = opt.matrix(wgt)
    sharpe = (wgt, risk, rtn)
    # locate positon of portfolio with minimum standard deviation
    min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
    risk, rtn, wgt = min_vol_port[1], min_vol_port[0], min_vol_port[3:]
    wgt = opt.matrix(wgt)
    gmv = (wgt, risk, rtn)
    return sharpe, gmv


def calc_neg_sharpe(weights, mean_returns, cov, rf, annualize):
    portfolio_return = np.sum(mean_returns * weights) * annualize
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(
        annualize)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return -sharpe_ratio


def weakMaxSharpe(mean_returns, cov, rf, **kwargs):
    # Uses scipy minimize function to find maximized sharpe portfolio
    num_assets = len(mean_returns)
    args = (mean_returns, cov, rf, kwargs['periodsPerAnnum'])
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(calc_neg_sharpe, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    wgt = opt.matrix(result['x'])
    return wgt, 0

# Reference end


# Check out
# https://investresolve.com/blog/portfolio-optimization-simple-optimal-methods/
# for ideas on types of portfolios to test
