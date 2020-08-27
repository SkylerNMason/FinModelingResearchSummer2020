from DataGeneration import *

from Plotting import *
from random import gauss
from random import seed
import numpy as np
from arch import arch_model
from GlobalVariables import *
import cvxopt as opt


# Idea: last 3 years or something rolling historical cov

def createSD(dfDict):
    # Takes a dateframe dict with daily returns and returns a
    # dataframe with annual standard deviations

    sdDF = pd.DataFrame()
    for asset in dfDict:
        sdDF[asset] = dfDict[asset]["Returns"]

    return sdDF.std()*np.sqrt(252)

def historicalCov(dfDict, unknownLen):
    # Returns the historical covariance matrix based on yTrain
    # Can use the unkownLen for a rolling window
    # yTrains must be the same size
    returnVec = []
    for asset in dfDict:
        xTrain, xTest, yTrain, yTest = dfDict[asset]
        returns = yTrain.append(yTest, ignore_index=True)
        returns = list(returns["Returns"].tolist())
        returnVec.append(returns[:-unknownLen])
    S = np.cov(returnVec)
    return S


def stdDevsToCov(numAssets, c, period, stdDevs):
    # Uses standard deviations and correlations to create a covariance matrix
    n = numAssets
    S = np.zeros(shape=(n, n))
    # Fills S left to right, top to bottom:
    for row in range(n):
        x = stdDevs.iloc[period, row]
        for col in range(n):
            y = stdDevs.iloc[period, col]
            corr = c.iloc[row, col]
            S[row][col] = corr*x*y
    return S


def historicalCor(dfDict, unknownLen):
    # Returns the historical correlation matrix based on yTrain
    # Can use the unkownLen for a rolling window
    returnVec = []
    for asset in dfDict:
        xTrain, xTest, yTrain, yTest = dfDict[asset]
        returns = yTrain.append(yTest, ignore_index=True)
        returns = list(returns["Returns"].tolist())
        returnVec.append(returns[:-unknownLen])
    c =pd.DataFrame(returnVec).T
    c.columns = (list(dfDict))
    c = c.corr()
    return c


def garchModel(dfDict, **excess):


    '''#For testing with gauss data:
    n = 1000
    omega = 0.5
    alpha_1 = 0.1
    alpha_2 = 0.2
    beta_1 = 0.3
    beta_2 = 0.4
    test_size = int(n*0.1)
    series = [gauss(0,1), gauss(0,1)]
    vols = [1,1]
    for _ in range(n):
        new_vol = np.sqrt(omega + alpha_1 * series[-1] ** 2 + alpha_2 * series[-2] ** 2 + beta_1 * vols[-1] ** 2 + beta_2 * vols[-2] ** 2)
        new_val = gauss(0, 1) * new_vol

        vols.append(new_vol)
        series.append(new_val)
    returns = series
    testLen = test_size'''

    garchPreds = pd.DataFrame()

    for asset in dfDict:
        xTrain, xTest, yTrain, yTest = dfDict[asset]
        returns = yTrain.append(yTest)
        returns = returns*100  # Rescaling for garch model convergence problems
        testLen = int(len(yTest))

        if graph:
            plotPACF(np.array(returns)**2)
        #p = int(input("Input p integer:"))
        p = 1
        q = p
        #q = int(input("Input q integer:"))

        vols = []
        rollingPredictions = []
        # Generate rolling predictions of the next periods std dev.
        # for each time period:
        for i in range(testLen):
            knownReturns = returns[: -(testLen-i)]
            mean = knownReturns.mean()
            train = knownReturns
            model = arch_model(train, p=p, q=q)
            fitModel = model.fit(disp="off")
            if debug is True and i is 0:
                print(asset + " model summary for first garch pred:")
                print(fitModel.summary())
            pred = fitModel.forecast(horizon=1) # Forecasts variance
            vols.append(np.sqrt((yTest.values[i][0]-mean)**2))  # Maybe wrong?
            rollingPredictions.append(np.sqrt(pred.variance.values[-1, :]
                                              [0])/100)

        if graph:
            plotData(rollingPredictions, vols, "Rolling Pred Vol & True Vol?" + asset)
        garchPreds[asset] = rollingPredictions

    return garchPreds
