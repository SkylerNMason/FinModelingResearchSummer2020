from DataGeneration import *

from Plotting import *
from random import gauss
from random import seed
import numpy as np
from arch import arch_model
from GlobalVariables import *
import cvxopt as opt

def createSD(dfDict):
    # Takes a dateframe dict with daily returns and returns a
    # dataframe with annual standard deviations

    sdDF = pd.DataFrame()
    for asset in dfDict:
        sdDF[asset] = dfDict[asset]["Returns"]

    return sdDF.std()*np.sqrt(252)

def historicalCov(dfDict):
    # Returns the historical covariance matrix based on yTrain
    # yTrains must be the same size
    returnVec = []
    for asset in dfDict:
        xTrain, xTest, yTrain, yTest = dfDict[asset]
        returnVec.append(list(yTrain["Returns"].tolist()))
    S = opt.matrix(np.cov(returnVec))
    return S

def createCovWER():
    return

def garchModel(dfDict, testSize, **excess):


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
        testLen = int(len(returns)*testSize)

        if graph: plotPACF(np.array(returns)**2)
        p = int(input("Input p integer:"))
        q = int(input("Input q integer:"))

        mean = returns.mean()
        vols = []
        rollingPredictions = []
        for i in range(testLen):
            train = returns[: -(testLen-i)]
            model = arch_model(train, p=p, q=q)
            fitModel = model.fit(disp="off")
            if debug is True and i is 0:
                print(asset + " model summary for first garch pred:")
                print(fitModel.summary())
            pred = fitModel.forecast(horizon=1)
            vols.append(np.sqrt((yTest.values[i][0]-mean)**2)) # Maybe wrong?
            rollingPredictions.append(np.sqrt(pred.variance.values[-1,:][0]))
        #plotData(rollingPredictions, vols[-testLen:])

        if graph: plotData(rollingPredictions, vols, "Rolling Pred Vol & True Vol?" + asset)
        garchPreds[asset] = rollingPredictions

    return garchPreds