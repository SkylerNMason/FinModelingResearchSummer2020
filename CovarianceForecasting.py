from DataManipulation import *

from Plotting import *
from random import gauss
from random import seed
import numpy as np
from arch import arch_model

def createSD(dfDict):
    # Takes a dateframe dict with daily returns and returns a
    # dataframe with annual standard deviations

    sdDF = pd.DataFrame()
    for asset in dfDict:
        sdDF[asset] = dfDict[asset]["Returns"]

    return sdDF.std()*np.sqrt(252)

def historicalCov(dfDict, annualize = 1):
    return

def createCovWER():
    return

def garchModel(dfDict):

    '''
    #For testing with gauss data:
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
    testSize = test_size
    '''
    garchPreds = pd.DataFrame()

    for asset in dfDict:
        df = dfDict[asset]
        returns = df["Returns"].values
        testSize = int(len(returns)*0.20) # 20% test size
        train, test = returns[:-testSize], returns[-testSize:]
        p, q = 1, 1

        plotPACF(np.array(returns)**2)
        '''
        plotData(returns[-testSize:], vols[-testSize:],
                 "True Returns & True Vol for Test")
        model = arch_model(train,p = p, q = q)
        fitModel = model.fit(disp="off")
        print(fitModel.summary())
        pred = fitModel.forecast(horizon=testSize)
        '''

        rollingPredictions = []
        for i in range(testSize):
            train = returns[: -(testSize-i)]
            model = arch_model(train,p = p, q = q)
            fitModel = model.fit(disp="off")
            if i is 0:
                print(asset + " model summary:")
                print(fitModel.summary())
            pred = fitModel.forecast(horizon=1)
            rollingPredictions.append(np.sqrt(pred.variance.values[-1,:][0]))


        plotData(rollingPredictions, test, "Rolling Pred Vol & True Returns " + asset)
        garchPreds[asset] = rollingPredictions

    return garchPreds


dfDict = generateDfDict()
garchModel(dfDict)