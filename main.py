"""
Financial Modeling Research: Risky Portfolio Optimization
Summer 2020
Overseen by Professor Burton Hollifield and Professor Bryan Routledge
Programmed By Skyler Mason

Installed plugins:
pandas
pandas_datareader (?)
matplotlib (?)
xlrd
cvxopt
sklearn
statsmodels
arch
featuretools
openpyxl
"""

from VolatilityForecasting import *
from PortfolioCreation import *
from ReturnForecasting import *
from GlobalVariables import *

def updateDict(**kwargs):
    fileLocation = kwargs["fileLocation"]
    timeFormat = kwargs["timeFormat"]
    testSize = kwargs["testSize"]
    normalize = kwargs["normalize"]
    randomState = kwargs["randomState"]
    normFunc = kwargs["normFunc"]
    primitives = kwargs["primitives"]
    new = {"dfDict": generateDfDict(fileLocation, timeFormat, testSize, normalize,
                                    randomState, normFunc, primitives)}
    kwargs.update(new)
    return kwargs

def defaultModelKwargs(randomState):
    # fileLocation = input("File or folder path: ")
    # timeFormat = input("Date format of your dates ("%Y%m%d", etc): ")= "E:\ProgrammingProjects\FinModelingResearchSummer2020\Data"
    fileLocation = "D:\\Users\\dzjre\PycharmProjects\FinModelingResearchSummer2020\Data"
    timeFormat = None # Defaultly uses inferred time,
    # Make something like %Y%m to not use this feature
    testSize = .2
    normalize = True
    normFunc = StandardScaler
    primitives = None
    dfDict = generateDfDict(fileLocation, timeFormat, testSize, normalize,
                            randomState, normFunc, primitives)

    kwargs = {"dfDict": dfDict, "fileLocation": fileLocation,
              "timeFormat": timeFormat, "testSize": .2, "alpha": -1,
              "minRsqrDifScorer": False, "randomState": None,
              "alphaRange": stdTestingRange, "normalize": True,
              "normFunc": StandardScaler, "primitives": primitives,
              "lambdaRatio": np.logspace(-1,0,25)}
    return kwargs

def callFuncs(**kwargs):
    # Calls the various above models for testing with
    # **kwargs as the inputs
    dfDict = kwargs["dfDict"]
    testSize = kwargs["testSize"]
    #testModels(testSize, mulLinReg(**kwargs), dfDict)
    #testModels(testSize, ridgeReg(**kwargs), dfDict)
    testModels(testSize, lassoReg(**kwargs), dfDict)
    #testModels(testSize, elasticNet(**kwargs), dfDict)
    return

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def main(randomState=None):
    # TODO:
    #   Process:
    #       Forecast mean, variance, cov to build risky portfolio
    # TODO:
    #   Output:
    #       Plug into utility function and output

    # Default model kwargs:
    kwargs = defaultModelKwargs(randomState)

    # Different testing kwargs:
    #kwargs.update({'randomState': randomState, 'primitives': 'fed',
    #               "normalize": True})
    #kwargs = updateDict(**kwargs)


    print("\n\n\n\n\n\nStage 1 Done\n\n\n\n\n\n")
    realReturns = []

    # Baseline performance with one/n portfolio:
    for asset in kwargs["dfDict"]:
        # Based on yTest data:
        realReturns.append(kwargs["dfDict"][asset][3]["Returns"].tolist())
    baseRtn = oneN(realReturns)[2]

    returnVec = []
    #stdDevPreds = garchModel(**kwargs)
    lassoModels = lassoReg(**kwargs)
    S = historicalCov(kwargs["dfDict"])

    # Pool together return forecasts:
    for model in lassoModels:
        preds = list(lassoModels[model][1])
        returnVec.append(preds)

    weights = []
    # Generate weights for each time period:
    for i in range(len(preds)):
        values = []
        for j in range(len(returnVec)):
            values.append([returnVec[j][i]])
        weights.append(globalMinVarPortfolio(np.asarray(values), S)[0])


    realized = []
    # Realize returns with portfolio weights:
    for i in range(len(weights)):
        temp = []
        for j in range(len(realReturns)):
            temp.append(realReturns[j][i])
        temp = opt.matrix(temp)
        realized.append(blas.dot(weights[i].T, temp))
    print(np.mean(realized))
    print(baseRtn)

    #print(returnVec)
    # Needs a dataframe with return vectors for each asset,
    # and optionally a covariance matrix
    # print(globalMinVarPortfolio(np.asarray(returnVec), S))

    #returnData = returnData.T.to_numpy()
    #print(globalMinVarPortfolio(returnData))

    print("Done")

    return 0

main()