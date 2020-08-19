from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ReturnForecasting import *
from PortfolioCreation import *
from VolatilityForecasting import *
from GlobalVariables import *

def updateDict(**kwargs):
    # Updates/creates the dfDict for a given kwargs
    new = {"dfDict": generateDfDict(**kwargs)}
    kwargs.update(new)

    realizedReturns = []  # Returns that occurred in actual reality
    stdDevs = [] # Standard deviation of realized returns
    for asset in kwargs["dfDict"]:
        # Based on yTest data for each asset:
        returns = kwargs["dfDict"][asset][3]["Returns"].tolist()
        realizedReturns.append(returns)
        stdDevs.append(np.std(returns, ddof=1))
    kwargs.update({"realizedReturns": realizedReturns, "stdDevs": stdDevs})
    return kwargs


def defaultModelKwargs(randomState):
    # fileLocation = input("File or folder path: ")
    # timeFormat = input("Date format of your dates ("%Y%m%d", etc): ")= "E:\ProgrammingProjects\FinModelingResearchSummer2020\Data"
    fileLocation = "D:\\Users\\dzjre\PycharmProjects\FinModelingResearchSummer2020\Data"
    timeFormat = None  # Defaultly uses inferred time,
    # Make something like %Y%m to not use this feature


    kwargs = {"fileLocation": fileLocation,
              "timeFormat": timeFormat, "testSize": .2, "alpha": -1,
              "minRsqrDifScorer": False, "randomState": None,
              "alphaRange": stdTestingRange, "normalize": True,
              "normFunc": StandardScaler, "primitives": None,
              "lambdaRatio": np.logspace(-1, 0, 25)}
    return kwargs


def basePerfGen(**kwargs):
    # Generates a base performance portfolio: the 1/n portfolio
    wgt, risk, rtn = oneN(kwargs["realizedReturns"])
    stdDevs = opt.matrix(kwargs["stdDevs"])
    stdDev = blas.dot(stdDevs, wgt)  # Actual standard deviation of portfolio
    return stdDev, rtn

def generateRtnModels(**kwargs):
    # Generates a dictionary of the return forecasting models for testing
    # TODO: Add more models to test with
    models = dict()
    models["lassoModels"] = lassoReg(**kwargs)
    return models

def testModels(**kwargs):
    results = dict()
    rtnModels = generateRtnModels(**kwargs)

    # Cycle through and test each model type one by one
    for modelType in rtnModels:
        modelList = rtnModels[modelType]
        # modelList is a list of each model that share the same settings
        # but have unique assets

        S = historicalCov(kwargs["dfDict"])

        returnVec = []
        # Pool together return forecasts for a given model:
        for model in modelList:
            preds = list(modelList[model][1])
            returnVec.append(preds)

        weights = []
        # Generate weights for each time period:
        for i in range(len(preds)):
            predValues = []
            for j in range(len(returnVec)):
                predValues.append([returnVec[j][i]])

            # TODO: Change so its not just gmv portfolio
            test = globalMinVarPortfolio(np.asarray(predValues), S)[0]
            weights.append(test)

        realized = []
        # Realize returns with portfolio weights:
        for i in range(len(weights)):
            temp = []
            for j in range(len(kwargs["realizedReturns"])):
                temp.append(kwargs["realizedReturns"][j][i])
            temp = opt.matrix(temp)
            realized.append(blas.dot(weights[i].T, temp))

        results.update({str(modelType): np.mean(realized)})
        print(np.mean(realized))
    return results


def testGMV():
    return