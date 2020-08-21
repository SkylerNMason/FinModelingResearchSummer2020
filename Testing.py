from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ReturnForecasting import *
from PortfolioCreation import *
from VolatilityForecasting import *
from GlobalVariables import *
from fredapi import Fred
fred = Fred(api_key='93a721eba3644b125df12f223484e552')
import dateutil.relativedelta


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
    if kwargs["rf"] is None:  # Generate the annual risk free rate
        # Get test dates range:
        dfDict = kwargs["dfDict"]
        testDates = (dfDict[list(dfDict.keys())[0]][1]).index
        start = list(testDates)[0]
        end = list(testDates)[-1]

        # Adjust dates so that tb3m for the end month is included:
        end = end + dateutil.relativedelta.relativedelta(months=1)
        tb3m = fred.get_series("TB3MS", observation_start=start,
                               observation_end=end)
        kwargs["rf"] = np.mean(tb3m)/100
    kwargs.update({"realizedReturns": realizedReturns, "stdDevs": stdDevs})
    return kwargs


def defaultModelKwargs():
    # fileLocation = input("File or folder path: ")
    # timeFormat = input("Date format of your dates ("%Y%m%d", etc): ")= "E:\ProgrammingProjects\FinModelingResearchSummer2020\Data"
    timeFormat = None  # Defaultly uses inferred time,
    # Make something like %Y%m to not use this feature

    kwargs = {"fileLocation": fileLocation,"periodsPerAnnum": 1,
              "timeFormat": timeFormat, "testSize": .2, "alpha": -1,
              "minRsqrDifScorer": False, "randomState": None,
              "alphaRange": stdTestingRange, "normalize": True,
              "normFunc": StandardScaler, "primitives": None,
              "lambdaRatio": np.logspace(-1, 0, 25), "rf": None,
              "rtnModels": loadModels(fileLocation)}
    return kwargs


def basePerfGen(**kwargs):
    # Generates a base performance portfolio: the 1/n portfolio
    wgt, stdDev, rtn = oneN(kwargs["realizedReturns"])
    return stdDev, rtn

def generateRtnModels(**kwargs):
    # Generates a dictionary of the return forecasting models for testing
    # TODO: Add more models to test with
    models = dict()
    models["lassoModels"] = lassoReg(**kwargs)
    return models

def generateCovariance(**kwargs):
    # Generates a covariance matrix for each testing time period
    testingLen = len(kwargs["dfDict"][list(kwargs["dfDict"].keys())[0]][3])
    sModels = dict()
    n = len(kwargs["dfDict"])

    sMatrices = []
    # Historical covariance:
    sMatrices.extend([historicalCov(kwargs["dfDict"]) for i in
                                    range(testingLen)])
    sModels["hist"] = sMatrices

    sMatrices = []
    # Garch predicted covariance based on historical correlation:
    ### Assumption: correlation remains constant over time
    c = historicalCor(kwargs["dfDict"])
    stdDevs = garchModel(**kwargs)
    for period in range(testingLen):
        S = np.zeros(shape=(n, n))
        # Fills S left to right, top to bottom:
        for row in range(n):
            x = stdDevs.iloc[period, row]
            for col in range(n):
                y = stdDevs.iloc[period, col]
                corr = c.iloc[row, col]
                S[row][col] = corr*x*y
        sMatrices.append(opt.matrix(S))
    sModels["garch"] = sMatrices

    return sModels


def generatePortfolios():
    # Defines the portfolio we will be testing over
    portModels = dict()
    portModels["gmv"] = globalMinVarPortfolio
    portModels["sharpe"] = sharpePortfolio
    return portModels


def testModels(**kwargs):
    results = dict()
    if kwargs["rtnModels"] is None:
        rtnModels = generateRtnModels(**kwargs)
    else:  # We are using saved models
        rtnModels = kwargs["rtnModels"]
    print("Returns Forecasted\n")
    sModels = generateCovariance(**kwargs)
    print("Covariance Forecasted\n\n\n")
    portModels = generatePortfolios()

    # Iterate through and test each model type one by one
    for modelType in rtnModels:
        modelList = rtnModels[modelType]
        # modelList is a list of models that shares the same settings
        # are for unique assets
        n = len(modelList)  # Number of assets

        returnVec = []  # Period based forecasts (ie daily forecasts)
        # Pool together return forecasts for a given model:
        for model in modelList:
            preds = list(modelList[model][1])
            returnVec.append(preds)

        testingLen = len(preds)
        # Iterate through and test each covariance forecast one by one
        for sType in sModels:
            sMatrices = sModels[sType]

            # Iterate through and test each portfolio type one by one
            for portType in portModels:
                portFunc = portModels[portType]

                weights = []  # Period based weights (ie daily weights)
                # Generate weights for each time period:
                for i in range(testingLen):
                    predValues = []
                    S = sMatrices[i]
                    for j in range(n):
                        predValues.append([returnVec[j][i]])
                    w = portFunc(np.asarray(predValues), S)[0]
                    weights.append(w)

                realized = []
                # Realize returns with portfolio weights for each time period:
                for i in range(testingLen):
                    temp = []
                    for j in range(n):
                        temp.append(kwargs["realizedReturns"][j][i])
                    temp = opt.matrix(temp)
                    realized.append(blas.dot(weights[i].T, temp))

                # Standard deviation of portfolio returns:
                stdDev = np.std(realized, ddof=1)

                if debug:
                    for i in range(testingLen):
                        print("New period below:")
                        for a in range(n):
                            print(kwargs["realizedReturns"][a][i])
                        print("Portfolio period return:", realized[i])

                # Save results under the name of the modelType as a tuple
                # of the standard deviation and realized returns:
                name = str(modelType).capitalize() + str(sType).capitalize()\
                       + str(portType).capitalize()
                results.update({name: (stdDev, realized)})
                print("\n{} Evaluated".format(str(name)))


    # Print results:
    print("\n\nAnnualized Results:\n")
    annualize = kwargs["periodsPerAnnum"]
    rf = kwargs["rf"]
    print("StDev  Rtn  Shrpe".rjust(outputRJust))
    for test in results:
        stdDev, realized = results[test]
        stdDev, avgRtn = stdDev*annualize**.5, np.mean(realized)*annualize
        sharpe = (avgRtn - rf) / (stdDev)
        result = "{} {:.3f} {:.3f} {:.3f}".format(test, round(stdDev, 3),
                                                  round(avgRtn, 3),
                                                  round(sharpe, 3))
        print(result.rjust(outputRJust))

    return results


def testGMV():
    return