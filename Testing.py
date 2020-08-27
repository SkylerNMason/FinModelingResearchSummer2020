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
    fileNameaddition = "MinRsqrDifScorer" + str(kwargs["minRsqrDifScorer"])
    models["lassoModels"+fileNameaddition] = lassoReg(**kwargs)
    #models["ridgeModels" + fileNameaddition] = ridgeReg(**kwargs)
    #models["elasticModels" + fileNameaddition] = elasticNet(**kwargs)
    return models

def generateCovariance(**kwargs):
    # Generates a covariance matrix for each testing time period
    testingLen = len(kwargs["dfDict"][list(kwargs["dfDict"].keys())[0]][3])  #Num of testing periods
    sModels = dict()
    n = len(kwargs["dfDict"])

    sMatrices = []
    # Rolling historical covariance:
    ### Assumption: covariance remains constant over time
    for i in range(testingLen):
        sMatrices.append(historicalCov(kwargs['dfDict'], testingLen-i))
    #sModels["hist"] = sMatrices

    sMatrices = []
    # Garch predicted covariance based on a rolling historical correlation:
    ### Assumption: correlation remains constant over time
    stdDevs = garchModel(**kwargs)
    for i in range(testingLen):
        # Rolling historical correlation:
        c = historicalCor(kwargs['dfDict'], testingLen-i)
        sMatrices.append(stdDevsToCov(n, c, i, stdDevs))
    sModels["garch"] = sMatrices

    return sModels


def generatePortfolios():
    # Defines the portfolio we will be testing over
    portModels = dict()
    #portModels["gmv"] = globalMinVarPortfolio
    portModels["sharpe"] = sharpePortfolio
    #portModels["oneNMixedGMV"] = oneNMixedGMV
    #portModels["oneNMixedSharpe"] = oneNMixedSharpe
    portModels["sharpe?"] = maxSomething
    #portModels["weakSharpe"] = weakMaxSharpe
    return portModels


def printRtnModels(rtnModels, **kwargs):
    for modelType in rtnModels:
        print("\n\n")
        print("Testing:", modelType, "\n")
        modelList = rtnModels[modelType]
        for model in modelList:
            xTrain, xTest, yTrain, yTest = kwargs["dfDict"][model]
            print(model + ":")
            yPred = modelList[model][1]
            printRtnResults(modelList[model][0], xTrain, xTest,
                            yTrain, yTest, yPred)


def testModels(**kwargs):
    results = dict()

    if kwargs["rtnModels"] is None:
        kwargs["rtnModels"] = generateRtnModels(**kwargs)
    rtnModels = kwargs["rtnModels"]
    # Print individual return model results:
    printRtnModels(**kwargs)

    print("\nReturns Forecasted\n")
    sModels = generateCovariance(**kwargs)
    print("Covariance Forecasted\n\n\n")
    portModels = generatePortfolios()

    # Iterate through and test each model type one by one:
    for modelType in rtnModels:
        modelList = rtnModels[modelType]
        # modelList is a list of models that shares the same settings
        # but are for unique assets
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

                # Generate weights for each time period (ie daily weights):
                weights = generateWeights(testingLen, portFunc, sMatrices,
                                          returnVec, n, **kwargs)

                realized = []
                # Realize returns with portfolio weights for each time period:
                for i in range(testingLen):
                    temp = []
                    for j in range(n):
                        temp.append(kwargs["realizedReturns"][j][i])
                    temp = opt.matrix(temp)
                    realized.append(blas.dot(weights[i].T, temp))
                    if i is 0:
                        '''print("hereeeeee")
                        print(portFunc)
                        print(sMatrices[i])
                        print(weights[i])
                        print("returns:")
                        print(returnVec[0][i], returnVec[1][i])'''

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
                name = str(modelType) + str(sType).capitalize()\
                       + str(portType).capitalize()
                results.update({name: (stdDev, realized)})
                print("\n{} Evaluated".format(str(name)))


    # Print results:
    print("\n\nRealized Annual Results:\n")
    annualize = kwargs["periodsPerAnnum"]
    rf = kwargs["rf"]
    print("StDev  Rtn  Shrpe".rjust(outputRJust))
    for test in results:
        stdDev, realized = results[test]
        stdDev, avgRtn = stdDev*annualize**.5, np.mean(realized)*annualize
        sharpe = (avgRtn - rf) / stdDev
        #plt.hist(realized, bins=int(np.sqrt(len(realized))))
        #plt.show()
        result = "{} {:.3f} {:.3f} {:.3f}".format(test, round(stdDev, 3),
                                                  round(avgRtn, 3),
                                                  round(sharpe, 3))
        print(result.rjust(outputRJust))

    return results


def testGMV():
    return