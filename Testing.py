from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ReturnForecasting import *
from PortfolioCreation import *
from VolatilityForecasting import *
from GlobalVariables import *
import dateutil.relativedelta


def updateDict(**kwargs):
    # Updates/creates the dfDict for a given kwargs
    new = {"dfDict": generateDfDict(**kwargs)}
    kwargs.update(new)

    realizedReturns = []  # Returns that occurred in actual reality
    stdDevs = []          # Standard deviation of realized returns
    assetNames = []
    for asset in kwargs["dfDict"]:
        # Based on yTest data for each asset:
        returns = kwargs["dfDict"][asset][3]["Returns"].tolist()
        realizedReturns.append(returns)
        stdDevs.append(np.std(returns, ddof=1))
        assetNames.append(str(asset).split(",")[0])
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
    kwargs.update({"realizedReturns": realizedReturns, "stdDevs": stdDevs,
                   "assetNames": assetNames,
                   "testingLen": len(realizedReturns[0])})
    return kwargs


def defaultModelKwargs():
    # fileLocation = input("File or folder path: ")
    # timeFormat = input("Date format of your dates ("%Y%m%d", etc): ")= "E:\ProgrammingProjects\FinModelingResearchSummer2020\Data"
    timeFormat = None  # Defaultly uses inferred time,
    # Make something like %Y%m to not use this feature

    kwargs = {"fileLocation": fileLocation,"periodsPerAnnum": 1,
              "timeFormat": timeFormat, "testSize": .22, "alpha": -1,
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


def printRtnModels(rtnModels, **kwargs):
    for modelType in rtnModels:
        i = 0
        print(sBrk + "Testing:", modelType + sBrk)
        modelList = rtnModels[modelType]
        for model in modelList:
            xTrain, xTest, yTrain, yTest = kwargs["dfDict"][model]
            print(model + ":")
            yPred = modelList[model][1]
            if extraDetails and i is 0:
                plotData(pd.DataFrame(yTest).reset_index(drop=True),
                         pd.DataFrame(yPred).reset_index(drop=True),
                         modelType + ": yTest vs yPred for first asset")
                i += 1
            printRtnResults(modelList[model][0], xTrain, xTest,
                            yTrain, yTest, yPred)
    return


def printCovModels(sModels, **kwargs):
    # Collects all of the values in a given set of matrices and prints out
    # a matrix with the averages and standard deviations for each position
    n = len(kwargs["dfDict"])
    testingLen = kwargs["testingLen"]
    for sType in sModels:
        print(sBrk + "Testing:", sType)
        sAvg = np.zeros(shape=(n, n))
        sStdDev = np.zeros(shape=(n, n))
        sMatrices = sModels[sType]
        for row in range(n):
            for col in range(n):
                tempValues = []
                for i in range(testingLen):
                    tempValues.append(sMatrices[i][row][col])
                sAvg[row][col] = np.mean(tempValues)
                sStdDev[row][col] = np.std(tempValues, ddof=1)
        np.set_printoptions(suppress=True, precision=6,
                            formatter={'float': '{:0.6f}'.format})
        print("Averaged entries from the covariance matrices:")
        print(sAvg)
        print(sBrk + "Standard deviation of the entries from the covariance matrices:")
        print(sStdDev)

    return


def generateRtnModels(**kwargs):
    # Generates a dictionary of the return forecasting models for testing
    # TODO: Add more models to test with
    models = dict()
    fileNameaddition = "MinRsqrDifScorer" + str(kwargs["minRsqrDifScorer"])
    models["lassoModels"+fileNameaddition] = lassoReg(**kwargs)
    models["ridgeModels" + fileNameaddition] = ridgeReg(**kwargs)
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
    sModels["hist"] = sMatrices

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
    #portModels["bruteGMV"] = bruteGMV  # Is usually the same as cvxoptGMV
    portModels["cvxoptGmv"] = globalMinVarPortfolio
    #portModels["oneNMixedGMV"] = oneNMixedGMV
    portModels["cvxoptSharpe"] = cvxoptSharpe
    #portModels["bruteSharpe"] = bruteSharpe
    #portModels["weakSharpe"] = weakMaxSharpe
    #portModels["oneNMixedSharpe"] = oneNMixedSharpe
    return portModels


def testModels(**kwargs):
    results = dict()
    n = len(kwargs['dfDict'])

    if kwargs["rtnModels"] is None:
        kwargs["rtnModels"] = generateRtnModels(**kwargs)
    rtnModels = kwargs["rtnModels"]
    # Print individual return model results:
    printRtnModels(**kwargs)

    print(sBrk + "Returns Forecasted" + brk)
    kwargs["sModels"] = generateCovariance(**kwargs)
    sModels = kwargs["sModels"]
    if n < 9:
        printCovModels(**kwargs)  # Cov matrices dont print properly if large
    print(sBrk + "Covariance Forecasted" + brk)
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


        testingLen = kwargs["testingLen"]
        # Iterate through and test each covariance forecast one by one
        for sType in sModels:
            sMatrices = sModels[sType]

            # Iterate through and test each portfolio type one by one
            for portType in portModels:
                portFunc = portModels[portType]

                # Generate weights for each time period (ie daily weights):
                weights = generateWeights(portFunc, sMatrices,
                                          returnVec, n, **kwargs)

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
                name = str(modelType) + str(sType).capitalize()\
                       + str(portType).capitalize()
                results.update({name: (stdDev, realized)})

                # Portfolio results printout:
                print(sBrk + "{} Evaluated".format(str(name)))
                print("Average Weights:")
                temp = np.concatenate(weights, axis=1)
                for i in range(n):
                    output = kwargs["assetNames"][i] + ": {:.5f}"\
                        .format(round(np.mean(temp[i]), 5))
                    print(output.rjust(outputRJust))
                print("Standard Deviation of Weights:")
                for i in range(n):
                    output = kwargs["assetNames"][i] + ": {:.5f}"\
                        .format(round(np.std(temp[i], ddof=1), 5))
                    print(output.rjust(outputRJust))


    # Print all results:
    print(brk + "Realized Annual Results:" + sBrk)
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

    # Print top results
    print(brk + "Top Realized Annual Results:" + sBrk)
    print("StDev  Rtn  Shrpe".rjust(outputRJust))
    for test in results:
        stdDev, realized = results[test]
        stdDev, avgRtn = stdDev*annualize**.5, np.mean(realized)*annualize
        sharpe = (avgRtn - rf) / stdDev
        result = "{} {:.3f} {:.3f} {:.3f}".format(test, round(stdDev, 3),
                                                  round(avgRtn, 3),
                                                  round(sharpe, 3))
        if sharpe >= kwargs["baseSharpe"]:
            print(result.rjust(outputRJust))

    return results


def testGMV():
    return