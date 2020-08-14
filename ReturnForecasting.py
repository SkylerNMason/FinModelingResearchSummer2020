# TODO: Ideas include linear reg, k nearest neighbors, multilayer perceptron
# that thing that looks at the difference in t and t-1, support vector machines,
# support vector regression
# use cross validation to improve results

#https://www.aclweb.org/anthology/W19-6403.pdf
#

from DataGeneration import *
from DataPreprocessing import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as sklm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn
import statsmodels.api as sm
from GlobalVariables import *

# Helper functions:

def testModels(testSize, models, dfDict):
    # Predicts and prints the results for testing a model
    for asset in dfDict:
        xTrain, xTest, yTrain, yTest = dfDict[asset]
        if testSize is not 0:
            yPred = models[asset].predict(xTest)
            printResults(models[asset], asset, xTrain, xTest, yTrain, yTest, yPred)
        else:
            yPred = models[asset].predict(xTrain)
            printResults(models[asset], asset, xTrain, 0, yTrain, yTrain, yPred)
    return

def printResults(model, asset, xTrain, xTest, yTrain, yTest, yPred):
    # Prints out a bunch of useful statistics to analyze the
    # effectiveness of the model

    # Print out results:
    print("!Results for:", asset,"\n" , model, "\n")
    print("Coef:", model.coef_)
    try:
        print("Alpha:", model.alpha_)
    except:
        try: print("Alpha:", model.alphas_)
        except: print("Alpha: na")
    try:
        print("Lambda:", model.l1_ratio_)
    except:
        print("Lambda: na")
        # Graphing:
    '''
    results = pd.DataFrame({"Actual": yTest, "Predicted": yPred})
    results.head(25).plot(kind='bar', figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title(str(model)[:str(model).find("(")])
    plt.show()'''
        # Statistics:

    print('Mean Absolute Error:', sklearn.metrics.mean_absolute_error(
        yTest, yPred))
    print('Mean Squared Error:', sklearn.metrics.mean_squared_error(
        yTest, yPred))
    print('Root Mean Squared Error:', np.sqrt(sklearn.metrics.mean_squared_error(
        yTest, yPred)))
    print("")
    trainRsqr = model.score(xTrain, yTrain)
    print("Train Score:", trainRsqr)
    if xTest is not 0:
        testRsqr = model.score(xTest, yTest)
        print("Test Score:", testRsqr)
    print("")
    return

def minRsqrDif(model, xTrain, xTest, yTrain, yTest,
               bestScore = None, bestModel = None):
    # Helper function for updating the below models based
    # on the % difference between the in sample and out of sample
    # r squared values away from out of sample r sqr.
    # Aka variation from out of sample testing/error away from test score
    # Ideal rsqrDif = 0%
    trainRsqr = model.score(xTrain, yTrain)
    testRsqr = model.score(xTest, yTest)
    rsqrDif = abs(trainRsqr - testRsqr)/abs(testRsqr)
    #print(abs(trainRsqr - testRsqr), rsqrDif, testRsqr)
    if bestScore is None or (rsqrDif <= bestScore):
        #print(rsqrDif, testAdjRsqr, "hereeeee")
        return rsqrDif, model
    return bestScore, bestModel

# Forecasting functions:

def historicalAvg(dfDict):
    # Returns the historical average returns
    returnVec = pd.DataFrame()
    for asset in dfDict:
        returnVec[asset] = [dfDict[asset]["Returns"].mean()]
    return returnVec


def mulLinReg(dfDict, **excess):
    # Takes in a dictionary of assets and uses multiple linear regression to
    # fit a model predicting the returns.
    # Currently there is test/training built into the function
    models = dict()

    for asset in dfDict:
        xTrain, xTest, yTrain, yTest = dfDict[asset]

        # Train the model:
        model = sklm.LinearRegression()
        fitModel = model.fit(xTrain, yTrain.values.ravel())
        models[asset] = [fitModel, fitModel.predict(xTest)]

        # Save the model:
        information = {"asset": asset.rsplit(".", 1)[0],
                       "testSize": excess["testSize"],
                       "numFeat": len(xTrain.columns),
                       "model": str(type(models[asset][0]).__name__)}
        filename = nameSave(**information)
        saveModel(models[asset][0], filename)
        '''
        # From statsmodels, use to get detailed stats:
        xTrain = sm.add_constant(xTrain)
        xTest = sm.add_constant(xTest)
        model2 = sm.OLS(yTrain, xTrain).fit()
        predictions = model2.predict(xTest)
        print(model2.summary())
        '''

    return models

def logReg():
    return

def ridgeReg(dfDict, alpha = -1, minRsqrDifScorer = False,
             alphaRange = stdTestingRange, **excess):
    # Note, smaller alpha provides for greater fitting/more complexity
    # Alpha of zero is the same as mulLinReg
    # Larger alphas damper coefficients (dense parameters still)
    models = dict()
    for asset in dfDict:
        xTrain, xTest, yTrain, yTest = dfDict[asset]

        # Train the model:
        if alpha is not -1:
            model = sklm.Ridge(alpha=alpha)
            fitModel = model.fit(xTrain, yTrain.values.ravel())
            models[asset] = [fitModel, fitModel.predict(xTest)]

        elif minRsqrDifScorer is True: # Use our own scoring method
            bestScore, bestModel = None, None
            for a in alphaRange:
                model = sklm.RidgeCV(alphas=[a])
                fitModel = model.fit(xTrain, yTrain.values.ravel())
                bestScore, bestModel = minRsqrDif(fitModel, xTrain, xTest,
                                                  yTrain, yTest, bestScore, bestModel)
            models[asset] = [bestModel, bestModel.predict(xTest)]


        else: # Perform Leave-One-Out cross validation
              # to find optimal alpha value
            model = sklm.RidgeCV(alphas = alphaRange)
            fitModel = model.fit(xTrain, yTrain.values.ravel())
            models[asset] = [fitModel, fitModel.predict(xTest)]

        # Save the model:
        information = {"asset": asset.rsplit(".", 1)[0],
                       "testSize": excess["testSize"],
                       "numFeat": len(xTrain.columns),
                       "model": str(type(models[asset][0]).__name__)}
        filename = nameSave(**information)
        saveModel(models[asset][0], filename)

    return models


def lassoReg(dfDict, alpha = -1, minRsqrDifScorer = False,
             alphaRange=stdTestingRange, **excess):
    # Note, smaller alpha provides for greater fitting/more complexity
    # Alpha of zero is the same as mulLinReg
    # Larger alphas lead to more zero coefficients (sparse parameters)
    # This helps provide simplicity in the model
    models = dict()
    if debug: print("Lasso Reg")
    for asset in dfDict:
        if debug: print("Made it to:", asset)
        xTrain, xTest, yTrain, yTest = dfDict[asset]

        # Train the model:
        if alpha is not -1: # Use user inputted alpha
            if debug: print("heyo-1")
            model = sklm.Lasso(alpha=alpha)
            fitModel = model.fit(xTrain, yTrain.values.ravel())
            models[asset] = [fitModel, fitModel.predict(xTest)]

        elif minRsqrDifScorer is True: # Use our own scoring method
            bestScore, bestModel = None, None
            if debug: print("heyo")
            for a in alphaRange:
                model = sklm.LassoCV(cv = 10, alphas=[a]) # Use Lasso or LassoCV?
                fitModel = model.fit(xTrain, yTrain.values.ravel())
                bestScore, bestModel = minRsqrDif(fitModel, xTrain, xTest,
                                                  yTrain, yTest, bestScore, bestModel)
            models[asset] = [bestModel, bestModel.predict(xTest)]

        else: # Perform 10-fold cross validation to find optimal alpha value
            if debug: print("heyo4")
            model = sklm.LassoCV(cv = 10, alphas=alphaRange)
            fitModel = model.fit(xTrain, yTrain.values.ravel())
            if debug: print(pd.DataFrame(np.array(list(zip(fitModel.coef_, xTrain.columns))).reshape(-1,2)))
            models[asset] = [fitModel, fitModel.predict(xTest)]

        # Save the model:
        if debug: print("heyo5")
        information = {"asset": asset.rsplit(".", 1)[0],
                       "testSize": excess["testSize"],
                       "numFeat": len(xTrain.columns),
                       "model": str(type(models[asset][0]).__name__)}
        filename = nameSave(**information)
        saveModel(models[asset][0], filename)

    return models

def elasticNet(dfDict, alpha=-1, minRsqrDifScorer=False,
               alphaRange=stdTestingRange, lambaRatio=np.logspace(-1,0,25),
               **excess):
    # Note, smaller alpha provides for greater fitting/more complexity
    # Alpha of zero is the same as mulLinReg
    # Larger alphas lead to more zero coefficients (sparse parameters)
    # This helps provide simplicity in the model
    models = dict()

    for asset in dfDict:
        xTrain, xTest, yTrain, yTest = dfDict[asset]

        # Train the model:
        if alpha is not -1: # Use user inputted alpha
            model = sklm.Lasso(alpha=alpha)
            fitModel = model.fit(xTrain, yTrain.values.ravel())
            models[asset] = [fitModel, fitModel.predict(xTest)]

        elif minRsqrDifScorer is True: # Use our own scoring method
            bestScore, bestModel = None, None
            alphaRange = np.logspace(-3,2,40)
            for a in alphaRange:
                for l in lambaRatio:
                    model = sklm.ElasticNetCV(cv=10, alphas=[a], l1_ratio=l)
                    fitModel = model.fit(xTrain, yTrain.values.ravel())
                    bestScore, bestModel = minRsqrDif(fitModel, xTrain, xTest,
                                                      yTrain, yTest, bestScore, bestModel)
            models[asset] = [bestModel, bestModel.predict(xTest)]

        else: # Perform 10-fold cross validation to find optimal alpha value
            model = sklm.ElasticNetCV(cv = 10, alphas=alphaRange,
                                      l1_ratio=lambaRatio)
            fitModel = model.fit(xTrain, yTrain.values.ravel())
            models[asset] = [fitModel, fitModel.predict(xTest)]

        # Save the model:
        information = {"asset": asset.rsplit(".", 1)[0],
                       "testSize": excess["testSize"],
                       "numFeat": len(xTrain.columns),
                       "model": str(type(models[asset][0]).__name__)}
        filename = nameSave(**information)
        saveModel(models[asset][0], filename)

    return models





'''# File Testing:

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
def test(randomState=None):
    # Possible parameters to include:
    # dfDict, testSize, alpha, minRsqrDifScorer, randomState,
    # alphaRange, lambdaRatio, normalize, normFunc

    # Default model kwargs:
    kwargs = defaultModelKwargs(randomState)

    # Different testing kwargs:
    kwargs.update({'randomState': randomState, 'primitives': 'fed',
                   "normalize": True})
    kwargs = updateDict(**kwargs)
    callFuncs(**kwargs)

    print("\n\n\n\n\n\nStage 1 Done\n\n\n\n\n\n")

    #kwargs.update({'minRsqrDifScorer': True})
    #callFuncs(**kwargs)

    print("Done")
    return

test(1)
'''