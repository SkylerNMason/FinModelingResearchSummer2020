# TODO: Ideas include linear reg, k nearest neighbors, multilayer perceptron
# that thing that looks at the difference in t and t-1, support vector machines,
# support vector regression
# use cross validation to improve results

#https://www.aclweb.org/anthology/W19-6403.pdf
#

from DataManipulation import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.linear_model as sklm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn
import statsmodels.api as sm
import featuretools as ft

stdTestingRange = np.logspace(-3, 2, 100)

# Helper functions:

def normalizeData(xTrain, xTest, normalize, normFunc):
    # Returns x and y data
    if normalize is True:
        scalar = normFunc()
        xTrain = scalar.fit_transform(xTrain)
        if xTest is not 0:
            xTest = scalar.transform(xTest)
            return pd.DataFrame(xTrain), pd.DataFrame(xTest)
        return pd.DataFrame(xTrain), 0
    return xTrain, xTest

def prepData(df, testSize, normalize, randomState, normFunc, primitives):
    # Takes a dataframe and splits it into independent/dependent varialbes
    # plus what the test and training set will be (randomly assigned)

    # Parse and modify the data:
    x = df.iloc[:, 1:]
    y = df["Returns"]

    # TODO make modifyData also save the new dataset

    # Split and normalize the data:
    if testSize is not 0:
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=testSize,
                                                        random_state=randomState)
        xTrain, xTest = modifyDataset(xTrain, primitives)[0],\
                        modifyDataset(xTest, primitives)[0]
        print(xTrain)
        xTrain, xTest = normalizeData(xTrain, xTest, normalize, normFunc)
        return xTrain, xTest, yTrain, yTest
    else: # No testing set
        xTrain = modifyDataset(x, primitives)[0]
        xTrain, xTest = normalizeData(xTrain, 0, normalize, normFunc)
        return xTrain, xTest, y, 0

def testModel(testSize, models, asset, xTrain, xTest, yTrain, yTest):
    # Predicts and prints the results for testing a model
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
    if bestScore is None or (rsqrDif <= bestScore
        and testRsqr >= 0):
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


def mulLinReg(dfDict, testSize = .2, randomState=None, normalize=True,
              normFunc = StandardScaler, primitives=None):
    # Takes in a dictionary of assets and uses multiple linear regression to
    # fit a model predicting the returns.
    # Currently there is test/training built into the function
    models = dict()

    for asset in dfDict:
        df = dfDict[asset]
        xTrain, xTest, yTrain, yTest = prepData(df, testSize, normalize,
                                                randomState, normFunc, primitives)

        # Train the model:
        model = sklm.LinearRegression()
        fitModel = model.fit(xTrain, yTrain)
        models[asset] = fitModel

        # Test the model:
        testModel(testSize, models, asset, xTrain, xTest, yTrain, yTest)
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


def ridgeReg(dfDict, testSize = .2, alpha = -1, minRsqrDifScorer = False,
             randomState=None, alphaRange = stdTestingRange,
             normalize=True, normFunc = StandardScaler, primitives=None):
    # Note, smaller alpha provides for greater fitting/more complexity
    # Alpha of zero is the same as mulLinReg
    # Larger alphas damper coefficients (dense parameters still)
    models = dict()
    for asset in dfDict:
        df = dfDict[asset]

        xTrain, xTest, yTrain, yTest = prepData(df, testSize, normalize,
                                                randomState, normFunc,
                                                primitives)

        # Train the model:
        if alpha is not -1:
            model = sklm.Ridge(alpha=alpha)
            fitModel = model.fit(xTrain, yTrain)
            models[asset] = fitModel

        elif minRsqrDifScorer is True: # Use our own scoring method
            bestScore, bestModel = None, None
            for a in alphaRange:
                model = sklm.RidgeCV(alphas=[a])
                fitModel = model.fit(xTrain, yTrain)
                bestScore, bestModel = minRsqrDif(fitModel, xTrain, xTest,
                                                  yTrain, yTest, bestScore, bestModel)
            models[asset] = bestModel


        else: # Perform Leave-One-Out cross validation
              # to find optimal alpha value
            model = sklm.RidgeCV(alphas = alphaRange)
            fitModel = model.fit(xTrain, yTrain)
            models[asset] = fitModel

        # Test the model:
        testModel(testSize, models, asset, xTrain, xTest, yTrain, yTest)

    return models


# noinspection PyTypeChecker
def lassoReg(dfDict, testSize = .2, alpha = -1, minRsqrDifScorer = False,
             randomState=None, alphaRange=stdTestingRange,
             normalize=True, normFunc = StandardScaler, primitives=None):
    # Note, smaller alpha provides for greater fitting/more complexity
    # Alpha of zero is the same as mulLinReg
    # Larger alphas lead to more zero coefficients (sparse parameters)
    # This helps provide simplicity in the model
    models = dict()

    for asset in dfDict:
        df = dfDict[asset]
        xTrain, xTest, yTrain, yTest = prepData(df, testSize, normalize,
                                                randomState, normFunc,
                                                primitives)

        # Train the model:
        if alpha is not -1: # Use user inputted alpha
            model = sklm.Lasso(alpha=alpha)
            fitModel = model.fit(xTrain, yTrain)
            models[asset] = fitModel

        elif minRsqrDifScorer is True: # Use our own scoring method
            bestScore, bestModel = None, None
            for a in alphaRange:
                model = sklm.LassoCV(cv=10, alphas=[a])
                fitModel = model.fit(xTrain, yTrain)
                bestScore, bestModel = minRsqrDif(fitModel, xTrain, xTest,
                                                  yTrain, yTest, bestScore, bestModel)
            models[asset] = bestModel

        else: # Perform 10-fold cross validation to find optimal alpha value
            model = sklm.LassoCV(cv = 10, alphas=alphaRange)
            fitModel = model.fit(xTrain, yTrain)
            models[asset] = fitModel

        # Test the model:
        testModel(testSize, models, asset, xTrain, xTest, yTrain, yTest)

    return models

def elasticNet(dfDict, testSize=.2, alpha=-1, minRsqrDifScorer=False,
               randomState=None, alphaRange=stdTestingRange,
               lambaRatio=np.logspace(-1,0,25), normalize=True,
               normFunc = StandardScaler, primitives=None):
    # Note, smaller alpha provides for greater fitting/more complexity
    # Alpha of zero is the same as mulLinReg
    # Larger alphas lead to more zero coefficients (sparse parameters)
    # This helps provide simplicity in the model
    models = dict()

    for asset in dfDict:
        df = dfDict[asset]
        xTrain, xTest, yTrain, yTest = prepData(df, testSize, normalize,
                                                randomState, normFunc,
                                                primitives)

        # Train the model:
        if alpha is not -1: # Use user inputted alpha
            model = sklm.Lasso(alpha=alpha)
            fitModel = model.fit(xTrain, yTrain)
            models[asset] = fitModel

        elif minRsqrDifScorer is True: # Use our own scoring method
            bestScore, bestModel = None, None
            alphaRange = np.logspace(-3,2,40)
            for a in alphaRange:
                for l in lambaRatio:
                    model = sklm.ElasticNetCV(cv=10, alphas=[a], l1_ratio=l)
                    fitModel = model.fit(xTrain, yTrain)
                    bestScore, bestModel = minRsqrDif(fitModel, xTrain, xTest,
                                                      yTrain, yTest, bestScore, bestModel)
            models[asset] = bestModel

        else: # Perform 10-fold cross validation to find optimal alpha value
            model = sklm.ElasticNetCV(cv = 10, alphas=alphaRange,
                                      l1_ratio=lambaRatio)
            fitModel = model.fit(xTrain, yTrain)
            models[asset] = fitModel

        # Test the model:
        testModel(testSize, models, asset, xTrain, xTest, yTrain, yTest)

    return models





# File Testing:
dfDict = generateDfDict()

def callFuncs(includeMulLinReg = False, **kwargs):
    # Calls the various above models for testing with
    # **kwargs as the inputs
    #if includeMulLinReg: mulLinReg(**kwargs)
    #ridgeReg(**kwargs)
    lassoReg(**kwargs)
    #elasticNet(**kwargs)
    return

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
@ignore_warnings(category=ConvergenceWarning)
def test(randomState=None):
    # Possible parameters to include:
    # dfDict, testSize, alpha, minRsqrDifScorer, randomState,
    # alphaRange, lambdaRatio, normalize, normFunc


    kwargs = {'dfDict': dfDict, 'randomState': randomState, 'primitives': "All"}
    callFuncs(True, **kwargs)

    print("\n\n\n\n\n\nStage 1 Done\n\n\n\n\n\n")

    kwargs.update({'minRsqrDifScorer': True})
    #callFuncs(**kwargs)

    print("Done")
    return

test(1)
