"""
Financial Modeling Research: Risky Portfolio Optimization
Summer 2020
Overseen by Professor Burton Hollifield and Professor Bryan Routledge
Programmed By Skyler Mason

Installed plugins:
pandas
pandas_datareader
matplotlib (?)
xlrd
cvxopt
sklearn
statsmodels
arch
featuretools
openpyxl
fredapi
"""

from Testing import *
from GlobalVariables import *


from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def main():
    # TODO:
    #   Process:
    #       Forecast mean, variance, cov to build risky portfolio
    # TODO:
    #   Output:
    #       Plug into utility function and output

    # Default model kwargs:
    kwargs = defaultModelKwargs()

    modify = True
    if modify:
        # Modified testing kwargs:
        kwargs.update({'primitives': None, "minRsqrDifScorer": False,
                       "periodsPerAnnum": 12})
    kwargs = updateDict(**kwargs)


    annualize = kwargs["periodsPerAnnum"]
    dfDict = kwargs["dfDict"]
    testDates = (dfDict[list(dfDict.keys())[0]][1]).index
    start = list(testDates)[0]
    end = list(testDates)[-1]
    print(sBrk + "Test period starts:", start)
    print("Test period ends:", end)
    # Print out for the assets in use:
    print(sBrk + "Assets:" + sBrk)
    for i in range(len(kwargs["dfDict"])):
        xTrain, xTest, yTrain, yTest = list(kwargs["dfDict"].values())[i]
        print(kwargs["assetNames"][i])
        print("Annualized Training Values:")
        yTrain = yTrain["Returns"]
        print("Average return: {:.3f}".format(round(np.mean(yTrain)*annualize,
                                                    3)).rjust(outputRJust))
        print("Average standard deviation: {:.3f}".format(round(
              np.std(yTrain, ddof=1)*(annualize**.5), 3)).rjust(outputRJust))

        print("Annualized Testing Values:")
        yTest = yTest["Returns"]
        print("Average return: {:.3f}".format(round(np.mean(yTest)*annualize,
                                                    3)).rjust(outputRJust))
        print("Average standard deviation: {:.3f}".format(round(
              np.std(yTest, ddof=1)*(annualize**.5), 3)).rjust(outputRJust))
        print()
    print(sBrk + "Data Imported" + brk)

    # Baseline performance with one/n portfolio:
    basePerfRisk, basePerfRtn = basePerfGen(**kwargs)  # [Std. dev., return]
    basePerfRisk, basePerfRtn = basePerfRisk*annualize**.5, \
                                basePerfRtn*annualize

    # Note: risk premium is calculated based on averages and so
    # results in a very slightly different calculation
    sharpe = (basePerfRtn - kwargs["rf"]) / basePerfRisk
    kwargs["baseSharpe"] = sharpe
    testModels(**kwargs)

    result = "Baseline {:.3f} {:.3f} {:.3f}".format(round(basePerfRisk, 3),
                                                    round(basePerfRtn, 3),
                                                    round(sharpe, 3))
    print(result.rjust(outputRJust))
    print("Rf: {:.5f}".format(kwargs["rf"]).rjust(outputRJust))

    print(sBrk + "Done")

    return 0


main()
