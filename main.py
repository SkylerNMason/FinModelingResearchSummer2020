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

from Testing import *
from GlobalVariables import *


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

    modify = True
    if modify:
        # Modified testing kwargs:
        kwargs.update({'randomState': randomState, 'primitives': None,
                       "minRsqrDifScorer": False})
    kwargs = updateDict(**kwargs)


    print("\n\n\n\n\n\nStage 1 Done\n\n\n\n\n\n")

    realizedReturns = kwargs["realizedReturns"]

    # Baseline performance with one/n portfolio:
    basePerf = [*basePerfGen(**kwargs)]  # [Std. dev., return]
    print(basePerf)


    returnVec = []
    # stdDevPreds = garchModel(**kwargs)

    testModels(**kwargs)


    # Needs a dataframe with return vectors for each asset,
    # and optionally a covariance matrix
    # print(globalMinVarPortfolio(np.asarray(returnVec), S))

    # returnData = returnData.T.to_numpy()
    # print(globalMinVarPortfolio(returnData))

    print("Done")

    return 0


main()
