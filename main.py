"""
Financial Modeling Research: Risky Portfolio Optimization
Summer 2020
Overseen by Professor Burton Hollifield and Professor Bryan Routledge
Programmed By Skyler Mason

Installed plugins:
pandas
pandas_datareader (?)
matplotlib (?)
xldr
cvxopt
sklearn
statsmodels
arch
featuretools
"""

from DataManipulation import *
from PortfolioCreation import *
import os

def main():
    # TODO:
    #   Input:
    #       Take in parameters, clean
    # TODO:
    #   Process:
    #       Forecast mean, variance, cov to build risky portfolio
    # TODO:
    #   Output:
    #       Plug into utility function and output

    # Determines data location and relevant information:
    dfDict = generateDfDict()
    #print(dfDict["ConsumerIndustryReturns.xlsx"])

    returnData = pd.DataFrame()
    for asset in dfDict:
        returnData[asset] = dfDict[asset]["Returns"]
    returnData = returnData.T.to_numpy()
    print(globalMinVarPortfolio(returnData, annualize = 12))

    return 0

main()