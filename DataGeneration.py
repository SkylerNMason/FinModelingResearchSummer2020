from DataPreprocessing import *
import pandas as pd
import os
from joblib import dump, load
from GlobalVariables import *

def addDfToDict(dfDict, fileLocation, fileType, timeFormat, testSize,
                normalize, randomState, normFunc, primitives, **excess):
    # Adds 4 dataframes to the dictionary as a list representing
    # xTrain, xTest, yTrain, and yTest respectively
    assetName = fileLocation.rsplit(os.sep, 1)[1].rsplit(".", 1)[0]
    data = buildDf(fileLocation, fileType, timeFormat, testSize, normalize,
                   randomState, normFunc, primitives, assetName, **excess)
    if not isinstance(data, int): # Used as a null check coming from a broken file, aka ignore it
        dfDict[assetName] = data
    return

def generateDfDict(fileLocation, timeFormat, testSize, normalize,
                   randomState, normFunc, primitives, **excess):
    # Creates dataframes (matrices) from inputted files which are then put into
    # a dictionary to be retrieved later.
    # The dictionary, dfDict, is setup such that the key is the filename.
    # Example: dfDict["ConsumerIndustry.xlsx"] would retrieve the dataframe
    # for ConsumerIndustry.xlsx.


    # Determines data location and generates dataframes from the xlsx
    # or csv files which are then put into dfDict:
    dfDict = {}
    searching = True
    while searching:

        # Checks that the input exists:
        while not os.path.exists(fileLocation):
            print("Invalid path")
            fileLocation = input("File or folder path: ")

        # Input is a specific file:
        if os.path.isfile(fileLocation):
            fileType = fileLocation.rsplit(".", 1)[1].lower()
            if fileType != "xlsx" and fileType != "csv":
                print("Invalid filetype (must be xlsx or csv)")
            else:
                fileType = fileLocation.rsplit(".", 1)[1]
                addDfToDict(dfDict, fileLocation, fileType, timeFormat, testSize,
                            normalize, randomState, normFunc, primitives)
                searching = False

        # Input is a folder of files:
        if os.path.isdir(fileLocation):
            for file in os.listdir(fileLocation):
                if os.path.isfile(fileLocation + os.sep + file) and file.count(".")>0:
                    fileType = file.rsplit(".", 1)[1].lower()
                    if fileType != "xlsx" and fileType != "csv":
                        print("Error: " + fileLocation+os.sep+file + " is an "
                              "invalid filetype in folder and was ignored "
                              "(must be xlsx or csv)")
                    else:
                        addDfToDict(dfDict, fileLocation+os.sep+file,
                                    fileType, timeFormat, testSize, normalize,
                                    randomState, normFunc, primitives)
            searching = False
    if debug: print("Generated dfDict")
    return dfDict


def buildDf(fileLocation, fileType, timeFormat, testSize, normalize,
            randomState, normFunc, primitives, assetName,
            **excess):
    # Takes a csv/xlsx file at fileLocation and builds a matrix out of
    # the given variables.
    # TODO: Make a better description. Involve what the csv file should look like.
    # First col = date, second col = dependent variable (returns)
    # Use empty spaces for na_values
    if debug: print("building")
    try:
        if (fileType == "xlsx"):
            df = pd.read_excel(fileLocation, na_values=[""])
        if (fileType == "csv"):
            df = pd.read_csv(fileLocation, na_values=[""])

        # Reshuffle data order and fix format:
        colNames = list(df.columns)
        newOrder = [colNames[1], colNames[0]] + colNames[2:]
        df = df.reindex(columns=newOrder)
        df.rename(columns = {colNames[1] : "Returns",
                             colNames[0]: "Date"}, inplace = True)
        if timeFormat is None:
            df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)
        else:
            df["Date"] = pd.to_datetime(df["Date"], format=timeFormat)

        xTrain, xTest, yTrain, yTest = prepData(df, testSize, normalize,
                                                randomState, normFunc, primitives,
                                                assetName)

        return [xTrain, xTest, yTrain, yTest]

    except:
        print("Error: " + fileLocation + " uses incorrect data formatting and was ignored")
        return 0


def saveModel(model, filename):
    # Saves models to Data/SavedModels folder for future reference
    path = "Data" + os.sep + "SavedModels" + os.sep + str(filename)
    dump(model, path)
    return

def loadModels(fileLocation):
    # Loads models from Data/SavedModels
    models = dict()

    # Input is a specific file:
    if os.path.isfile(fileLocation):
        file = fileLocation.rsplit(os.sep, 1)[1]
        if file.count(".") is 0:
            models[file.split(",", 1)[0]] = load(fileLocation)

    # Input is a folder of files:
    if os.path.isdir(fileLocation):
        for file in os.listdir(fileLocation):
            if os.path.isfile(fileLocation + os.sep + file) and file.count(".") is 0:
                models[file.split(",", 1)[0]] = load(fileLocation + os.sep + file)

    if len(models) is 0:
        return None
    else:
        return models


# Could make into separate data import file:

import pandas_datareader.data as web  # module for reading datasets directly from the web
from pandas_datareader.famafrench import get_available_datasets


def importFred(start=None, end=None):
    # incomplete implementation
    vars = ['RPI', "W875RX1", "DPCERA3M086SBEA", "CMRMTSPLx", "RETAILx",
            "INDPRO", "IPFPNSS", "IPFINAL", "IPCONGD", "IPDCONGD", "IPNCONGD",
            "IPBUSEQ", "IPMAT", "IPDMAT", "IPNMAT", "IPMANSICS", "IPB51222S",
            "IPFUELS", "CUMFNS", "HWI", "HWIURATIO", "CLF16OV", "CE16OV",
            "UNRATE", "UEMPMEAN", "UEMPLT5", "UEMP5TO14", "UEMP15OV",
            "UEMP15T26", "UEMP27OV", "CLAIMSx", "PAYEMS", "USGOOD",
            "CES1021000001", "USCONS", "MANEMP", "DMANEMP", "NDMANEMP",
            "SRVPRD", "USTPU", "USWTRADE", "USTRADE", "USFIRE", "USGOVT",
            "CES0600000007", "AWOTMAN", "AWHMAN", "HOUST", "HOUSTNE",
            "HOUSTMW", "HOUSTS", "HOUSTW", "PERMIT", "PERMITNE", "PERMITMW",
            "PERMITS", "PERMITW", "ACOGNO", "AMDMNOx", "ANDENOx", "AMDMUOx",
            "BUSINVx", "ISRATIOx", "M1SL", "M2SL", "M2REAL", "BOGMBASE",
            "TOTRESNS", "NONBORRES", "BUSLOANS", "REALLN", "NONREVSL",
            "CONSPI", "FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5",
            "GS10", "AAA", "BAA", "COMPAPFFx", "TB3SMFFM", "TB6SMFFM", "T1YFFM",
            "T5YFFM", "T10YFFM", "AAAFFM", "BAAFFM", "TWEXAFEGSMTHx",
            "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx", "WPSFD49207",
            "WPSFD49502", "WPSID61", "WPSID62", "OILPRICEx", "PPICMM",
            "CPIAUCSL", "CPIAPPSL", "CPITRNSL", "CPIMEDSL", "CUSR0000SAC",
            "CUSR0000SAD", "CUSR0000SAS", "CPIULFSL", "CUSR0000SA0L2",
            "CUSR0000SA0L5", "PCEPI", "DDURRG3M086SBEA", "DNDGRG3M086SBEA",
            "DSERRG3M086SBEA", "CES0600000008", "CES2000000008",
            "CES3000000008","UMCSENTx", "MZMSL", "DTCOLNVHFNM", "DTCTHFNM",
            "INVEST", "VXOCLSx"]

    return


def importKenFrench(startDate='1963-07-01', endDate=None):
    endDate = None
    allDatasets = get_available_datasets()
    setsToUse = [dataset for dataset in allDatasets if ('5' in dataset or '10'
                 in dataset or '30' in dataset or '49' in dataset)
                 and 'Industry_Portfolios_Wout_Div' in dataset]

    # Note that these monthly returns are eom for their given periods
    for portfolio in setsToUse:
        print(portfolio)
        # Uses monthly average value weighted returns which are transformed
        # according to lambda x: np.log(x/100+1)
        df = web.DataReader(portfolio, 'famafrench', start=startDate,
                            end=endDate)[0]

        n = len(df.columns)
        df = df.transform(lambda x: np.log(x/100+1)).reset_index()
        for i in range(n):
            asset = str(list(df.columns)[i+1])
            temp = df[["Date", asset]]
            assetName = asset + portfolio
            saveData(assetName, df=temp)

    return


def combinePredictedAndPredictors(xDf, yDf):

    return
