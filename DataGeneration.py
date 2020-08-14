from DataPreprocessing import *
import pandas as pd
import os
from joblib import dump, load
from GlobalVariables import *

def addDfToDict(dfDict, fileLocation, fileType, timeFormat, testSize,
                normalize, randomState, normFunc, primitives):
    # Adds 4 dataframes to the dictionary as a list representing
    # xTrain, xTest, yTrain, and yTest respectively
    assetName = fileLocation.rsplit(os.sep, 1)[1].rsplit(".", 1)[0]
    data = buildDf(fileLocation, fileType, timeFormat, testSize, normalize,
                   randomState, normFunc, primitives, assetName)
    if not isinstance(data, int): # Used as a null input coming from a broken file - ignored
        dfDict[assetName] = data
    return

def generateDfDict(fileLocation, timeFormat, testSize, normalize,
                   randomState, normFunc, primitives):
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
            randomState, normFunc, primitives, assetName):
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
                                                randomState, normFunc,
                                                primitives, assetName)

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

    '''path = "Data" + os.sep + "SavedModels" + os.sep + str(filename)
    load(path)'''
    return models