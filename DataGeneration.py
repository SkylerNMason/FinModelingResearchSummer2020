import pandas as pd
import os
#from DataPreprocessing import *



def addDfToDict(fileLocation, startRow, fileType, dfDict, timeFormat):
    # Adds a dataframe to the dictionary
    df = buildDf(fileLocation, startRow, fileType, timeFormat)
    if not isinstance(df, int): # Used as a null input coming from a broken file - ignored
        dfDict[fileLocation.rsplit(os.sep, 1)[1]] = df
    return

def generateDfDict():
    # Creates dataframes (matrices) from inputted files which are then put into
    # a dictionary to be retrieved later.
    # The dictionary, dfDict, is setup such that the key is the filename.
    # Example: dfDict["ConsumerIndustry.xlsx"] would retrieve the dataframe
    # for ConsumerIndustry.xlsx.

    # TODO: Keep?

    startRow = 0
    '''
    while startRow < 0:
        try:
            startRow = int(input("What is the first row of usable data "
                                 "(zero indexed, excluding header)?"))
        except:
            print("That is not an integer input")
    '''

    # Determines data location and generates dataframes from the xlsx
    # or csv files which are then put into dfDict:
    dfDict = {}
    searching = True
    while searching:
        #fileLocation = input("File or folder path: ")
        #timeFormat = input("Date format of your dates ("%Y%m%d", etc): ")
        # TODO Remove line directly below this for manual inputting and uncomment above
        fileLocation = "E:\ProgrammingProjects\FinModelingResearchSummer2020\Data"
        timeFormat = '%Y%m'
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
                addDfToDict(fileLocation, startRow, fileType, dfDict, timeFormat)
                searching = False
        # Input is a folder of files:
        if os.path.isdir(fileLocation):
            for file in os.listdir(fileLocation):
                if os.path.isfile(fileLocation + os.sep + file):
                    fileType = file.rsplit(".", 1)[1].lower()
                    if fileType != "xlsx" and fileType != "csv":
                        print("Error: " + fileLocation+os.sep+file + " is an "
                              "invalid filetype in folder and was ignored "
                              "(must be xlsx or csv)")
                    else:
                        addDfToDict(fileLocation+os.sep+file, startRow,
                                    fileType, dfDict, timeFormat)
            searching = False
    return dfDict

def buildDf(fileLocation, startRow, fileType, timeFormat):
    # Takes a csv/xlsx file at fileLocation and builds a matrix out of
    # the given variables.
    # TODO: Make a better description. Involve what the csv file should look like.
    # TODO: Implement startRow
    # First col = date, second col = dependent variable (returns)
    # Use empty spaces for na_values
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
        df["Date"] = pd.to_datetime(df["Date"], format=timeFormat)
        return df

    except:
        print("Error: " + fileLocation + " uses incorrect data formatting and was ignored")
        return 0

def saveData(assetName, testSize, xTrain=None, xTest=None,
             yTrain=None, yTest=None, df=None):
    if xTrain is not None:
        numFeat = len(xTrain.columns)
        # Rebuild dataframe:
        colNames = list(yTrain.columns) + list(xTrain.columns)
        x = pd.concat([xTrain, xTest], axis=0, ignore_index=True)
        y = pd.concat([yTrain, yTest], axis=0, ignore_index=True)
        df = pd.concat([y, x], axis=1, ignore_index=True)
        df.columns = colNames
        print(df)

    else:
        numFeat = len(df.columns[2:])


    filename = (assetName.rsplit(".", 1)[0] + ", testSize " +
                str(int(testSize*100)) + "%, Features " + str(numFeat)
                + ".xlsx")

    path = "Data" + os.sep + "SavedData" + os.sep + str(filename)
    print(path)
    df.to_excel(path)

    return