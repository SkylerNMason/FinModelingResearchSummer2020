import pandas as pd
import os
import featuretools as ft



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

def modifyDataset(df, primitives):
    # Uses deep feature synthesis to create
    # a bunch of potentially useful variables
    if primitives is None:
        return df.iloc[:, 1:], 0
    if primitives is "All":
        allPrims = ft.list_primitives()
        transformPrims = allPrims[allPrims.type == "transform"]["name"]
        primitives = transformPrims[2:3]
    es = ft.EntitySet()
    es = es.entity_from_dataframe(entity_id = "asset", dataframe=df,
                                  time_index = "Date", make_index=True,
                                  index = "index")
    featureMatrix, featureDef = ft.dfs(entityset = es, target_entity = "asset",
                                       max_depth = 1, trans_primitives=primitives,
                                       verbose=2)
    print(featureDef)
    featureMatrix, featureDef = ft.encode_features(featureMatrix, featureDef)
    featureMatrix.dropna(inplace = True)
    return featureMatrix, featureDef

def buildDf(fileLocation, startRow, fileType, timeFormat):
    # Takes a csv file at fileLocation and builds a matrix out of
    # the given variables.
    # TODO: Make a better description. Involve what the csv file should look like.
    # TODO: Implement startRow
    # First col = date, second col = dependent variable (returns)
    #Use empty spaces for na_values
    # Drops any NA values
    try:
        if (fileType == "xlsx"):
            df = pd.read_excel(fileLocation, na_values=[""])
        if (fileType == "csv"):
            df = pd.read_csv(fileLocation, na_values=[""])

        # Reshuffle data order and fix format:
        colNames = list(df.columns)
        df[colNames[0]], df[colNames[1]] = df[colNames[1]], df[colNames[0]]
        df.rename(columns = {colNames[0] : "Returns",
                             colNames[1]: "Date"}, inplace = True)
        df["Date"] = pd.to_datetime(df["Date"], format=timeFormat)
        return df.dropna()

    except:
        print("Error: " + fileLocation + " uses incorrect data formatting and was ignored")
        return 0
