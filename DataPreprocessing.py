
from sklearn.model_selection import train_test_split
import pandas as pd
import featuretools as ft
from sklearn.impute import SimpleImputer
import numpy as np


# Clean the Data:
#pd.set_option('display.max_rows', None)
def cleanData(df, testSize, randomState, bothXAndY=True):
    # To clean just a single dataframe (like just the set of
    # independent vars) use testSize=0 and bothXandY=False,
    # then just reference the first returned output
    df.replace([np.inf, -np.inf], value = np.nan, inplace=True)

    if bothXAndY:  # Dealing with both dependent and independent vars
        # Remove columns with too many nans:
        n = len(df)
        naTolerance = 1/5 # max % of a column that can be nan
        colNames = list(df.columns)
        for col in colNames:
            if df[col].isnull().sum() > n*naTolerance:
                df.drop(col, axis=1, inplace=True)

        # Parse the data:
        x = df.iloc[:, 1:]
        y = df["Returns"]
    else: # Dealing with just 1 dataframe
        x, y = df.iloc[:, 0:], 0

    # Split the data:
    if testSize is not 0:
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=testSize,
                                                        random_state=randomState)
    else:
        xTrain, xTest = x, 0
        yTrain, yTest = y, 0


    # Replace missing data with a constant 0:
    imputer = SimpleImputer(strategy='mean')

    tempXVals = xTrain.iloc[:, 1:].reset_index(drop=True)
    tempXDates = xTrain.iloc[:, 0:1].reset_index(drop=True)

    imputer = imputer.fit(tempXVals)

    tempXVals = pd.DataFrame(imputer.transform(tempXVals))
    xTrain = pd.concat([tempXDates, tempXVals], axis=1, ignore_index=True)
    xTrain.columns = x.columns

    if xTest is not 0:
        tempXVals = xTest.iloc[:, 1:].reset_index(drop=True)
        tempXDates = xTest.iloc[:, 0:1].reset_index(drop=True)

        tempXVals = pd.DataFrame(imputer.transform(tempXVals))
        xTest = pd.concat([tempXDates, tempXVals], axis=1, ignore_index=True)
        xTest.columns = x.columns

    return xTrain, xTest, yTrain, yTest


# Prepare the data:

def normalizeData(xTrain, xTest, normalize, normFunc):
    # Returns x and y data
    print("Started normalizing")
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
    print("Prepping")
    # Parse and modify the data:
    xTrain, xTest, yTrain, yTest = cleanData(df, testSize, randomState)


    # TODO make modifyData also save the new dataset

    # Split and normalize the data:
    xTrain = modifyDataset(xTrain, randomState, primitives)
    if xTest is not 0:
        xTest = modifyDataset(xTest, randomState, primitives)
    xTrain, xTest = normalizeData(xTrain, xTest, normalize, normFunc)
    print("finished prepping")
    return xTrain, xTest, yTrain, yTest


# Manipulate the Data:

def modifyDataset(df, randomState, primitives):
    # Uses deep feature synthesis to create
    # a bunch of potentially useful variables
    if primitives is None:
        return df.iloc[:, 1:] # Return without a dates column
    if primitives is "All":
        allPrims = ft.list_primitives()
        transformPrims = allPrims[allPrims.type == "transform"]["name"]
        primitives = transformPrims[0:1] # Shouldn't be indexed for all primitives
        primitives = ["divide_numeric"]
        print(primitives)
    es = ft.EntitySet()
    es = es.entity_from_dataframe(entity_id = "asset", dataframe=df,
                                  time_index = "Date", make_index=True,
                                  index = "index")
    featureMatrix, featureDef = ft.dfs(entityset = es, target_entity = "asset",
                                       max_depth = 1, trans_primitives=primitives,
                                       verbose=2)
    print(featureDef)
    featureMatrix, featureDef = ft.encode_features(featureMatrix, featureDef)
    featureMatrix = cleanData(featureMatrix, 0, randomState, False)[0]
    return featureMatrix
