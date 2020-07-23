
from sklearn.model_selection import train_test_split
import pandas as pd
import featuretools as ft
from sklearn.impute import SimpleImputer
import numpy as np
from DataGeneration import saveData

#pd.set_option('display.max_columns', 6)
debug = True

# Clean the Data:

def cleanData(df=None, testSize=-1, randomState=None, xTrain=None, xTest=None):
    # To clean just a single dataframe (like just the set of
    # independent vars) use testSize=0 and bothXandY=False,
    # then just reference the first returned output

    yTrain, yTest = 0, 0 # Default values for case where only
    # xTrain and xTest are imported

    if xTrain is None: # We are dealing with a df and not xTrain and xTest
        df.replace([np.inf, -np.inf], value = np.nan, inplace=True)

        # Remove columns with too many nans:
        n = len(df)
        naTolerance = 1/5 # max % of a column that can be nan
        colNames = list(df.columns)
        for col in colNames:
            if df[col].isnull().sum() > n*naTolerance:
                df.drop(col, axis=1, inplace=True)

        # Parse the data:
        x = df.iloc[:, 1:]
        y = df.iloc[:, :1]

        # Split the data:
        if testSize is not 0:
            xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=testSize,
                                                            random_state=randomState,
                                                            shuffle=False)
        else:
            xTrain, xTest = x, 0
            yTrain, yTest = y, 0
    else: # We are importing xTrain and xTest
        xTrain.replace([np.inf, -np.inf], value = np.nan, inplace=True)
        if xTest is not None:
            xTest.replace([np.inf, -np.inf], value=np.nan, inplace=True)


    # Replace missing data with a constant 0:
    imputer = SimpleImputer(strategy='constant')
    colNames = list(xTrain.columns)

    tempXVals = xTrain.iloc[:, 1:].reset_index(drop=True)
    tempXDates = xTrain.iloc[:, 0:1].reset_index(drop=True)

    imputer = imputer.fit(tempXVals)

    tempXVals = pd.DataFrame(imputer.transform(tempXVals))
    xTrain = pd.concat([tempXDates, tempXVals], axis=1, ignore_index=True)
    xTrain.columns = colNames

    if xTest is not 0:
        tempXVals = xTest.iloc[:, 1:].reset_index(drop=True)
        tempXDates = xTest.iloc[:, 0:1].reset_index(drop=True)

        tempXVals = pd.DataFrame(imputer.transform(tempXVals))
        xTest = pd.concat([tempXDates, tempXVals], axis=1, ignore_index=True)
        xTest.columns = colNames

    return xTrain, xTest, yTrain, yTest


# Prepare the data:

def normalizeData(xTrain, xTest, normalize, normFunc):
    # Returns x and y data
    if debug: print("Started normalizing")
    if normalize is True:
        scalar = normFunc()
        xTrain = scalar.fit_transform(xTrain)
        if xTest is not 0:
            xTest = scalar.transform(xTest)
            return pd.DataFrame(xTrain), pd.DataFrame(xTest)
        return pd.DataFrame(xTrain), 0
    return xTrain, xTest

def prepData(df, testSize, normalize, randomState, normFunc, primitives,
             assetName):
    # Takes a dataframe and splits it into independent/dependent varialbes
    # plus what the test and training set will be (randomly assigned)
    if debug: print("Prepping")
    # Parse and clean:
    print(df)
    xTrain, xTest, yTrain, yTest = cleanData(df, testSize, randomState)

    # TODO make modifyData also save the new dataset

    # Perform feature engineering:
    xTrain, xTest = modifyDataset(xTrain, xTest, randomState, primitives)
    saveData(assetName, testSize, xTrain, xTest, yTrain, yTest)
    # Normalize the data:
    xTrain, xTest = normalizeData(xTrain, xTest, normalize, normFunc)
    if debug: print("Finished prepping")
    return xTrain, xTest, yTrain, yTest


# Manipulate the Data:

def modifyDataset(xTrain, xTest, randomState, primitives):
    # Uses deep feature synthesis to create
    # a bunch of potentially useful variables
    if primitives is None:
        return xTrain.iloc[:, 1:], xTest.iloc[:, 1:] # Return without a dates column
    if primitives is "All":
        allPrims = ft.list_primitives()
        transformPrims = allPrims[allPrims.type == "transform"]["name"]
        primitives = transformPrims # Shouldn't be indexed for all primitives
        #primitives = ["divide_numeric"]
        #print(primitives)
    esTrain = ft.EntitySet()
    esTrain = esTrain.entity_from_dataframe(entity_id = "Asset", dataframe=xTrain,
                                  time_index = "Date", make_index=True,
                                  index = "index")

    # Perform feature engineering:
    featureMatrix, featureDef = ft.dfs(entityset=esTrain, target_entity="Asset",
                                       max_depth=1, trans_primitives=primitives)

    # Encode any categorical variables:
    featureMatrixEnc, featureDefEnc = ft.encode_features(featureMatrix, featureDef,
                                                         include_unknown=False)

    xTrain = featureMatrixEnc.copy()
    # Perform same transforms on testing set:
    if xTest is not 0:
        esTest = ft.EntitySet()
        esTest = esTest.entity_from_dataframe(entity_id="Asset", dataframe=xTest,
                                              time_index="Date", make_index=True,
                                              index="index")
        featureMatrixTest = ft.calculate_feature_matrix(features=featureDefEnc,
                                                        entityset=esTest)
        xTest = featureMatrixTest.copy()




    xTrain, xTest = cleanData(xTrain=xTrain, xTest=xTest, randomState=randomState)[:2]
    if debug: print("Features:", len(featureDefEnc))
    if debug: print("Finished feature engineering")
    return xTrain, xTest
