
from sklearn.model_selection import train_test_split
import pandas as pd
import featuretools as ft
from sklearn.impute import SimpleImputer
import numpy as np
from DataGeneration import saveData
from featuretools.primitives import make_trans_primitive
from featuretools.variable_types import Numeric

debug = True

# Clean the Data:

def cleanData(df=None, testSize=-1, randomState=None, xTrain=None, xTest=None):
    # To clean just a single dataframe (like just the set of
    # independent vars) use testSize=0 and bothXandY=False,
    # then just reference the first returned output

    yTrain, yTest = 0, 0 # Default values for case where only
    # xTrain and xTest are imported

    if xTrain is None: # We are dealing with a df and not xTrain and xTest
        # NaN Values labeled below:
        df.replace([np.inf, -np.inf], value = np.nan, inplace=True)

        # Remove columns with too many nans:
        n = len(df)
        naTolerance = 1/5 # max % of a column that can be nan
        colNames = list(df.columns)
        '''for col in colNames:
            if df[col].isnull().sum() > n*naTolerance:
                df.drop(col, axis=1, inplace=True)'''

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
        xTrain.replace([np.inf, -np.inf],
                       value = np.nan, inplace=True)
        if xTest is not None:
            xTest.replace([np.inf, -np.inf],
                          value=np.nan, inplace=True)

    # Replace missing data with a constant 0:
    imputer = SimpleImputer(strategy='constant')
    colNames = list(xTrain.columns)

    tempXVals = xTrain.iloc[:, 1:].reset_index(drop=True)
    tempXDates = xTrain.iloc[:, 0:1].reset_index(drop=True)

    imputer = imputer.fit(tempXVals)

    tempXVals = pd.DataFrame(imputer.transform(tempXVals))
    xTrain = pd.concat([tempXDates, tempXVals], axis=1, ignore_index=True)
    xTrain.columns = colNames
    xTrain.replace(["missing_value"], value=0, inplace=True)

    if xTest is not 0:
        tempXVals = xTest.iloc[:, 1:].reset_index(drop=True)
        tempXDates = xTest.iloc[:, 0:1].reset_index(drop=True)

        tempXVals = pd.DataFrame(imputer.transform(tempXVals))
        xTest = pd.concat([tempXDates, tempXVals], axis=1, ignore_index=True)
        xTest.columns = colNames
        xTest.replace(["missing_value"], value=0, inplace=True)

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
    xTrain, xTest, yTrain, yTest = cleanData(df, testSize, randomState)

    # TODO make modifyData also save the new dataset

    # Perform feature engineering:
    if primitives is not None:
        xTrain, xTest = modifyDataset(xTrain, xTest, randomState, primitives)
    saveData(assetName, xTrain=xTrain, xTest=xTest, yTrain=yTrain, yTest=yTest)
    xTrain = xTrain.iloc[:,1:]
    xTest = xTest.iloc[:, 1:]

    # Normalize the data:
    xTrain, xTest = normalizeData(xTrain, xTest, normalize, normFunc)
    if debug: print("Finished prepping")
    return xTrain, xTest, yTrain, yTest


# Manipulate the Data:

def defineFedPrims():
    # Definitions of 7 transforms for the mccracken database
    prims = []
    small = 1e-6
    def firstDif(column):
        # Case two: First difference
        results = []
        for i in range(len(column)):
            if i < 1:
                results.append(np.nan)
            else:
                results.append(column[i] -column[i-1])
        return results
    FirstDif = make_trans_primitive(function=firstDif, input_types=[Numeric],
                                    return_type=Numeric)
    prims.append(FirstDif)

    def secondDif(column):
        # Case three: Second difference
        results = []
        for i in range(len(column)):
            if i < 2:
                results.append(np.nan)
            else:
                firstDif = column[i]-column[i-1]
                secondDif = column[i-1]-column[i-2]
                results.append(firstDif-secondDif)
        return results
    SecondDif = make_trans_primitive(function=secondDif, input_types=[Numeric],
                                     return_type=Numeric)
    prims.append(SecondDif)

    def LN(column):
        # Case four: Natural log
        results = []
        for i in range(len(column)):
            if column[i] < small:
                results.append(np.nan)
            else:
                results.append(np.log(column[i]))
        return results
    LN = make_trans_primitive(function=LN, input_types=[Numeric],
                                     return_type=Numeric)
    prims.append(LN)

    def firstDifLN(column):
        # Case five: First difference of natural log
        results = []
        for i in range(len(column)):
            if i < 1 or column[i]<small or column[i-1]<small:
                results.append(np.nan)
            else:
                first = np.log(column[i])
                second = np.log(column[i-1])
                results.append(first-second)
        return results
    FirstDifLN = make_trans_primitive(function=firstDifLN, input_types=[Numeric],
                                      return_type=Numeric)
    prims.append(FirstDifLN)

    def secondDifLN(column):
        # Case six: Second difference of natural log
        results = []
        for i in range(len(column)):
            if i < 2 or column[i]<small or column[i-1]<small or column[i-2]<small:
                results.append(np.nan)
            else:
                first = np.log(column[i]) - np.log(column[i-1])
                second = np.log(column[i-1]) - np.log(column[i-2])
                results.append(first-second)
        return results
    SecondDifLN = make_trans_primitive(function=secondDifLN, input_types=[Numeric],
                                       return_type=Numeric)
    prims.append(SecondDifLN)

    def firstDifPct(column):
        # Case Seven: First difference of percentage change
        results = []
        for i in range(len(column)):

            if i < 2 or column[i-1] is 0 or column[i-2] is 0:
                results.append(np.nan)
            else:
                first = column[i]/column[i-1] - 1
                second = column[i-1]/column[i-2] - 1
                results.append(first-second)
        return results
    FirstDifPct = make_trans_primitive(function=firstDifPct, input_types=[Numeric],
                                       return_type=Numeric)
    prims.append(FirstDifPct)

    return prims

def modifyDataset(xTrain, xTest, randomState, primitives):
    # Uses deep feature synthesis to create
    # a bunch of potentially useful variables
    if debug: print("Started Modifying")
    if primitives is "All":
        allPrims = ft.list_primitives()
        transformPrims = allPrims[allPrims.type == "transform"]["name"]
        # The above are all default transforms, below are ome hand picked ones:
        potentialPrims = ["not_equal", "absolute", "month", "diff",
                          "less_than_scalar", "less_than_equal_to_scalar",
                          "percentile", "subtract_numeric_scalar", "week",
                          "less_than", "modulo_numeric", "modulo_by_feature",
                          "divide_by_feature", "greater_than", "not_equal_scalar",
                          "subtract_numeric", "divide_numeric_scalar",
                          "greater_than_equal_to_scalar", "negate", "add_numeric",
                          "equal_scalar", "not", "multiply_numeric_scalar",
                          "and", "multiply_numeric", "less_than_equal_to",
                          "add_numeric_scalar", "greater_than_equal_to",
                          "or", "divide_numeric", "weekday", "modulo_numeric_scalar",
                          "greater_than_scalar"]
        primitives=potentialPrims
    if primitives is "fed":
        primitives = defineFedPrims()
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
    xTrainDates = xTrain.iloc[:, 1:2].reset_index(drop=True)
    colNames = ["Date"] + list(featureMatrixEnc.columns)
    xTrain = pd.concat([xTrainDates, featureMatrixEnc], axis=1,
                          ignore_index=True)
    xTrain.columns = colNames

    # Perform same transforms on testing set:
    if xTest is not 0:
        esTest = ft.EntitySet()
        esTest = esTest.entity_from_dataframe(entity_id="Asset", dataframe=xTest,
                                              time_index="Date", make_index=True,
                                              index="index")
        featureMatrixTest = ft.calculate_feature_matrix(features=featureDefEnc,
                                                        entityset=esTest)
        xTestDates = xTest.iloc[:, 1:2].reset_index(drop=True)
        colNames = ["Date"] + list(featureMatrixTest.columns)
        xTest = pd.concat([xTestDates, featureMatrixTest], axis=1,
                          ignore_index=True)
        xTest.columns= colNames

    xTrain, xTest = cleanData(xTrain=xTrain, xTest=xTest, randomState=randomState)[:2]
    if debug: print("Features:", len(featureDefEnc))
    if debug: print("Finished feature engineering")
    return xTrain, xTest

def featureSelection():

    return
