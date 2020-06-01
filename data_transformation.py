
"""

    Data transformation functions for the pretrained model to be used within our API.

    Author: EDSA_2020 JHB_Team_16 Jedi_Nerds

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly together with model.py correctly

    ----------------------------------------------------------------------

    Description: This file contains several functions used for data preprocessing steps 
    within the API.

"""

# 1. Import Modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# 1. Import Data

train_df = pd.read_csv('utils/data/train_data.csv')

# 2. Functions to Extract Train Data:

def get_train_predictors(X, y, tsize = 0.2, rstate=16):

    """
    This function takes predictor, X, and response, Y, variables and returns
    the train data predctor,X_train.
    """

    X_train = train_test_split(X, y, test_size = tsize, random_state = rstate)[0]

    return X_train

def get_train_data(X, y, tsize = 0.2, rstate=16):

    """
    This function takes predictor, X, and response, Y, variables and returns
    the train data predctor,X_train and response y_train.
    """

    X_train = train_test_split(X, y, test_size = tsize, random_state = rstate)[0]

    y_train = train_test_split(X, y, test_size = tsize, random_state = rstate)[2]

    return X_train, y_train


# 3. Functions to impute missing data

def impute_request_data_median(list_features, str_response, predict_vector_df):

    """
    This function a dataframe of training data, df, list of features selected for model and datframe of variables
    to get a predict from as inputs. It returns a datframe of variables with null values replced with the median
    """

    # instantiate an imputer object with a median filling strategy
    imputer = SimpleImputer(missing_values = np.nan, strategy='median')

    # split predictors and response
    X = train_df[list_features].values

    y = train_df[str_response].values

    # extract training data to calibrate missing data
    X_train = get_train_predictors(X, y)

    # Calibrate imputation on training data
    imputer.fit(X_train)

    # replace null values with median
    predict_vector = imputer.transform(predict_vector_df.values.reshape(1,-1))
    
    # convert to dataframe
    return pd.DataFrame(predict_vector, columns=list_features)

def impute_missing_data_median(data):

    """
    This function takes a list of features selected for model and a dataframe of predictor variables, data, as inputs.
    It returns a datframe of predictor variables with null values replaced with the median
    """

    # instantiate an imputer object with a median filling strategy
    imputer = SimpleImputer(missing_values = np.nan, strategy='median')

    # Calibrate imputation on training data
    imputer.fit(data)

    # replace null values with median
    return imputer.transform(data)

# 3 Functions to scale data

def scale_request_data(list_features, str_response, predict_vector_df):

    """
    This function a dataframe of training data, df, list of features selected for model and datframe of variables
    to get a predict from as inputs. It returns a datframe of variables with feature values scaled.
    """

    # instantiate scaler object
    scaler = StandardScaler()


    # split predictors and response
    X = train_df[list_features].values

    y = train_df[str_response].values

    # extract training data to calibrate missing data
    X_train = get_train_predictors(X, y)

    # calibrate scaler object to train dataset
    scaler.fit(X_train)

    # scale data for prediction
    predict_vector = scaler.transform(predict_vector_df.values.reshape(1,-1))
    
    # convert to dataframe
    return pd.DataFrame(predict_vector, columns=list_features)

def scale_data(data):

    """
    This function takes a list of features selected for model and datframe of variables
    to get a predict from as inputs. It returns a datframe of variables with feature values scaled.
    """

    # instantiate scaler object
    scaler = StandardScaler()

    # calibrate scaler object to train dataset
    scaler.fit(data)

    # scale data for prediction
    return scaler.transform(data)