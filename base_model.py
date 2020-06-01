"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# 1. Helper Dependencies


# 1.1 import Libraries, classes and functions
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1.2 define variables and import training data
list_predictors = ['Distance (KM)', 'Temperature', 'Pickup - Day of Month',
'Confirmation - Weekday (Mo = 1)', 'Platform Type']

response = 'Time from Pickup to Arrival'

train_df = pd.read_csv('utils/data/train_data.csv')

# 1.3 define preprocessing functions

# 1.3.1 Get Training Predictors
def get_train_predictors(X, y, tsize = 0.2, rstate=16):

    """
    This function takes predictor, X, and response, Y, variables and returns
    the train data predctor,X_train.
    """

    X_train = train_test_split(X, y, test_size = tsize, random_state = rstate)[0]

    return X_train

# 1.3.1 Get Training Predictors
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
    X_train = get_train_predictors(X, y, rstate=50)

    # Calibrate imputation on training data
    imputer.fit(X_train)

    # replace null values with median
    predict_vector = imputer.transform(predict_vector_df.values.reshape(1,-1))
    
    # convert to dataframe
    return pd.DataFrame(predict_vector, columns=list_features)

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
    X_train = get_train_predictors(X, y, rstate=50)

    # calibrate scaler object to train dataset
    scaler.fit(X_train)

    # scale data for prediction
    predict_vector = scaler.transform(predict_vector_df.values.reshape(1,-1))
    
    # convert to dataframe
    return pd.DataFrame(predict_vector, columns=list_features)



def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    
    # Load the dictionary as a Pandas DataFrame.
    predict_vector = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    
    # 1. Select variables for model
    predict_vector = predict_vector[list_predictors]

    # 2. Impute Missing Data
    predict_vector = impute_request_data_median(list_predictors, response, predict_vector)

    # 3. Data Scaling
    predict_vector = scale_request_data(list_predictors, response, predict_vector)

    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return [prediction[0]]
