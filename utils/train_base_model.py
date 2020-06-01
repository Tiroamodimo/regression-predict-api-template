"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# 0. Dependencies

# 0.1 Libraries, classes and functions

# Data Analysis
import pandas as pd
import numpy as np

# Model Libraries
import pickle
from sklearn.linear_model import LinearRegression

# preprocessing Classes and Functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
# import data_transformation as dt

# 0.2 Data Variables
list_predictors = ['Distance (KM)', 'Temperature', 'Pickup - Day of Month',
'Confirmation - Weekday (Mo = 1)', 'Platform Type']
str_target = 'Time from Pickup to Arrival'

# 0.3 Preprocessing Functions

def get_train_data(X, y, tsize = 0.2, rstate=16):

    """
    This function takes predictor, X, and response, Y, variables and returns
    the train data predctor,X_train and response y_train.
    """

    X_train = train_test_split(X, y, test_size = tsize, random_state = rstate)[0]

    y_train = train_test_split(X, y, test_size = tsize, random_state = rstate)[2]

    return X_train, y_train


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


# 1. Fetch training data and preprocess for modeling
train = pd.read_csv('utils/data/train_data.csv')
riders = pd.read_csv('utils/data/riders.csv')
train = train.merge(riders, how='left', on='Rider Id')

# 1.1 Select variables for model
feature_matrix = train[list_predictors].values
target_vector = train[[str_target]].values

# 1.2 get training data
X_train, y_train = get_train_data(feature_matrix, target_vector, tsize = 0.2, rstate=50)

# 1.3 impute missing values
X_train = impute_missing_data_median(X_train)

# 1.4 scale data
X_train = scale_data(X_train)

# 2. Fit model
lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)

# 3. Pickle model for use within our API
save_path = 'assets/trained-models/base_model_test.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))
