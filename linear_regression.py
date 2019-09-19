### Tools for linear regression ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def simulate_data(nobs):
    """
    Simulates data for testing linear_regression models.
    INPUT
        nobs (int) the number of observations in the dataset
    RETURNS
        data (dict) contains X, y, and beta vectors.
    """
    X = np.random.random((nobs, 2))
    X = sm.add_constant(X)
    beta = [1, .1, .5]
    e = np.random.random(nobs)
    y = np.dot(X, beta) + e
    return {'X':X,'y':y,'beta':beta}



def compare_models(X, y):
    """
    Compares output from different implementations of OLS.
    INPUT
        X (ndarray) the independent variables in matrix form
        y (array) the response variables vector
    RETURNS
        results (pandas.DataFrame) of estimated beta coefficients
    """
    results = sm.OLS(y, X).fit()
    return(results.params)


def load_hospital_data(path_to_data):
    """
    Loads the hospital charges data set found at data.gov.
    INPUT
        path_to_data (str) indicates the filepath to the hospital charge data (csv)
    RETURNS
        clean_df (pandas.DataFrame) containing the cleaned and formatted dataset for regression
    """
    df = pd.read_csv(path_to_data)
    df = df.rename(columns = {' Average Covered Charges ':'average covered charges', 
                              ' Average Total Payments ':'average total payments',
                              ' Total Discharges ': 'total discharges',
                              'Provider State':'provide state',
                              'Average Medicare Payments':'average medicare payments'})
    df[['average covered charges','average total payments','total discharges','provide state']] = \
        df[['average covered charges','average total payments','total discharges','provide state']].astype(np.float64)
    df = df[(df['average covered charges']>0)&(df['average total payments']>0)&(df['total discharges']>0)&(df['average medicare payments']>0)]
    return df

def prepare_data(df):
    """
    Prepares hospital data for regression (basically turns df into X and y).
    INPUT
        df (pandas.DataFrame) the hospital dataset
    RETURNS
        data (dict) containing X design matrix and y response variable
    """
    y = df[['total discharges']].values
    X = sm.add_constant(df[['average total payments']])
    return {'X':X, 'y':y}


def run_hospital_regression(path_to_data):
    """
    Loads hospital charge data and runs OLS on it.
    INPUT
        path_to_data (str) filepath of the csv file
    RETURNS
        results (str) the statsmodels regression output
    """
    pass
 

### END ###