import logging
from sklearn.linear_model import LinearRegression as lr
from scipy.stats import f
import pandas as pd
import numpy as np

def chowtest(X, y, last_index_in_model_1, first_index_in_model_2, significance_level, dfn=5):
    '''
    Conducts a Chow Test.

    Inputs:
        X: independent variable(s) (Pandas DataFrame Column(s)).
        y: dependent variable (Pandas DataFrame Column (subset)).
        last_index_in_model_1: index of final point prior to assumed structural break (index value).
        first_index_in_model_2: index of the first point following the assumed structural break (index value).
        significance_level: the significance level for hypothesis testing (float).
        dfn: degrees of freedom for the numerator in the F-statistic (int).

    Outputs:
        result: a dictionary containing the Chow Statistic, p-value, and test outcome.

    References:
        Chow, Gregory C. "Tests of equality between sets of coefficients in two linear regressions."
        Econometrica: Journal of the Econometric Society (1960): 591-605.
    '''

    logger = logging.getLogger(__name__)

    # Check input data types
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise ValueError("X must be a DataFrame and y must be a Series.")

    # Check indices
    if not X.index.is_monotonic_increasing or not y.index.is_monotonic_increasing:
        logger.warning("Input indices are not monotonic increasing.")

    # Fit linear models and calculate residuals
    residuals_pooled = calculate_residuals(X, y)
    residuals1 = calculate_residuals(X.loc[:last_index_in_model_1], y.loc[:last_index_in_model_1])
    residuals2 = calculate_residuals(X.loc[first_index_in_model_2:], y.loc[first_index_in_model_2:])

    # Calculate RSS for the entire dataset
    rss_pooled = calculate_rss(residuals_pooled)
    
    # Calculate separate RSS values
    rss1 = calculate_rss(residuals1)
    rss2 = calculate_rss(residuals2)
    
    # Calculate Chow Statistic
    chow_statistic = calculate_chow_statistic(rss_pooled, rss1, rss2, X.shape[1])

    # Calculate p-value
    p_value = 1 - f.cdf(chow_statistic, dfn=dfn, dfd=(X1.shape[0] + X2.shape[0] - 2 * X.shape[1]))

    logger.info('*' * 100)
    logger.info(f'Chow Statistic: {chow_statistic}, p value: {p_value.round(5)}')
    logger.info('*' * 100)

    # Determine test outcome
    test_outcome = "Reject the null hypothesis" if p_value <= significance_level else "Fail to reject the null hypothesis"

    # Return a dictionary with the results
    result = {
        'Chow_Stat': chow_statistic,
        'p_value': p_value,
        'test_outcome': test_outcome
    }

    return result


def calculate_residuals(X, y):
    '''
    This function fits a linear model and calculates residuals.

    Inputs:
        X: independent variable(s) (Pandas Series' or Pandas DataFrame Column(s)).
        y: dependent variable (Pandas Series or Pandas DataFrame Column).

    Outputs:
        residuals: Pandas DataFrame containing error information.
    '''
    # fits the linear model:
    model = lr().fit(X, y)
    
    # creates a dataframe with the predicted y in a column called y_hat:
    summary_result = pd.DataFrame(columns=['y_hat'])
    yhat_list = [float(i) for i in model.predict(X)]
    summary_result['y_hat'] = yhat_list  
    # saves the actual y values in the y_actual column:
    summary_result['y_actual'] = y.values
    # calculates the residuals:
    summary_result['residuals'] = (summary_result.y_actual - summary_result.y_hat)
    # squares the residuals:
    summary_result['residuals_sq'] = (summary_result.residuals ** 2)
    
    return summary_result


def calculate_rss(residuals):
    '''
    This function calculates the sum of squared residuals.

    Inputs:
        residuals: Pandas DataFrame containing error information.

    Outputs:
        rss: sum of squared residuals (float).
    '''
    return residuals['residuals_sq'].sum()


def calculate_chow_statistic(rss_pooled, rss1, rss2, num_variables):
    '''
    This function calculates the Chow Statistic.

    Inputs:
        rss_pooled: RSS for the entire dataset.
        rss1: RSS for the first period.
        rss2: RSS for the second period.
        num_variables: number of independent variables.

    Outputs:
        chow_statistic: Chow Statistic (float).
    '''
    numerator = (rss_pooled - (rss1 + rss2)) / num_variables
    denominator = (rss1 + rss2) / (X1.shape[0] + X2.shape[0] - 2 * num_variables)
    return numerator / denominator
