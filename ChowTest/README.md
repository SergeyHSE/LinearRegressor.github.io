This project provides an implementation of the Chow break test.

The Chow test was initially developed by Gregory Chow in 1960 to test whether one regression or two or more regressions best characterise the data. As such, the Chow test is capable of detecting "structural breaks" within time-series. Additional information can be obtained from:

Chow, Gregory C. "Tests of equality between sets of coefficients in two linear regressions." Econometrica: Journal of the Econometric Society (1960): 591-605.

Toyoda, Toshihisa. "Use of the Chow test under heteroscedasticity." Econometrica: Journal of the Econometric Society (1974): 601-608.

This implementation supports simple linear models, and finding breaks where k = 2.

Installation
This module requires Python 3.0+ to run. The module can can be imported by:

pip install chowtest
from chow_test import chowtest
Input Arguments
The required input arguments are listed below:

Argument	Description
y	dependent variable (Pandas DataFrame Column)
X	independent variable(s) (Pandas Dataframe Column(s))
last_index_in_model_1	index of final point prior to assumed structural break (str)
first_index_in_model_2	index of first point following structural break (str)
significance_level	the significance level for hypothesis testing (float)
