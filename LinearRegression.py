"""
id - a unique identifier for each trip
vendor_id - a code indicating the provider associated with the trip record
pickup_datetime - date and time when the meter was engaged
dropoff_datetime - date and time when the meter was disengaged
passenger_count - the number of passengers in the vehicle (driver entered value)
pickup_longitude - the longitude where the meter was engaged
pickup_latitude - the latitude where the meter was engaged
dropoff_longitude - the longitude where the meter was disengaged
dropoff_latitude - the latitude where the meter was disengaged
store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
trip_duration - duration of the trip in seconds
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import datetime
pd.set_option('display.max_columns', None)

# We can use Google colab to download file or download from this page: https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data?select=train.zip

# For colab:
!pip install -q kaggle
from google.colab import files
uploaded = files.upload()
!mkdir /root/.kaggle
!mv kaggle.json /root/.kaggle/kaggle.json
kaggle competitions download -c nyc-taxi-trip-duration
df = pd.read_csv('train.zip', compression='zip', header=0, sep=',', quotechar='"')

# You also may read this file downlouded it from kagle on the link above
path = r"your path"
df = pd.read_csv(path, header=0, sep=',', quotechar='"')

df.head()

df = df.drop('dropoff_datetime', axis=1)

# Now the dates are written as strings. Let's convert them into python datetime objects.
# This way we will be able to perform arithmetic operations with dates and pull out the necessary information without working with strings.

df.pickup_datetime = pd.to_datetime(df.pickup_datetime)

#sort date

df = df.sort_values(by='pickup_datetime')

df_train = df[:10 ** 6]
df_test = df[10 ** 6:]
len(df_test)

plt.figure(figsize=(10, 6), dpi=100)
plt.hist(df_train.trip_duration, bins=100, color='skyblue', edgecolor='black')
plt.title('Distribution of Trip Durations')
plt.xlabel('Trip Duration (seconds)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


