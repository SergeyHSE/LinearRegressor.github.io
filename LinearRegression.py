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
import numpy as np
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

# Let's try to take log(1 + x) from the length of the trip. We add one to avoid problems with trips that, for example, ended instantly.

plt.figure(figsize=(10, 6), dpi=100)
plt.hist(np.log1p(df_train.trip_duration), bins=100, color='skyblue', edgecolor='black')
plt.title('Distribution of Trip Durations(log scale)')
plt.xlabel('Log(Trip Duration + 1)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

df_train['log_trip_duration'] = np.log1p(df_train.trip_duration)
df_test['log_trip_duration'] = np.log1p(df_test.trip_duration)

#change from list to datetime objects

df.pickup_datetime = pd.to_datetime(df.pickup_datetime)

# Let's draw what the distribution of the number of trips by day looks like.

date_sorted = df_train.pickup_datetime.apply(lambda x: x.date()).sort_values()

plt.figure(figsize=(16, 6), dpi=150)
date_count_plot = sns.countplot(x=date_sorted, palette='viridis')
plt.grid(axis='y', linestyle='--', alpha=0.9)
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Pickup Count by Date')
date_count_plot.set_xticklabels(date_count_plot.get_xticklabels(), rotation=90, fontsize=10)
plt.tight_layout()
plt.show()

# Let's see what the distribution by the clock looks like.

df_train['pickup_hour'] = df_train.pickup_datetime.dt.hour

# Create a countplot for pickups by hour
plt.figure(figsize=(10, 6), dpi=100)
hour_count_plot = sns.countplot(x=df_train['pickup_hour'], palette='viridis')
plt.grid(axis='y', linestyle='--', alpha=0.9)
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.title('Pickup Count by Hour of Day')
plt.tight_layout()
plt.show()

# Now let's see how the day and the length of the trip are related.

group_by_weekday = df_train.groupby(df_train.pickup_datetime.apply(lambda x: x.date()))

plt.figure(figsize=(14, 8), dpi=100)
sns.relplot(data=group_by_weekday.log_trip_duration.aggregate('mean'), kind='line', height=6, aspect=2)
plt.xlabel('Weekday')
plt.ylabel('Mean Log Trip Duration')
plt.title('Average Log Trip Duration by Weekday')
plt.tight_layout()
plt.show()

"""
We are gonna prepare the dataset. Let's include the day of the year and the hour of the day in it. We need to write 'create_features' fuction, which
collect necessary attributes for us in a separate DataFrame.
"""
import datetime
def create_features(data_frame):
    X = pd.concat([data_frame.pickup_datetime.apply(lambda x: x.timetuple().tm_yday),
    data_frame.pickup_datetime.apply(lambda x: x.hour)],                  
    axis=1, keys=['day', 'hour'])
    
    return X, data_frame.log_trip_duration

X_train, y_train =  create_features(df_train)
X_train.shape
X_train.head()
X_test, y_test = create_features(df_test)

# Переменная час, хоть и является целым числом, не может трактоваться как вещественная. 
Дело в том, что после 23 идет 0, и что будет означать коэффициент регрессии в таком случае, совсем не ясно. 
Поэтому применим к этой переменной one -hot кодирование. В тоже время, переменная день должна остаться вещественной, 
так как значения из обучающей выборке не встреться нам на тестовом подмножестве.

# We can't use the variable 'hour' like numeric variable, becouse after 23 there is 0.
# Therefore, we apply 'One-Hot' encoding.

ohe = ColumnTransformer([("One hot", OneHotEncoder(sparse=False), [1] )],remainder="passthrough")
X_train = ohe.fit_transform(X_train)
X_test = ohe.fit_transform(X_test)
X_train.shape
X_train

# After that we can use 'Ridge regression' and also select parametrs with GridSearchCV

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, lin_reg.predict(X_test))

#Regularization
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1000).fit(X_train, y_train)
mean_squared_error(ridge.predict(X_test), y_test)

from sklearn.model_selection import GridSearchCV
grid_searcher = GridSearchCV(Ridge(),
                             param_grid={'alpha': np.linspace(100, 750, 10)},
                             cv=5).fit(X_train, y_train)

grid_searher.best_params_
mean_squared_error(y_test, grid_searher.predict(X_test))

# Let's build figure corresponding to the number of trips depending on the day of the week according to the training sample.

import calendar

df_train['day_of_week'] = df_train['pickup_datetime'].apply(lambda x: x.weekday())

# Calculate the count of trips for each day of the week
day_counts = df_train['day_of_week'].value_counts().reset_index()
day_counts.columns = ['day_of_week', 'trip_count']
# Map day of the week index to day name
day_counts['day_of_week'] = day_counts['day_of_week'].map(lambda x: calendar.day_name[x])
# Sort the DataFrame by the count of trips in descending order
day_counts = day_counts.sort_values(by='trip_count', ascending=False)

plt.figure(figsize=(10, 8), dpi=100)
week_count_plot = sns.barplot(x='day_of_week', y='trip_count', data=day_counts, palette='dark:#3498db')
plt.grid(axis='y', linestyle='--', alpha=0.9)
plt.title('Count of Trips by Day of the Week (Sorted)', fontsize=16)
plt.xlabel('Day of the Week', fontsize=14)
plt.ylabel('Count of Trips', fontsize=14)
plt.tight_layout()
plt.show()

"""
Let's add binary variables that equal 1 for two days with minimum count of trips and 0 for other
"""

date_sorted = df.pickup_datetime.apply(lambda x: x.date()).sort_values()
df['pickup_datetime'].dt.date.value_counts()[-5:]

# We also can look at these days by building the figure
plt.figure(figsize=(24, 6), dpi=150)  
date_count_plot = sns.countplot(x=date_sorted, palette='viridis') 
plt.grid(axis='y', linestyle='--', alpha=0.9)
plt.xlabel('Date') 
plt.ylabel('Count') 
plt.title('Pickup Count by Date')
date_count_plot.set_xticklabels(date_count_plot.get_xticklabels(), rotation=90, fontsize=10) 
plt.tight_layout()
plt.show()

# Create target_dates and column 'binary_feture'

target_dates = ['2016-01-24', '2016-01-23']
df_train['binary_feature'] = 0

# Iterate through the DataFrame and set 'binary_feature' to 1 for target dates

for date in target_dates:
    df_train.loc[df_train['pickup_datetime'].dt.date == pd.to_datetime(date).date(), 'binary_feature'] = 1

df_train.head()
df_train['binary_feature'].value_counts()

# Make the same things for test

df_test['binary_feature'] = 0
df_test.head()

for date in target_dates:
    df_test.loc[df_test['pickup_datetime'].dt.date == pd.to_datetime(date).date(), 'binary_feature'] = 1

df_test.head()
df_test['binary_feature'].value_counts()

"""
Now we should modify 'create_features' to concatenate hours, days, weeks and binary vareables.
"""

def create_features(data_frame):
    X = pd.concat([data_frame.pickup_datetime.apply(lambda x: x.timetuple().tm_yday),
                   data_frame.pickup_datetime.apply(lambda x: x.hour),
                   data_frame.binary_feature,
                   data_frame.pickup_datetime.apply(lambda x: x.weekday())],
                   axis=1, keys=['day', 'hour', 'binary_features', 'weekday'])
    return X, data_frame.log_trip_duration

X_train, y_train =  create_features(df_train)
X_test, y_test = create_features(df_test)
X_train.head()
X_train.tail()

# Apply one-hote encoding again

ohe_modify = ColumnTransformer([("One hot", OneHotEncoder(sparse=False), [1] )],remainder="passthrough")
X_train = ohe_modify.fit_transform(X_train)
X_test = ohe_modify.fit_transform(X_test)

# Calculate number of features
X_train.shape

# We have got 27 features, 
