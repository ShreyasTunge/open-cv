pip install scikit-learn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew

data = pd.read_csv(r'Clean_Dataset.csv', index_col=0)
data.head()
data.shape
data.info()
pd.options.display.max_columns = 20
data.flight.value_counts()
# Drop flight column bcuase it has lot of unnecessary unique values
data.drop(['flight'], axis=1, inplace=True)
data
data.isnull().sum()
data
sns.countplot(x='airline', data=data)
plt.figure(figsize=(12, 8))
sns.lineplot(x='days_left', y='price', data=data, hue='airline')
plt.figure(figsize=(12, 8));
sns.barplot(x='stops', y='price', data=data, hue='airline')
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in data.columns:
    if data[i].dtype == "object":
        data[i] = le.fit_transform(data[i])
data
# Convert Duration from hrs to mins
data['duration'] = data['duration'].apply(lambda x: int(round(x * 60)))
data.duration.value_counts()
training_data = data.iloc[0:210108]
testing_data = data.iloc[210108:]
print(training_data.shape)
print(testing_data.shape)
training_data.describe(include='all')
X = training_data.values[:, :-1]
Y = training_data.values[:, -1]
Y = Y.astype(int)
print(X.shape)
print(Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)
# scaler.fit_transform(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
np.sqrt(len(X_train))

from sklearn.ensemble import RandomForestRegressor

# create regressor object
regressor = RandomForestRegressor(n_estimators=100, n_jobs=-1)

# fit the regressor with x and y data
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
rmse = mse ** .5
print(mse)
print(rmse)
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

r2 = r2_score(Y_test, Y_pred)
print("R-squared:", r2)

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
print("RMSE:", rmse)

adjusted_r_squared = 1 - (1 - r2) * (len(Y) - 1) / (len(Y) - X.shape[1] - 1)
print("Adj R-square:", adjusted_r_squared)
regressor.score(X_train, Y_train)





