import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from functions import errors
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE
from functions import remove_outlier
from functions import remove_outlier_h

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score

data = pd.read_csv("forestfires.csv")
# data = data.drop(labels=['day'],axis=1)
# data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
# data.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)
# data = data.drop(data[data[['area']] >500])

df_day = pd.get_dummies(data['day'])
df_month = pd.get_dummies(data['month'])

# Join the dummy variables to the main dataframe
data = pd.concat([data, df_day], axis=1)
data = pd.concat([data, df_month], axis=1)

print(data.head())