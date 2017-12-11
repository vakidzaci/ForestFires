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

data = pd.read_csv("../forestfires.csv")
# data = data.drop(labels=['day'],axis=1)
data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
data.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)
# data = data.drop(data[data[['area']] >500])
#
# df_day = pd.get_dummies(data['day'])
# df_month = pd.get_dummies(data['month'])

# Join the dummy variables to the main dataframe
# data = pd.concat([data, df_day], axis=1)
# data = pd.concat([data, df_month], axis=1)


labels = [
    'X',
    'Y',
    'month',
    'day',
    'FFMC',
    'DMC',
    'DC',
    # 'ISI',
    # 'temp',
    # 'RH',
    'wind',
    'rain'
]
# data = pd.DataFrame(data)
data.drop(labels=labels,axis=1,inplace=True)
data = remove_outlier(data,'area')
title = [
    # 'X',
    # 'Y',
    # 'month',
    # 'day',
    # 'FFMC',
    # 'DMC',
    # 'DC',
    # 'ISI',
    # 'temp',
    # 'RH',
    # 'wind',
    # 'rain'
    ]

d = remove_outlier_h(data,'area',0.83)
y = d.area
plt.show()
y = d.area
x = d.drop(labels=['area'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10, test_size=0.3)



reg = RandomForestRegressor(
    n_estimators=30,
    criterion='mae',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=10,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=1,
    random_state=None,
    verbose=0,
    warm_start=False)

reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
errors(y_test,y_predict)
l = range(1,len(y_test)+1)

plt.plot(l,y_test)
plt.plot(l,y_predict)
plt.show()