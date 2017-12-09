import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from functions import errors
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE

data = pd.read_csv("../forestfires.csv")
# data = data.drop(labels=['day'],axis=1)
data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
data.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)
# data = data.drop(data[data[['area']] >500])
labels = [
    'X',
    'Y',
    'month',
    'day',
    'FFMC',
    'DMC',
    'DC',
    'ISI',
    'temp',
    'RH',
    'wind',
    'rain',
    # 'area'
]
# data.drop(labels=labels,axis=1,inplace=True)
y = data.area
x = data.drop(labels=['area'],axis=1)



# model = ExtraTreesRegressor()
# rfe = RFE(model, 3)
# fit = rfe.fit(x, y)


predMAE = []
predMSE = []
w = range(1,50)
# for i in range(1,100):
criterions = [('mse','red'),('mae','green')]

dMSE = []
dMAE = []

for crit, color in criterions:
    for i in labels:
        xD = x.drop(labels=[i], axis=1,inplace=False)
        x_train, x_test, y_train, y_test = train_test_split(xD, y, random_state=10, test_size=0.3)
        reg = RandomForestRegressor(
            n_estimators=30,
            criterion=crit,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
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
        if (crit == 'mse'):
            dMSE.append(mean_squared_error(y_test, y_predict))
        else:
            dMAE.append(mean_squared_error(y_test, y_predict))

d = pd.DataFrame()
d['names'] = labels
d['ERROR'] = dMSE
d.sort_values("ERROR",inplace=True)
d.to_csv("MSE.csv")
d = pd.DataFrame()
d['names'] = labels
d['ERROR'] = dMAE
d.sort_values("ERROR",inplace=True)
d.to_csv("MAE.csv")