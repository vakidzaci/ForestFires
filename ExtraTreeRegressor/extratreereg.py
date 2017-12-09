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
    'rain'
]
data = remove_outlier(data,'area')
y = data.area
x = data.drop(labels=['area'],axis=1)


model = ExtraTreesRegressor()
d = pd.DataFrame()
rfe = RFE(model, 1)
fit = rfe.fit(x, y)
d["Names"] = labels
d["Feature Ranking"] = fit.ranking_
d.sort_values("Feature Ranking",inplace=True)
d.to_csv("Feature Ranking.csv")