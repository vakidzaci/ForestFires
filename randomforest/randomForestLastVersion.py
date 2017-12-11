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
labels = [
    # 'X',
    # 'Y',
    # 'month',
    # 'day',
    # 'FFMC',
    # 'DMC',
    # 'DC',
    'ISI',
    'temp',
    'RH',
    # 'wind',
    # 'rain'
]
# data = remove_outlier(data,'area')
title = [
    # 'X',
    # 'Y',
    # 'month',
    # 'day',
    # 'FFMC',
    # 'DMC',
    # 'DC',
    'ISI',
    'temp',
    'RH',
    # 'wind',
    # 'rain'
    ]

err = []
err1 = []
err2 = []
err3 = []
err4 = []
out = []
for i in range(75,100):
    out.append(float(i)/100)
    d = remove_outlier_h(data,'area',float(i)/100)
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
    err.append(mean_squared_error(y_test,y_predict))
    err1.append(mean_absolute_error(y_test,y_predict))
    err2.append(r2_score(y_test,y_predict))
    err3.append(median_absolute_error(y_test,y_predict))
    err4.append(explained_variance_score(y_test,y_predict))
    errors(y_test,y_predict)


def normalization(data):
    # data = np.array(data)
    # data = ((data - np.mean(data)) /
    #            np.std(data))
    # data = pd.DataFrame(data)
    return data
# err = normalization(err)
# err1 = normalization(err1)
# err2 = normalization(err2)
# err3 = normalization(err3)
# err4 = normalization(err4)
plt.subplot(2,1,1)
plt.plot(out,err,label='mean_squared_error')
plt.legend()
plt.subplot(2,1,2)
plt.plot(out,err1,label='mean_absolute_error')
plt.plot(out,err2,label='r2_score')
plt.plot(out,err3,label='median_absolute_error')
plt.plot(out,err4,label='explained_variance_score')
plt.legend(loc='lower left')
plt.show()
