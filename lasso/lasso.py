
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from functions import errors
from functions import remove_outlier_h
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
data = pd.read_csv("../forestfires.csv")
data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
data.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)
labels = [
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
data.drop(labels= labels,axis= 1,inplace =True)

d = remove_outlier_h(data, 'area',0.85)
y = d.area
x = d.drop(labels=['area'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10, test_size=0.3)



err = []
err1 = []
err2 = []
err3 = []
err4 = []
out = []

for i in range(1,100):
    w = float(i)/100
    print(w)
    out.append(w)
    reg = Lasso(alpha=w,
                fit_intercept=True,
                normalize=False,
                precompute=False,
                copy_X=True,
                max_iter=1000,
                tol=0.0001,
                warm_start=False,
                positive=False,
                random_state=None,
                selection='cyclic')
    reg.fit(x_train, y_train)
    y_predict = reg.predict(x_test)
    err.append(mean_squared_error(y_test, y_predict))
    err1.append(mean_absolute_error(y_test, y_predict))
    err2.append(r2_score(y_test, y_predict))
    err3.append(median_absolute_error(y_test, y_predict))
    err4.append(explained_variance_score(y_test, y_predict))
    errors(y_test, y_predict)
    print("*******************************************************")


def normalization(data):
    # data = np.array(data)
    # data = ((data - np.mean(data)) /
    #            np.std(data))
    # data = pd.DataFrame(data)
    return data


err = normalization(err)
err1 = normalization(err1)
err2 = normalization(err2)
err3 = normalization(err3)
err4 = normalization(err4)
# plt.xticks(out)
plt.subplot(2,1,1)
plt.plot(out, err, label='mean_squared_error')
plt.legend(loc='upper left')
plt.subplot(2,1,2)
plt.plot(out, err1, label='mean_absolute_error')
plt.plot(out, err2, label='r2_score')
plt.plot(out, err3, label='median_absolute_error')
plt.plot(out, err4, label='explained_variance_score')
plt.legend(loc='upper left')
plt.show()
