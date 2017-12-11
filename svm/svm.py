
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
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
# data.drop(labels= labels,axis= 1,inplace =True)


err = []
err1 = []
err2 = []
err3 = []
err4 = []
out = []

for i in range(75,100,1):
    out.append(float(i) / 100)
    d = remove_outlier_h(data, 'area',float(i)/100)
    y = d.area
    plt.show()
    y = d.area
    x = d.drop(labels=['area'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10, test_size=0.3)

    reg = SVR(kernel='rbf',
              degree=1,
              gamma='auto',
              coef0=0.0,
              tol=0.001,
              C=1.0,
              epsilon=0.1,
              shrinking=True,
              cache_size=200,
              verbose=False,
              max_iter=-1)
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
plt.xticks(out)
plt.subplot(3,1,1)
plt.plot(out, err, label='mean_squared_error')
plt.legend(loc='lower left')
plt.subplot(3,1,2)

plt.plot(out, err1, label='mean_absolute_error')
plt.plot(out, err3, label='median_absolute_error')
plt.legend(loc='lower left')
plt.subplot(3,1,3)
plt.plot(out, err2, label='r2_score')
plt.plot(out, err4, label='explained_variance_score')
plt.legend(loc='lower left')
plt.show()
