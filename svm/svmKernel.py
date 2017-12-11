
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from functions import errors
from sklearn.model_selection import cross_val_score
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
data = remove_outlier_h(data,'area',0.83)
y = data.area
x = data.drop(labels=['area'], axis=1)
# data.drop(labels= labels,axis= 1,inplace =True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10, test_size=0.3)

err = []
err1 = []
err2 = []
err3 = []
err4 = []
out = []
kernels = [
           'linear',
           'poly',
           'rbf',
           'sigmoid',
           # 'precomputed'
           ]
w = []
for k in kernels:
    print(k)
    reg = SVR(kernel=k,
              degree=1,
              gamma='auto',
              coef0=0.0,
              tol=0.001,
              C=1.0,
              epsilon=2,
              shrinking=True,
              cache_size=200,
              verbose=False,
              max_iter=-1)
    reg.fit(x_train, y_train)
    y_predict = reg.predict(x_test)
    score = cross_val_score(reg, x, y, scoring='neg_mean_squared_error', cv=10).mean()
    w.append(score)
    print(score)
    # errors(y_test, y_predict)
    print("*******************************************************")


w = pd.DataFrame(w)


ind = np.arange(1, 5)
# plt.xticks(kernels)


l = np.array(w)
print(l[:,0])
plt.bar(ind,l[:,0])
plt.ylabel('cross val score')
plt.xticks(range(1,5), kernels)
plt.show()