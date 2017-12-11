
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
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
    # 'wind',
    'rain'
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
out = range(10,100,10)

for k in range(10,110,10):
    plt.figure()
    for j in range(10,110,10):

        plt.subplot(5, 2, j / 10)
        plt.title("" + str(j))
        for i in out:
            reg = MLPRegressor(hidden_layer_sizes=(j,i,k),
                               activation='relu',
                               solver='adam',
                               alpha=0.0001,
                               batch_size='auto',
                               learning_rate='constant',
                               learning_rate_init=0.001,
                               power_t=0.5,
                               max_iter=200,
                               shuffle=True,
                               random_state=None,
                               tol=0.0001,
                               verbose=False,
                               warm_start=False,
                               momentum=0.9,
                               nesterovs_momentum=True,
                               early_stopping=False,
                               validation_fraction=0.1,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-08)
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
        # plt.subplot(2,1,1)
        plt.plot(out, err, label='mean_squared_error')
        plt.legend(loc='upper left')
        # plt.subplot(2,1,2)
        # plt.plot(out, err1, label='mean_absolute_error')
        # plt.plot(out, err2, label='r2_score')
        # plt.plot(out, err3, label='median_absolute_error')
        # plt.plot(out, err4, label='explained_variance_score')
        # plt.legend(loc='upper left')
        err = []
        err1 = []
        err2 = []
        err3 = []
        err4 = []

plt.show()
