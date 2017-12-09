import random as rd
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
samples = 100000
a = [rd.randint(1,samples) for ia in range(samples)]
b = [rd.randint(1,samples) for ib in range(samples)]
c = [((a[i]**2)+b[i]**2)**.5 for i in range(samples)]

x = pd.DataFrame()
y = pd.DataFrame()



x['a'] = a
x['b'] = b
y['c'] = c

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.1)
mlp = MLPRegressor(hidden_layer_sizes=(100))
for p in range(100):
    mlp.fit(x_train,y_train['c'])
    y_pred = mlp.predict(x_test)
    print(mean_squared_error(y_test,y_pred))

