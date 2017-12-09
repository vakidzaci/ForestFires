import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from functions import errors
data = pd.read_csv("forestfires.csv")
data = data.drop(labels=['month','day'],axis=1)

y = data.area
x = data.drop(labels=['area'],axis=1)



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

reg = RadiusNeighborsRegressor()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)

for i in range(len(y_predict)):
    print("pred %s act %s"%(y_predict[i],y_test.ravel()[i]))

errors(y_test, y_predict)