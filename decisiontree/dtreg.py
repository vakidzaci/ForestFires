
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.model_selection import train_test_split
from functions import errors
data = pd.read_csv("../forestfires.csv")


# data = data.drop(labels=['month','day'],axis=1)
data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
data.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)

y = data.area
x = data.drop(labels=['area'],axis=1)



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

reg = DecisionTreeRegressor()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)

errors(y_test, y_predict)
print("********************************")
reg = ExtraTreesRegressor()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)

errors(y_test, y_predict)
print("********************************")

reg = BaggingRegressor()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)

errors(y_test, y_predict)