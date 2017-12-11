import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from functions import errors

from functions import remove_outlier
data = pd.read_csv("../forestfires.csv")
data = data.drop(labels=['month','day'],axis=1)

y = data.area
x = data.drop(labels=['area'],axis=1)



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

reg = LinearRegression()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)

errors(y_test, y_predict)