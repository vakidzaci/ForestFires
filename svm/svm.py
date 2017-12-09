
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from functions import errors
data = pd.read_csv("../forestfires.csv")
data = data.drop(labels=['month','day'],axis=1)

y = data.area
x = data.drop(labels=['area'],axis=1)



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)



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

# for i in range(len(y_predict)):
#     print("pred %s act %s"%(y_predict[i],y_test.ravel()[i]))

errors(y_test, y_predict)