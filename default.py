import numpy
import pandas as pd
import functions as fnc
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

seed = 7
numpy.random.seed(seed)

dataframe = pd.read_csv("forestfires.csv")

dataframe.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
dataframe.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)

dataset = dataframe.values

X = dataset[:,0:12]
Y = dataset[:,12]

X = pd.DataFrame(X)
Y = pd.DataFrame(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.3)
# model = ExtraTreesRegressor()
# rfe = RFE(model, 3)
# fit = rfe.fit(X, Y)
modelnames = ['LinearRegression','Ridge','Lasso',
              'ElasticNet','BaggingRegressor','RandomForest','ExtraTreesRegressor',
              'KNeighborsRegressor','DecisionTreeRegressor',
              'MLPRegressor','SVR']
num_instances = len(X)
models = []
models.append(('LinearRegression', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Lasso', Lasso()))
models.append(('ElasticNet', ElasticNet()))
models.append(('BaggingRegressor', BaggingRegressor()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
models.append(('KNeighborsRegressor', KNeighborsRegressor()))
models.append(('DecisionTreeRegressor', DecisionTreeRegressor()))
models.append(('MLPRegressor', MLPRegressor()))
models.append(('SVR', SVR()))

# Evaluations
results = []
names = []
mse = []
mae = []
r2s = []
mde = []
evs = []

for name, model in models:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    names.append(name)
    ms,ma,r2,md,ev = fnc.ret_error(y_test,predictions)
    mse.append(ms)
    mae.append(ma)
    r2s.append(r2)
    mde.append(md)
    evs.append(ev)
    # print("************* %s ****************"%name)
    # fnc.errors(y_test,predictions)

d = pd.DataFrame()
d['algorithms'] = names
d['mse'] = mse
d['mae'] = mae
d['r2'] = r2s
d['median_absolute_error'] = mde
d['explained variance error'] = evs



d.to_csv("default_error.csv")

