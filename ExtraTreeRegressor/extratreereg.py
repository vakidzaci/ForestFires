import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from functions import errors
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE
from functions import remove_outlier
data = pd.read_csv("../forestfires.csv")
# data = data.drop(labels=['day'],axis=1)
data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
data.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)
# data = data.drop(data[data[['area']] >500])
labels = [
    'X',
    'Y',
    'month',
    'day',
    'FFMC',
    'DMC',
    'DC',
    'ISI',
    'temp',
    'RH',
    'wind',
    'rain'
]
data = remove_outlier(data,'area')
y = data.area
x = data.drop(labels=['area'],axis=1)


forest = ExtraTreesRegressor()
forest.fit(x,y)
d = pd.DataFrame()
rfe = RFE(forest, 1)
fit = rfe.fit(x, y)





importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x.shape[1]):
    print("%s. feature %d (%f)" % (labels[f], indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), labels)
plt.xlim([-1, x.shape[1]])
plt.show()
d["Names"] = labels
d["Feature Ranking"] = fit.ranking_
d.sort_values("Feature Ranking",inplace=True)
d.to_csv("Feature Ranking.csv")