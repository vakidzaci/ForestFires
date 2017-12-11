import pandas as pd
import matplotlib.pyplot as plt
from functions import remove_outlier_y
import math
data = pd.read_csv("../forestfires.csv")
# data = data.drop(labels=['day'],axis=1)
data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
data.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)

print(data.describe())
y = data.area

l = range(1,len(y)+1)

plt.plot(l,y)
d = pd.DataFrame()
d['area'] = y
plt.plot(l,y)
y = remove_outlier_y(y)
print(y)
# print(len(y))
# y.hist()
# # plt.show()
# print(y.describe())
# l = range(1,len(y)+1)
# plt.plot(l,y)
plt.show()