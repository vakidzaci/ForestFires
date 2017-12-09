import pandas as pd
import matplotlib.pyplot as plt
import functions as fnc
import math
data = pd.read_csv("../forestfires.csv")
# data = data.drop(labels=['day'],axis=1)
data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
data.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)

data = fnc.remove_outlier(data,'area')
y = data.area
# plt.subplot(1,2,1)
plt.title("Burned area in ha")
plt.ylabel("Frequency")
plt.hist(y)
y = list(y)
# for i in range(len(y)):
#     y[i] = math.log1p(y[i])
# y = pd.DataFrame(y)
# plt.subplot(1,2,2)
# plt.title("Ln(area+1)")
# plt.ylabel("Frequency")
# plt.hist(y)
plt.show()

