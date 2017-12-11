import pandas as pd
import matplotlib.pyplot as plt
import functions as fnc
import math
data = pd.read_csv("../forestfires.csv")
# data = data.drop(labels=['day'],axis=1)
data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
data.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)


d = fnc.remove_outlier_h(data,'area',0.83)
y = d.area
plt.title("Burned area in ha")
plt.ylabel("Frequency")
plt.hist(y)
y = list(y)
print(len(y))
plt.show()

