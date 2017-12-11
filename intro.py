import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions import remove_outlier_h
data = pd.read_csv("forestfires.csv")
import math

#CORELATION
f,ax = plt.subplots(figsize=(18, 18))
data = remove_outlier_h(data,'area',0.83)
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

#
# # Line Plot
# # color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
# data.wind.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
# data.RH.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
#
# plt.legend(loc='upper right')     # legend = puts label into plot
# plt.xlabel('x axis')              # label = name of label
# plt.ylabel('y axis')
# plt.title('Line Plot')            # title = title of plot
#




# # Scatter Plot
# # x = attack, y = defense
# data.plot(kind='scatter', x='temp', y='RH',alpha = 0.5,color = 'red')
# plt.xlabel('temp')              # label = name of label
# plt.ylabel('RH')
# plt.title('Attack Defense Scatter Plot')            # title = title of plot
#
#

# clf() = cleans it up again you can start a fresh



# Plotting all data
data1 = data.loc[:,["RH","temp","area"]]
data1.plot()
# it is confusing


plt.show()

