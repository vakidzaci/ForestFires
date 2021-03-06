from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import pandas as pd
def errors(y_test,y_pred):
    print(mean_squared_error(y_test, y_pred))
    print(mean_absolute_error(y_test, y_pred))
    print(r2_score(y_test, y_pred))
    print(median_absolute_error(y_test, y_pred))
    print(explained_variance_score(y_test, y_pred))

    # print("mean_squared_error %s"%mean_squared_error(y_test,y_pred))
    # print("mean_absolute_error %s"%mean_absolute_error(y_test,y_pred))
    # print("r2_score %s"%r2_score(y_test,y_pred))
    # print("median_absolute_error %s"%median_absolute_error(y_test,y_pred))
    # print("explained_variance_score %s"%explained_variance_score(y_test,y_pred))

def ret_error(algo,y_test,y_pred):
    return algo,mean_squared_error(y_test,y_pred),mean_absolute_error(y_test,y_pred),r2_score(y_test,y_pred),median_absolute_error(y_test,y_pred),explained_variance_score(y_test,y_pred)

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0)
    q3 = df_in[col_name].quantile(0.83)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out
def remove_outlier_y(df_in):
    q1 = df_in.quantile(0)
    q3 = df_in.quantile(0.83)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in > fence_low) & (df_in < fence_high)]
    return df_out

def plot_act_pred(y_test,y_pred):
    l = range(1,len(list(y_test))+1)
    plt.plot(l,y_test)
    plt.plot(l,y_pred)
    plt.show()

def remove_outlier_h(df_in, col_name,h):
    q1 = df_in[col_name].quantile(0)
    q3 = df_in[col_name].quantile(h)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


def write_err(algo,y_test,y_pred):
    res = pd.read_csv('../result.csv')
    # d = pd.DataFrame(columns=['algorithm', 'mean_squared_error',
    #                             'mean_absolute_error',
    #                             'r2_score',
    #                             'median_absolute_error',
    #                             'explained_variance_score'])
    d = pd.DataFrame(ret_error(algo,y_test,y_pred))
    res.append(d)
    res.to_csv('../result.csv')

