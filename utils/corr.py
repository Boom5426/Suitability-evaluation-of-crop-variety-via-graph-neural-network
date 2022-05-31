# -*- coding: utf-8 -*-
import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding('utf-8')
    sys.path.append('…')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D  # 不要去掉这个import
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error
import pandas as pd
import csv 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import metrics


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

f_data = open('result_nocorr.csv','w',encoding='utf-8')
csv_writer = csv.writer(f_data)

# 读取数据
csv_file=open('513_data.csv',encoding='utf-8')    #打开文件  
csv_reader_lines = csv.reader(csv_file)    #用csv.reader读文件  
f1 = open('corr.txt','w')
# f2 = open('result.txt','w'\]
date_PyList=[]
i=0 
for one_line in csv_reader_lines:
    # print(one_line)
    if i>0:
      data=[]
      for n in one_line:
         if n =='':
           data.append(float(0))
         else:
           data.append(float(n))  
      date_PyList.append(data)    #逐行将读到的文件存入python的列表  
    i=i+1
date_ndarray = np.array(np.array(date_PyList[1:len(date_PyList)]))
# d1 = date_ndarray[:,1:3]
# d2 = date_ndarray[:,5:9]
# d3 = date_ndarray[:,10].reshape(date_ndarray[:,10].shape[0],1)

# d4 = date_ndarray[:,13:19]
# d5 = date_ndarray[:,21:25]
# d6 = date_ndarray[:,29:33]
# d7 = date_ndarray[:,34:36]

# d8 = date_ndarray[:,37].reshape(date_ndarray[:,37].shape[0],1)


# # 导入数据
data = date_ndarray[:,:40]
# print(len(data))
data2 = pd.DataFrame(data, columns = ["最高气温均值","最高气温方差","平均气温均值","平均气温方差",\
  "最低气温均值","最低气温方差","温差均值","温差方差","地面气压均值","地面气压方差","相对湿度均值",\
    "相对湿度方差","降水量均值","降水量方差","最大风速均值","最大风速方差","平均风速均值","平均风速方差",\
      "风向角度均值","风向角度方差","日照时间均值","日照时间方差","风力等级均值","风力等级方差","大斑病",\
        "倒伏率","倒折率","灰斑病","株高","穗位高","空杆率","生育期","穗腐病","百粒重","穗长",\
          "秃尖长","鲜穗果重","亩产","增减产","评审状态"])

# 相关性计算
np.set_printoptions(threshold=10000)
print(data2.corr())
f1.write(str(data2.corr().values))

# data_X = np.concatenate([d1,d2,d3,d4,d5,d6,d7,d8],axis=1)
# data_y = date_ndarray[:, 38]

# x_train,x_test,y_train,y_test = train_test_split(data_X,data_y,test_size = 0.3,random_state = 1)
# models=[LinearRegression(),KNeighborsRegressor(),SVR(),Ridge(),Lasso(),MLPRegressor(alpha=20),DecisionTreeRegressor(),ExtraTreeRegressor(),RandomForestRegressor(),AdaBoostRegressor(),GradientBoostingRegressor(),BaggingRegressor()]
# models_str=['LinearRegression','KNNRegressor','SVR','Ridge','Lasso','MLPRegressor','DecisionTree','ExtraTree','RandomForest','AdaBoost','GradientBoost','Bagging']

# for name,model in zip(models_str,models):
#     print('开始训练模型：'+name)
#     model=model   #建立模型
#     model.fit(x_train,y_train)
#     y_pred=model.predict(x_test)  
  
#     print('Root Mean Squared Error:',r2_score(y_test, y_pred))
#     print('均方差:',mean_squared_error(y_test, y_pred))
#     print('回归方差:',explained_variance_score(y_test, y_pred))
#     print('平均绝对误差:',mean_absolute_error(y_test, y_pred))
#     print('中值绝对误差:',median_absolute_error(y_test, y_pred))

#     test=pd.DataFrame({name:y_test,name+"pred":y_pred})
#     name2 = "1" + name + '.csv'
#     test.to_csv(name2,encoding='gbk')
    # csv_writer.writerow(y_test)
    # csv_writer.writerow(y_pred)
    
  
    

