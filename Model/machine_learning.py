import torch
import torch.nn as nn
import numpy as np
import pandas as pd
torch.set_default_tensor_type(torch.FloatTensor)
import torch.nn.functional as f
from torch.autograd import Variable
from pandas.core.frame import DataFrame
import torch.nn.functional as F


data_path = './our_data.xls'

# 处理性状数据
data =  pd.read_excel(data_path, sheet_name=0)
data_select = []
data_target = []
a, b = 0, 0
k = 0
for row in data.index.values:
    if data['评审状态'][row] == "淘汰":
        # if a >= 5000: # 均衡采样
        #     continue
        data_target.append(0)
        a += 1
    elif data['评审状态'][row] == "续试" or data['评审状态'][row] == "续试1" :
        # if b >= 5000:
        #     continue
        data_target.append(1)
        b += 1
    else:
        continue

    feature_1 = data['最高气温均值'][row]
    feature_2 = data['最高气温方差'][row]
    feature_3 = data['平均气温均值'][row]
    feature_4 = data['平均气温方差'][row]
    feature_5 = data['最低气温均值'][row]
    feature_6 = data['最低气温方差'][row]
    feature_7 = data['温差均值'][row]
    feature_8 = data['温差方差'][row]
    feature_9 = data['地面气压均值'][row]
    feature_10 = data['地面气压方差'][row]
    feature_11 = data['相对湿度均值'][row]
    feature_12 = data['相对湿度方差'][row]
    feature_13 = data['降水量均值'][row]
    feature_14 = data['降水量方差'][row]
    feature_15 = data['最大风速均值'][row]
    feature_16 = data['最大风速方差'][row]
    feature_17 = data['平均风速均值'][row]
    feature_18 = data['平均风速方差'][row]
    feature_19 = data['风向角度均值'][row]
    feature_20 = data['风向角度方差'][row]
    feature_21 = data['日照时间均值'][row]
    feature_22 = data['日照时间方差'][row]
    feature_23 = data['风力等级均值'][row]
    feature_24 = data['风力等级方差'][row]
    feature_25 = data['大斑病'][row] 
    feature_26 = data['倒伏率'][row]
    feature_27 = data['倒折率'][row]
    feature_28 = data['灰斑病'][row]
    feature_29 = data['株高'][row]
    if feature_29<10:
        feature_29 *= 100
    feature_30 = data['穗位高'][row]
    feature_31 = data['空杆率'][row]
    feature_32 = data['生育期'][row]
    feature_33 = data['穗腐病'][row]
    feature_34 = data['百粒重'][row]
    feature_35 = data['穗长'][row] 
    feature_36 = data['秃尖长'][row]
    feature_37 = data['鲜穗果重'][row]
    feature_38 = data['亩产'][row]
    # feature_39 = data['增减产'][row]

    data_select.append([feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, \
        feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, \
        feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23,\
        feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, \
        feature_31, feature_32, feature_33, feature_34, feature_35, feature_36, feature_37,  \
        feature_38])

print("正负样本比：",b,":", a) # 正负样本比例： 21979 : 15649 

data_select = DataFrame(data_select)
# print(data_select.shape)

# 特征做标准化
numeric_features = data_select.dtypes[data_select.dtypes != 'object'].index
data_select[numeric_features] = data_select[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

# 特征做归一化
# numeric_features = data_select.dtypes[data_select.dtypes != 'object'].index
# data_select[numeric_features] = data_select[numeric_features].apply(
#     lambda x: (x - x.min()) / (x.max()-x.min()))

# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
# data_select[numeric_features] = data_select[numeric_features].fillna(0)

# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(data_select, dummy_na=True)

# 确保没有NAN值
all_features = np.nan_to_num(all_features)
print(all_features.shape) # (198, 15)

# 确保标签和训练样本一一对应
assert len(all_features) == len(data_target)

# 选取数据

train_data = all_features[:8000]
train_label = data_target[:8000]
test_data = all_features[8000:10000]
test_label = data_target[8000:10000]

# train_features = torch.tensor(.values, dtype=torch.float)
# test_features = torch.tensor(all_features[3000:3500].values, dtype=torch.float)
# train_labels = torch.tensor(data_target[1000:3000], dtype=torch.float).view(-1, 1)
# test_labels = torch.tensor(data_target[3000:3500], dtype=torch.float).view(-1, 1)


from sklearn.linear_model import LogisticRegression # 导入逻辑回
from sklearn.ensemble import RandomForestClassifier # 导入随机森林包
from sklearn.svm import SVC # SVM
from sklearn import naive_bayes # 贝叶斯
from sklearn.neighbors import KNeighborsClassifier   # KNN
from sklearn import tree

model = [LogisticRegression(), RandomForestClassifier(), SVC(), naive_bayes.GaussianNB(), \
        KNeighborsClassifier(n_neighbors=15), KNeighborsClassifier(n_neighbors=5), \
        KNeighborsClassifier(n_neighbors=7), KNeighborsClassifier(n_neighbors=9), \
        KNeighborsClassifier(n_neighbors=12), tree.DecisionTreeClassifier()]
for i in model:
    clf = i
    # print(all_features[:3])
    # print(data_target[:3])

    # line = len(data_select)//5 * 4
    # print(line,":", all_features.shape[0]-line)

    x_train = train_data
    x_val = test_data

    y_train = train_label
    y_val = test_label

    clf.fit(x_train,y_train) # 模型训练，取前五分之四作训练集

    # print(x_val)
    predictions = clf.predict(x_val) # 模型测试，取后五分之一作测试集
    # print(predictions.shape)
    # predictions = [1]*400
    # print("Ground_Truth : ", y_val[:30])
    # print("Predictions:", predictions[:30])

    from sklearn.metrics import accuracy_score # 导入准确率评价指标
    print(str(i),' accuracy:%s'% accuracy_score(y_val, predictions))