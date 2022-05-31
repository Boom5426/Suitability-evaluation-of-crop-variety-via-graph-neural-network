import torch
import torch.nn as nn
import numpy as np
import pandas as pd
torch.set_default_tensor_type(torch.FloatTensor)
import torch.nn.functional as f
from torch.autograd import Variable
from pandas.core.frame import DataFrame


data_path = './our_data.xls'

# 处理性状数据
data =  pd.read_excel(data_path, sheet_name=0)
data_select = []
data_target = []
a, b = 0, 0
k = 0
for row in data.index.values:
    if data['评审状态'][row] == "淘汰":
        # if a >= 2000: # 均衡采样
        #     continue
        data_target.append(0)
        a += 1
    elif data['评审状态'][row] == "续试" or data['评审状态'][row] == "续试1" :
        # if b >= 2000:
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


    # log 平滑处理 ： 长尾分布 --> 正态分布
    # feature_1 = np.log(feature_1+1) if feature_1>=0 else -np.log(abs(feature_1)+1)
    # feature_2 = np.log(feature_2+1) if feature_2>=0 else -np.log(abs(feature_2)+1)
    # feature_3 = np.log(feature_3+1) if feature_3>=0 else -np.log(abs(feature_3)+1)
    # feature_4 = np.log(feature_4+1) if feature_4>=0 else -np.log(abs(feature_4)+1)
    # feature_5 = np.log(feature_5+1) if feature_5>=0 else -np.log(abs(feature_5)+1)
    # feature_6 = np.log(feature_6+1) if feature_6>=0 else -np.log(abs(feature_6)+1)
    # feature_7 = np.log(feature_7+1) if feature_7>=0 else -np.log(abs(feature_7)+1)
    # feature_8 = np.log(feature_8+1) if feature_8>=0 else -np.log(abs(feature_8)+1)
    # feature_9 = np.log(feature_9+1) if feature_9>=0 else -np.log(abs(feature_9)+1)
    # feature_10 = np.log(feature_10+1) if feature_10>=0 else -np.log(abs(feature_10)+1)
    # feature_11 = np.log(feature_11+1) if feature_11>=0 else -np.log(abs(feature_11)+1)
    # feature_12 = np.log(feature_12+1) if feature_12>=0 else -np.log(abs(feature_12)+1)
    # feature_13 = np.log(feature_13+1) if feature_13>=0 else -np.log(abs(feature_13)+1)
    # feature_14 = np.log(feature_14+1) if feature_14>=0 else -np.log(abs(feature_14)+1)
    # feature_15 = np.log(feature_15+1) if feature_15>=0 else -np.log(abs(feature_15)+1)
    # feature_16 = np.log(feature_16+1) if feature_16>=0 else -np.log(abs(feature_16)+1)
    # feature_17 = np.log(feature_17+1) if feature_17>=0 else -np.log(abs(feature_17)+1)
    # feature_18 = np.log(feature_18+1) if feature_18>=0 else -np.log(abs(feature_18)+1)
    # feature_19 = np.log(feature_19+1) if feature_19>=0 else -np.log(abs(feature_19)+1)
    # feature_20 = np.log(feature_20+1) if feature_20>=0 else -np.log(abs(feature_20)+1)
    # feature_21 = np.log(feature_21+1) if feature_21>=0 else -np.log(abs(feature_21)+1)
    # feature_22 = np.log(feature_22+1) if feature_22>=0 else -np.log(abs(feature_22)+1)
    # feature_23 = np.log(feature_23+1) if feature_23>=0 else -np.log(abs(feature_23)+1)
    # feature_24 = np.log(feature_24+1) if feature_24>=0 else -np.log(abs(feature_24)+1)
    # feature_25 = np.log(feature_25+1) if feature_25>=0 else -np.log(abs(feature_25)+1)
    # feature_26 = np.log(feature_26+1) if feature_26>=0 else -np.log(abs(feature_26)+1)
    # feature_27 = np.log(feature_27+1) if feature_27>=0 else -np.log(abs(feature_27)+1)
    # feature_28 = np.log(feature_28+1) if feature_28>=0 else -np.log(abs(feature_28)+1)
    # feature_29 = np.log(feature_29+1) if feature_29>=0 else -np.log(abs(feature_29)+1)
    # feature_30 = np.log(feature_30+1) if feature_30>=0 else -np.log(abs(feature_30)+1)
    # feature_31 = np.log(feature_31+1) if feature_31>=0 else -np.log(abs(feature_31)+1)
    # feature_32 = np.log(feature_32+1) if feature_32>=0 else -np.log(abs(feature_32)+1)
    # feature_33 = np.log(feature_33+1) if feature_33>=0 else -np.log(abs(feature_33)+1)
    # feature_34 = np.log(feature_34+1) if feature_34>=0 else -np.log(abs(feature_34)+1)
    # feature_35 = np.log(feature_35+1) if feature_35>=0 else -np.log(abs(feature_35)+1)
    # feature_36 = np.log(feature_36+1) if feature_36>=0 else -np.log(abs(feature_36)+1)
    # feature_37 = np.log(feature_37+1) if feature_37>=0 else -np.log(abs(feature_37)+1)
    # feature_38 = np.log(feature_38+1) if feature_38>=0 else -np.log(abs(feature_38)+1)
    # feature_39 = np.log(feature_39+1) if feature_39>=0 else -np.log(abs(feature_39)+1)

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
train_data = torch.tensor(all_features[:8000], dtype=torch.float)
train_label = torch.tensor(data_target[:8000], dtype=torch.float)
test_data = torch.tensor(all_features[8000:10000], dtype=torch.float)
test_label = torch.tensor(data_target[8000:10000], dtype=torch.float)
# n_train = train_data.shape[0]
# train_features = torch.tensor(all_features[:line].values, dtype=torch.float)
# test_features = torch.tensor(all_features[line:].values, dtype=torch.float)
# train_labels = torch.tensor(data_target[:line], dtype=torch.float).view(-1, 1)
# test_labels = torch.tensor(data_target[line:], dtype=torch.float).view(-1, 1)

# print(train_features[:3])
# print(test_features[:3])
# print(train_labels[:3])

loss = nn.BCEWithLogitsLoss()

x = train_data.type(torch.FloatTensor)  # cat是将两个张量，按维度0（行）进行拼接，指定为FloatTensor形式
print(x.shape) # [156, 26]
y = train_label.squeeze().type(torch.LongTensor)  # 标签一般一维，使用LongTensor形式
# print(y.shape) # [156]
x, y = Variable(x), Variable(y)  # 变成Variable的形式，神经网络只能输入Variable

x_test = test_data.type(torch.FloatTensor)  # cat是将两个张量，按维度0（行）进行拼接，指定为FloatTensor形式
print(x_test.shape) # [156, 26]
y_test = test_label.squeeze().type(torch.LongTensor)  # 标签一般一维，使用LongTensor形式
# print(y_test.shape) # [156]
x_test, y_test = Variable(x_test), Variable(y_test)  # 变成Variable的形式，神经网络只能输入Variable

"""创建神经网络"""
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):  # 初始化信息
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, 8)  # 一层隐藏层神经网络，输入信息，输出信息（神经元个数）
        # self.hidden2 = torch.nn.Linear(88, 128)
        # self.hidden3 = torch.nn.Linear(128, 32)
        # self.hidden4 = torch.nn.Linear(88, 8)
        self.out = torch.nn.Linear(8, n_output)  # 一层预测层神经网络，输入信息，输出信息（神经元个数）
    def forward(self, x):  # 前向传递，x为输入信息
        x = f.relu(self.hidden1(x))  # 出了隐藏层要激活
        # x = f.relu(self.hidden2(x))
        # x = f.relu(self.hidden3(x))
        # x = f.relu(self.hidden4(x))
        x = self.out(x)
        return x

"""打印处理数据"""
net = Net(38, 2)  # 输入值1个x，隐藏层10个，输出值一个y
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)  # 使用优化器优化神经网络参数，lr为学习效率，SGD为随机梯度下降法
loss_func = torch.nn.CrossEntropyLoss()  # 处理多分类问题

MAX_train = 0
MAX_val = 0
for t in range(100000):  # 循环训练
    out = net(x)  # 输入x，得到预测值
    loss = loss_func(out, y)  # 计算损失，预测值和真实值的对比
    # print(out, " #### ")
    # print(y, " #### ")
    # print(loss)
    optimizer.zero_grad()  # 梯度先全部降为0
    loss.backward()  # 反向传递过程
    optimizer.step()  # 以学习效率0.5来优化梯度
    """循环打印"""
    if t % 5 == 0:  # 每5步打印一次
        prediction = torch.max(out, 1)[1]  # troch.max(out,1)取每行的最大值，troch.max()[1]， 只返回最大值的每个索引
        pred_y = prediction.data.numpy()  # 预测分类标签转换成numpy格式
        target_y = y.data.numpy()  # 正确标签转换成numpy格式
        accuracy = sum(pred_y == target_y) / target_y.shape[0]
        MAX_train = max(accuracy, MAX_train)
        if MAX_train > 0.95:
            break

        out_test = net(x_test)
        prediction_test = torch.max(out_test, 1)[1]  # troch.max(out,1)取每行的最大值，troch.max()[1]， 只返回最大值的每个索引
        pred_y_test = prediction_test.data.numpy()  # 预测分类标签转换成numpy格式
        target_y_test = y_test.data.numpy()  # 正确标签转换成numpy格式
        accuracy_test = sum(pred_y_test == target_y_test) / x_test.shape[0]
        MAX_val = max(accuracy_test, MAX_val)
        # print("Ground_Truth : ", target_y_test[100:110])
        # print("Predictions  : ", pred_y_test[100:110])
        # print("Ground_Truth : ", target_y[100:110])
        # print("Predictions  : ", pred_y[100:110])

        print('epoch=%.2f' % t,'Train Accuracy=%.2f' % accuracy, 'Val Accuracy=%.2f' % accuracy_test)
print("训练集和验证集最高准确率为：", MAX_train, MAX_val)
