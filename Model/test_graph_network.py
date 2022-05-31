import torch
import torch.nn as nn
import numpy as np
import pandas as pd
torch.set_default_tensor_type(torch.FloatTensor)
from torch.autograd import Variable
from pandas.core.frame import DataFrame
from networkx import karate_club_graph,to_numpy_matrix
import matplotlib.pyplot as plt
import networkx as nx
import torch as t
import torch.nn.functional as F


#########################################################
######################   数据加载  ######################
#########################################################

data_path = './数据集剔除抗鉴对照-北工大.xls'

# 处理气候数据
# weather = pd.read_excel(data_path, sheet_name=1)
# weather_select = []

# 建立字典进行映射
# weather_dict = {}
# for row in weather.index.values:
#     place = weather['试验点'][row]

#     feature_1 = weather['平均温度'][row]-20 if not np.isnan(weather['平均温度'][row]) else 0
#     feature_2 = weather['最高温度'][row]-20 if not np.isnan(weather['最高温度'][row]) else 0
#     feature_3 = weather['最低温度'][row]-10 if not np.isnan(weather['最低温度'][row]) else 0
#     feature_4 = weather['降水量'][row] if not np.isnan(weather['降水量'][row]) else 0
#     feature_5 = weather['蒸发量'][row] if not np.isnan(weather['蒸发量'][row]) else 0
#     feature_6 = weather['平均风速'][row] if not np.isnan(weather['平均风速'][row]) else 0
#     feature_7 = weather['最大风速'][row] if not np.isnan(weather['最大风速'][row]) else 0
#     feature_8 = weather['平均地温'][row]-20 if not np.isnan(weather['平均地温'][row]) else 0
#     feature_9 = weather['最大地温'][row]-30 if not np.isnan(weather['最大地温'][row]) else 0
#     feature_10 = weather['最小地温'][row]-10 if not np.isnan(weather['最小地温'][row]) else 0
#     feature_11 = weather['日照时数'][row] if not np.isnan(weather['日照时数'][row]) else 0

#     # log平滑处理
#     feature_1 = np.log(feature_1+1) if feature_1>=0 else -np.log(abs(feature_1)+1)
#     feature_2 = np.log(feature_2+1) if feature_2>=0 else -np.log(abs(feature_2)+1)
#     feature_3 = np.log(feature_3+1) if feature_3>=0 else -np.log(abs(feature_3)+1)
#     feature_4 = np.log(feature_4+1) if feature_4>=0 else -np.log(abs(feature_4)+1)
#     feature_5 = np.log(feature_5+1) if feature_5>=0 else -np.log(abs(feature_5)+1)
#     feature_6 = np.log(feature_6+1) if feature_6>=0 else -np.log(abs(feature_6)+1)
#     feature_7 = np.log(feature_7+1) if feature_7>=0 else -np.log(abs(feature_7)+1)
#     feature_8 = np.log(feature_8+1) if feature_8>=0 else -np.log(abs(feature_8)+1)
#     feature_9 = np.log(feature_9+1) if feature_9>=0 else -np.log(abs(feature_9)+1)
#     feature_10 = np.log(feature_10+1) if feature_10>=0 else -np.log(abs(feature_10)+1)
#     feature_11 = np.log(feature_11+1) if feature_11>=0 else -np.log(abs(feature_11)+1)

#     weather_feature = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6,\
#         feature_7, feature_8, feature_9, feature_10, feature_11]
#     weather_dict[place] = weather_feature
#     # print(weather_dict)
    # break
    
# 处理性状数据
data =  pd.read_excel(data_path, sheet_name=0)
data_select = []
data_target = []
a, b = 0, 0
for row in data.index.values:
    if data['评审状态'][row] == "淘汰":
        data_target.append(0)
        a += 1
    elif data['评审状态'][row] == "续试" or data['评审状态'][row] == "续试1" :
        data_target.append(1) 
        b += 1
    else:
        continue
        # print("Lable Error %d", row)
        # break
    # place = data['试验点'][row]

    # feature_1 = data['褐斑病(级)'][row] if not np.isnan(data['褐斑病(级)'][row]) else 0
    feature_2 = data['倒伏率(%)'][row] if not np.isnan(data['倒伏率(%)'][row]) else 0
    feature_3 = data['倒折率(%)'][row] if not np.isnan(data['倒折率(%)'][row]) else 0
    feature_4 = data['大斑病(级)'][row] if not np.isnan(data['大斑病(级)'][row]) else 0
    feature_5 = data['茎腐病(%)'][row] if not np.isnan(data['茎腐病(%)'][row]) else 0
    # feature_6 = data['黑粉病(%)'][row] if not np.isnan(data['黑粉病(%)'][row]) else 0
    # feature_7 = data['弯孢叶斑病'][row] if not np.isnan(data['弯孢叶斑病'][row]) else 0
    # feature_8 = data['小斑病'][row] if not np.isnan(data['小斑病'][row]) else 0
    # feature_9 = data['粗缩病(%)'][row] if not np.isnan(data['粗缩病(%)'][row]) else 0
    feature_10 = data['灰斑病(级)'][row] if not np.isnan(data['灰斑病(级)'][row]) else 0
    feature_11 = data['空秆率(%)'][row] if not np.isnan(data['空秆率(%)'][row]) else 0
    # feature_12 = data['比对照长短天数(天)'][row] if not np.isnan(data['比对照长短天数(天)'][row]) else 0
    feature_13 = data['穗腐病(%)'][row] if not np.isnan(data['穗腐病(%)'][row]) else 0
    feature_14 = data['亩产(kg)'][row] if not np.isnan(data['亩产(kg)'][row]) else 0
    feature_15 = data['比对照增减产(%)'][row] if not np.isnan(data['比对照增减产(%)'][row]) else 0
    feature_16 = data['丝黑穗病(%)'][row] if not np.isnan(data['丝黑穗病(%)'][row]) else 0
    feature_17 = data['株高(cm)'][row] if not np.isnan(data['株高(cm)'][row]) else 0
    feature_18 = data['穗位高(cm)'][row] if not np.isnan(data['穗位高(cm)'][row]) else 0
    feature_19 = data['株高(cm)'][row] if not np.isnan(data['株高(cm)'][row]) else 0
    feature_20 = data['生育期(天)'][row] if not np.isnan(data['生育期(天)'][row]) else 0
    feature_21 = data['百粒重(g)'][row] if not np.isnan(data['百粒重(g)'][row]) else 0
    feature_22 = data['穗长(cm)'][row] if not np.isnan(data['穗长(cm)'][row]) else 0
    feature_23 = data['秃尖长(cm)'][row] if not np.isnan(data['秃尖长(cm)'][row]) else 0
    feature_24 = data['出籽率(%)'][row] if not np.isnan(data['出籽率(%)'][row]) else 0
    feature_25 = data['收获时籽粒含水量平均(%)'][row] if not np.isnan(data['收获时籽粒含水量平均(%)'][row]) else 0
    feature_26 = data['行粒数(粒)'][row] if not np.isnan(data['行粒数(粒)'][row]) else 0
    feature_27 = data['穗粗(cm)'][row] if not np.isnan(data['穗粗(cm)'][row]) else 0
    feature_28 = data['轴粗(cm)'][row] if not np.isnan(data['轴粗(cm)'][row]) else 0
    feature_29 = data['全生育期叶数(片)'][row] if not np.isnan(data['全生育期叶数(片)'][row]) else 0
    feature_30 = data['果穗鲜重(kg)'][row] if not np.isnan(data['果穗鲜重(kg)'][row]) else 0


    # log 平滑处理 ： 长尾分布 --> 正态分布
    # feature_1 = np.log(feature_1+1) if feature_1>=0 else -np.log(abs(feature_1)+1)
    feature_2 = np.log(feature_2+1) if feature_2>=0 else -np.log(abs(feature_2)+1)
    feature_3 = np.log(feature_3+1) if feature_3>=0 else -np.log(abs(feature_3)+1)
    feature_4 = np.log(feature_4+1) if feature_4>=0 else -np.log(abs(feature_4)+1)
    feature_5 = np.log(feature_5+1) if feature_5>=0 else -np.log(abs(feature_5)+1)
    # feature_6 = np.log(feature_6+1) if feature_6>=0 else -np.log(abs(feature_6)+1)
    # feature_7 = np.log(feature_7+1) if feature_7>=0 else -np.log(abs(feature_7)+1)
    # feature_8 = np.log(feature_8+1) if feature_8>=0 else -np.log(abs(feature_8)+1)
    # feature_9 = np.log(feature_9+1) if feature_9>=0 else -np.log(abs(feature_9)+1)
    feature_10 = np.log(feature_10+1) if feature_10>=0 else -np.log(abs(feature_10)+1)
    feature_11 = np.log(feature_11+1) if feature_11>=0 else -np.log(abs(feature_11)+1)
    # feature_12 = np.log(feature_12+1) if feature_12>=0 else -np.log(abs(feature_12)+1)
    feature_13 = np.log(feature_13+1) if feature_13>=0 else -np.log(abs(feature_13)+1)
    feature_14 = np.log(feature_14+1) if feature_14>=0 else -np.log(abs(feature_14)+1)
    feature_15 = np.log(feature_15+1) if feature_15>=0 else -np.log(abs(feature_15)+1)
    feature_16 = np.log(feature_16+1) if feature_16>=0 else -np.log(abs(feature_16)+1)
    feature_17 = np.log(feature_17+1) if feature_17>=0 else -np.log(abs(feature_17)+1)
    feature_18 = np.log(feature_18+1) if feature_18>=0 else -np.log(abs(feature_18)+1)
    feature_19 = np.log(feature_19+1) if feature_19>=0 else -np.log(abs(feature_19)+1)
    feature_20 = np.log(feature_20+1) if feature_20>=0 else -np.log(abs(feature_20)+1)
    feature_21 = np.log(feature_21+1) if feature_21>=0 else -np.log(abs(feature_21)+1)
    feature_22 = np.log(feature_22+1) if feature_22>=0 else -np.log(abs(feature_22)+1)
    feature_23 = np.log(feature_23+1) if feature_23>=0 else -np.log(abs(feature_23)+1)
    feature_24 = np.log(feature_24+1) if feature_24>=0 else -np.log(abs(feature_24)+1)
    feature_25 = np.log(feature_25+1) if feature_25>=0 else -np.log(abs(feature_25)+1)
    feature_26 = np.log(feature_26+1) if feature_26>=0 else -np.log(abs(feature_26)+1)
    feature_27 = np.log(feature_27+1) if feature_27>=0 else -np.log(abs(feature_27)+1)
    feature_28 = np.log(feature_28+1) if feature_28>=0 else -np.log(abs(feature_28)+1)
    feature_29 = np.log(feature_29+1) if feature_29>=0 else -np.log(abs(feature_29)+1)
    feature_30 = np.log(feature_30+1) if feature_30>=0 else -np.log(abs(feature_30)+1)

    data_select.append([feature_2, feature_3, feature_4, feature_5,  \
        feature_10, feature_11, feature_13, feature_14, feature_15, \
        feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23,  \
        feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30])

    # print(data_select)
    # 加上气候数据
    # data_select[row] += weather_dict[place]
    
    # print(data_select)
    # break

#########################################################
######################   数据处理  ######################
#########################################################
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
print(all_features.shape) # (198, 15)

line = len(data_select)//5 * 4
# print(line) # 156

# train_data = data_select[:line]
# test_data = data_select[line:]

# n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:line].values, dtype=torch.float)
test_features = torch.tensor(all_features[line:].values, dtype=torch.float)
train_labels = torch.tensor(data_target[:line], dtype=torch.float).view(-1, 1)
test_labels = torch.tensor(data_target[line:], dtype=torch.float).view(-1, 1)

# print(train_features[:3])
# print(test_features[:3])
# print(train_labels[:3])

# loss = nn.BCEWithLogitsLoss()

x = train_features.type(torch.FloatTensor)  # cat是将两个张量，按维度0（行）进行拼接，指定为FloatTensor形式
print(x.shape) # [156, 26]
y = train_labels.squeeze().type(torch.LongTensor)  # 标签一般一维，使用LongTensor形式
# print(y.shape) # [156]
x, y = Variable(x), Variable(y)  # 变成Variable的形式，神经网络只能输入Variable

x_test = test_features.type(torch.FloatTensor)  # cat是将两个张量，按维度0（行）进行拼接，指定为FloatTensor形式
# print(x_test.shape) # [156, 26]
y_test = test_labels.squeeze().type(torch.LongTensor)  # 标签一般一维，使用LongTensor形式
# print(y_test.shape) # [156]
x_test, y_test = Variable(x_test), Variable(y_test)  # 变成Variable的形式，神经网络只能输入Variable


#########################################################
######################  图网络训练 ######################
#########################################################

# club_name=['Officer','Mr.Hi'] 
# zkc = karate_club_graph()     # 获取俱乐部关系图，2分类['Officer','Mr.Hi']对应[0,1]
# all_labels=[]
# for i in range(34):
#     # print("节点%d:"%i,zkc.nodes[i])
#     if zkc.nodes[i]['club'] == "Officer":
#         all_labels.append(0)
#     else:
#         all_labels.append(1)
# print("\n一共%d个节点的类别:"%(len(all_labels)),all_labels)
# print("\n节点的连接关系,一共%d条边:"%len(zkc.edges),zkc.edges)  

'''
求邻接矩阵A和度矩阵D, 提前设置（先验知识）
'''
# print(zkc.nodes())
# order = sorted(list(zkc.nodes()))
# print(order)

A = to_numpy_matrix(zkc, nodelist=order)
print("\nA:",A.shape,'\n',A)
# 34名成员之间的社会关系, 1表示有交往，0表示没有。所以是34 X 34

I = np.eye(zkc.number_of_nodes()) # 生成对角矩阵，用以增加自环，即聚合特征时包括自己的特征
A_hat = A + I

# 度矩阵：每一行的权重除以该节点的度，使得每行标准化(相加为0)
D_hat = np.array(np.sum(A_hat, axis=0))[0]
# np.sum(A_hat, axis=0)将A_hat中每一列的元素相加,将矩阵压缩为一行

# print("D_hat:",D_hat.shape) # 此时为一维矩阵
D_hat = np.matrix(np.diag(D_hat)) # 输出以上面的一维矩阵为对角线的多维矩阵
# print("D_hat:",D_hat)

class GCNLayer(nn.Module):
    """
    Define the GCNLayer module.
    nn.Linear相当于乘上W_1\W_2
    """
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)#全连接层（in_feats输入二维张量大小, out_feats输出张量）

    def forward(self, g, inputs):
        #print("inputs:",inputs.dtype,inputs.shape)
        #print((D_hat**-1*A_hat).dtype,t.Tensor(D_hat**-1*A_hat).shape)#float64 torch.Size([34, 34])
        h = t.mm(t.Tensor(D_hat**-1*A_hat),inputs) # mm相乘
        #print(self.linear(h).shape)
        return self.linear(h)

class GCN(nn.Module):
    """
    Define a 2-layer GCN model.
    in_feats: 34
    hidden_size: 4
    num_classes: 2
    """
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = t.relu(h)
        h = self.gcn2(g, h)
        return h

    
def main():
    inputs = t.eye(34) # 对角矩阵, 对应特征维度, 自闭环
    '''选取4个节点做loss计算'''
    labels = t.tensor([1,1,0,0]) # 0表示officer类，1表示Mr.Hi类
    labeled_nodes = t.tensor([0, 8 , 9, 33]) # 2个类别对应的索引
    
    t.manual_seed(0) #固定初始化参数
    net=GCN(34,4,2)
    # print("\nGCN结构:\n",net)
    optimizer = t.optim.Adam(net.parameters(), lr=0.005)
    all_logits = []
    
    fig = plt.figure(figsize=[10,8])    
    plt.ion()
    
    for epoch in range(1):
        outputs = net(zkc, inputs) #模型输出     
        # print(inputs) 
        print(zkc)
        # plot_graph(fig,outputs,epoch)        
        all_logits.append(outputs.detach())
        #print("outputs.shape:",outputs.shape,outputs)
        logp = F.log_softmax(outputs, 1) #dim=1,对每一行进行softmax,当dim=0为按列
        #print("after_log_softmax:",logp)
        
        preds=t.argmax(F.softmax(outputs, 1),dim=1)
        # print(all_labels)
        # print(preds)
        correct_nodes=t.eq(preds,t.Tensor(all_labels)).float().sum().item()
        # 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False。
        precision = correct_nodes/len(preds)
        #print(preds,correct_nodes)
        
        # compute loss for labeled nodes
        #print('\n',logp[labeled_nodes])
        loss = F.nll_loss(logp[labeled_nodes], labels)
        
        # compute gradient and do Adam step
        '''
        梯度归零 optimizer.zero_grad()
        反向传播计算得到每个参数的梯度值 loss.backward()
        通过梯度下降执行一步参数更新 optimizer.step()
        '''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('\nEpoch %d | Loss: %.4f' % (epoch, loss.item()))        
        print('>>Fault: %d | Correct: %d | Precision: %f' % (len(preds)-correct_nodes,correct_nodes, precision))
    
    plt.ioff()
    #plt.close()


if __name__ == "__main__":
    main()

