import pandas as pd
import numpy as np
import h5py
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
torch.set_default_tensor_type(torch.FloatTensor)
from torch.autograd import Variable
from pandas.core.frame import DataFrame
from networkx import karate_club_graph,to_numpy_matrix
import matplotlib.pyplot as plt
import networkx as nx
import torch as t
import torch.nn.functional as F
import scipy.sparse as sp
from dgl.nn.pytorch import GraphConv
import dgl

#########################################################
######################   数据加载  ######################
#########################################################

data_path = './数据集剔除抗鉴对照-北工大.xls'

# 处理性状数据
data =  pd.read_excel(data_path, sheet_name=0)
data_select = []
data_target = []
k = 0
for row in data.index.values:
    if data['评审状态'][row] == "淘汰":
        data_target.append(0)
    elif data['评审状态'][row] == "续试" or data['评审状态'][row] == "续试1" :
        data_target.append(1) 
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

    k += 1
    if k>1000:
        break

    data_select.append([feature_2, feature_3, feature_4, feature_5,  \
        feature_10, feature_11, feature_13, feature_14, feature_15, \
        feature_16, feature_17, feature_18, feature_19, feature_20, feature_21, feature_22, feature_23,  \
        feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30])


    # print(data_select)
    # 加上气候数据
    # data_select[row] += weather_dict[place]


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
train_labels = torch.tensor(data_target[:line], dtype=torch.float)
test_labels = torch.tensor(data_target[line:], dtype=torch.float)

# 图网络模型

def load_graph():
    path = 'graph.txt'
    data = np.loadtxt('graph.txt')
    n, _ = data.shape
    # print(data.shape) # (107495, 2)

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj).todense()

    return adj

topk = 3
def construct_graph(features, label, method='heat'):
    fname = 'graph.txt'
    num = len(label) # 要学习的标签数量
    dist = None

    # 计算特征之间的距离
    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        # 实现了对元素(可能是列表)的顺式访问
        dist = np.exp(dist) # e的dist次方
    elif method == 'cos': # 余弦距离
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos': # 正则余弦距离
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        # print(features.shape) # (30100, 24)
        dist = np.dot(features, features.T) # 乘法
        # print(dist.shape)     # (30100, 30100) 的对角阵

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):] # 排序函数，按位置条件从小到大"排序"
        inds.append(ind)
    # print("inds:", len(inds), inds.shape) # 30100

    f = open(fname, 'w')
    counter = 0
    A = np.zeros_like(dist)
    # print(A.shape) # (30100, 30100)
    # print("inds:", inds)
    src = []
    dst = []
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                if label[vv] != label[i]:
                    counter += 1
                src += [i]
                dst += [vv]
                f.write('{} {}\n'.format(i, vv))
    f.close()
    print('error rate: {}'.format(counter / (num * topk)))
    return src, dst

def main():
    # 截取数据
    train_data = train_features[500:550]
    train_label = train_labels[500:550]
    print("Train:",train_data.shape) # [50, 24]

    # 源节点与目标节点建图
    src, dst = construct_graph(train_data, train_label, 'cos')
    u = torch.tensor(np.concatenate([src, dst]))
    v = torch.tensor(np.concatenate([dst, src]))
    graph = dgl.graph((u, v))
    print(graph)

    # 输出adj
    # print(len(src), src)
    # print(len(dst), dst)
    # adj = load_graph()
    print(graph.nodes())
    # print(graph.edges()) # 打印2个tensor，一个源及诶单，一个目标节点

    # adj = graph.adj(scipy_fmt='coo')
    adj = graph.adjacency_matrix(transpose=True, scipy_fmt=None, etype=None)
    # order = sorted(list(graph.nodes()))
    print(adj)

    # A = to_numpy_matrix(graph).numpy()
    # print("\nA:",A.shape,'\n',A)

if __name__ == "__main__":
    main()

