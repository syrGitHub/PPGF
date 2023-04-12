import random
from gensim.models import Word2Vec
import networkx
import pickle
import numpy as np
import datetime
import torch
# 从 start_node 开始随机游走
def deepwalk_walk(walk_length, start_node):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk

# 产生随机游走序列
def _simulate_walks(nodes, num_walks, walk_length):
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for v in nodes:
            walks.append(deepwalk_walk(walk_length=walk_length, start_node=v))
    return walks
with open('../../data/deephawkes_cmp_data/G_G_{}_timewindow{}.pkl'.format('train', 5), 'rb') as fw0:

    G_G_sp = pickle.load(fw0)
# 得到所有节点
G2={}
time=0
#
# for k,v in G_G_sp.items():
#
#     if time==2824:
#         b=v
#         break
#     time+=1
# nodes = list(b[1].nodes())
#
#         # 得到序列
# walks = _simulate_walks(nodes, num_walks=80, walk_length=10)
# w2v_model = Word2Vec(walks, sg=1, hs=1)


def feature2tensor(timewindow,typename):
    feature_dict = {}
    list1=[]
    list1_11=[]
    with open('../../data/deephawkes_cmp_data/G_G_deepwalk_index_{}_timewindow{}.pkl'.format(typename,timewindow), 'rb') as f13:
        a = pickle.load(f13)
    for k,v in a.items():
        for v2 in v.values():
            for v3 in v2.values():
                list1.append(v3)

            for su in range(100-len(list1)):
                list1.append(np.zeros(shape = [32], dtype=np.float32))
            list_tensor = torch.tensor(list1)
            list1 = []
            list_tensor.unsqueeze(1)
            list1_11.append(list_tensor)

        t1=torch.stack(list1_11)
        list1_11=[]

        feature_dict[k]=t1
    # np.save('../data/graphpool_trans_data/G_G_feature_tensor',feature_dict)
    with open('../../data/deephawkes_cmp_data/G_G_deepwalk_tensor_deepwalk{}_timewindow{}.pkl'.format(typename,timewindow), 'wb') as f1:
        pickle.dump(feature_dict, f1, pickle.HIGHEST_PROTOCOL)

with open('../../data/deephawkes_cmp_data/G_G_deepwalk_{}_timewindow{}.pkl'.format('test', 5), 'rb') as f13:
        a = pickle.load(f13)
        index = 0
        list1 = {}
        for v in a.values():
            list1[index] = v
            index += 1

with open('../../data/deephawkes_cmp_data/G_G_deepwalk_index_{}_timewindow{}.pkl'.format('test', 5), 'wb') as f12:
        pickle.dump(list1, f12, pickle.HIGHEST_PROTOCOL)
feature2tensor(5, 'test')
# for k,v in G_G_sp.items():
#     G3 = {}
#     for i in range(5):
#         G=v[i]
#
#         nodes = list(G.nodes())
#         if len(nodes)<=1:
#             G3.setdefault(i, {})[no] = np.zeros(shape = [32], dtype=np.float32)
#             continue
#         # 得到序列
#         walks = _simulate_walks(nodes, num_walks=80, walk_length=10)
#         w2v_model = Word2Vec(walks, sg=1, hs=1,vector_size=32,workers=8)
#         for no in nodes:
#             G3.setdefault(i,{})[no]=w2v_model.wv[no]
#     G2[k]=G3
#     print(time)
#     time+=1
# with open('../../data/deephawkes_cmp_data/G_G_deepwalk_{}_timewindow{}.pkl'.format('train', 5), 'wb') as f1:
#     pickle.dump( G2, f1, pickle.HIGHEST_PROTOCOL)
# # 默认嵌入到100维
# print(1)
# 打印其中一个节点的嵌入向量


