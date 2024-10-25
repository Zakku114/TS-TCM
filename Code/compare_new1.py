# 用于一些旧方法的对比
import xlrd
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from collections import defaultdict
import torch
import pickle
import random
xlrd.xlsx.ensure_elementtree_imported(False, None)
xlrd.xlsx.Element_has_iter = True
import time


# 读取原始数据集
from utils import load_raw_dataset,community_to_node,excel_read,sample_nodes,jc_vector,get_geneid_symbol_mapping,euclidean_distance,\
                  get_network,calculate_proximity,node_to_community
geneid_to_name, name_to_geneid = get_geneid_symbol_mapping("../Dataset/Homo_sapiens.gene_info")

def name_to_id(x,ids=False):
    X = []
    for i in range(len(x)):
        temp = []
        for j in range(len(x[i])):
            if ids==False:
                try:
                    temp.append(name_to_geneid[x[i][j]])
                except:
                    pass
            else:
                try:
                    temp.append(str(int(float(x[i][j]))))
                except ValueError:
                    pass 
        X.append(temp)
    return X

    

# 计算score
def mean_jcd(x,y,z):
    # 取样药方中的靶点
    x = sample_nodes(x,20)
    # 取样随机中药的靶点
    index = random.sample(range(0, len(y)-1), 8)
    tmp_y = set()
    for ind in index:
        for i in range(len(y[ind])):
            tmp_y.add(y[ind][i])
    y = list(tmp_y)
    y = np.random.choice(y,20,replace=True)
    # 重写这一块，使得取样相同数量的靶点
    
    tmp = z[0]
    z = tmp
    jc_drugs = jc_vector(y,cluster_result,community_result)
    jc_disease = jc_vector(z,cluster_result,community_result)
    # 随机组方到疾病的欧式距离
    d1 = euclidean_distance(jc_drugs,jc_disease)
    # 药方中取点到疾病的欧式距离
    t_d = []
    for i in range(len(x)):
        t_d.append([])
    for i in range(len(x)):
        jc_tmp = jc_vector(x[i],cluster_result,community_result)
        t_d[i] = euclidean_distance(jc_tmp,jc_disease)
    return t_d,d1
    

# 一些比较时用到的参数
import argparse
parser = argparse.ArgumentParser(description='some parameters')
parser.add_argument('--rand_seed', type=int, default=1234, help='random seed')
args = parser.parse_args()


with open('test_recipes1.txt', 'rb') as handle:
    test_x = pickle.load(handle)
handle.close()
with open('test_label1.txt', 'rb') as handle:
    test_y = pickle.load(handle)
handle.close()
with open('../Dataset/ids1.txt','r') as f:
    lines = f.readlines()
    nodes = []
    for line in lines:
        line = line.strip('\n')
        nodes.append(int(line))
print(test_y)


time_start = time.time()
# 控制总体流程(是否已经完成社区划分)
community_slashed = True
# 目前可选，gcn,none,kmeans,spectral
method='gcn'
if community_slashed==False:
    if method =='kmeans':
        # 1.使用kmeans获取社区
        network = '../Dataset/ppi_remove_self_loop.txt'
        A,ID2NODE = load_raw_dataset(network)
        svd = TruncatedSVD(n_components=3800, random_state=args.rand_seed)
        node_embed = svd.fit_transform(A)
        kmeans = KMeans(n_clusters = 768, random_state=args.rand_seed).fit(node_embed)
        cluster_result = defaultdict(list)
        for id in ID2NODE:
            cluster_result[str(ID2NODE[id])].append(kmeans.labels_[id])
        with open('../Output/kmeans_cluster_results1.txt', 'wb') as handle:
            pickle.dump(cluster_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
    
    if method=='spectral':
        # 2.使用谱聚类获取社区
        network = '../Dataset/ppi_remove_self_loop.txt'
        A,ID2NODE = load_raw_dataset(network)
        sc = SpectralClustering(n_clusters = 768,affinity='precomputed')
        sc.fit(A)
        cluster_result = defaultdict(list)
        for id in ID2NODE:
            cluster_result[str(ID2NODE[id])].append(sc.labels_[id])
        with open('../Output/spectral_cluster_results1.txt', 'wb') as handle:
            pickle.dump(cluster_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
    exit()
else:
    if method=='kmeans':
        with open('../Output/kmeans_cluster_results1.txt', 'rb') as handle:
            cluster_result=pickle.load(handle)
        handle.close()
    if method=='spectral':
        with open('../Output/spectral_cluster_results1.txt', 'rb') as handle:
            cluster_result=pickle.load(handle)
        handle.close()
        
if method=='gcn':
    # 直接读取社区划分的结果（需要事先进行社区划分）
    with open('../Output/community_cluster_results.txt', 'rb') as handle:
        cluster_result=pickle.load(handle)
    handle.close()
    


# 2.依据社区结果获取药物和疾病的社区距离
recipes_score = []
# 中药的数量
drug_num = 100
x = []
for i in range(len(test_x)):
    temp = []
    for j in range(len(test_x[i])):
        temp.append(str(nodes[test_x[i][j]]))
    x.append(temp)
for i in range(len(x)):
    recipes_score.append(0)
# 分别是药方的，药物的，(某种)疾病的路径
y = excel_read(drug_num,'../Dataset/StomacheDrugs.xlsx')
# cao,原来是Z标错了
z = excel_read(1,'../Dataset/StomacheDiseases.xlsx')

# 如果输入已经是节点id，设置x和z的ids=True
y = name_to_id(y,ids=False)
z = name_to_id(z,ids=False)



if method=='none':
    file_name = '../Dataset/protein interaction.sif'
    network = get_network(file_name, only_lcc=True)
    
    nodes_to_disease = z[0]
    # 使用原始方法进行距离计算
    for i in range(len(x)):
        nodes_from = x[i]
        d, score, (m, s) = calculate_proximity(network, nodes_from, nodes_to_disease, min_bin_size=100, seed=args.rand_seed)
        recipes_score[i]=score
        print(i,end='\t')
        print(score)
    exit()


# 获取社区结果或者靶点所属的社区
community_result = community_to_node(cluster_result)

epoch=1000
random_score=0
# 计算社区距离
for i in range(epoch):
    wz,j1 = mean_jcd(x,y,z)
    random_score += j1
    for i in range(len(x)):
        recipes_score[i] += wz[i]
# 结果展示
res = 0
for i in range(len(x)):
    if recipes_score[i]-random_score<0 and test_y[i]==1:
        res+=1
    if recipes_score[i]-random_score>0 and test_y[i]==0:
        res+=1
print(res/len(x))
time_end = time.time()
print(time_end - time_start)
# 相当于将模型反转
if res/len(x)-0.5<1e-6:
    for i in range(len(x)):
        print(-1*(recipes_score[i]-random_score))
    exit()
for i in range(len(x)):
    print(recipes_score[i]-random_score)