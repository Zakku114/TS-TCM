# 稀疏矩阵相关
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack

# 系统函数
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 控制多线程执行
import multiprocessing
cpu_num = multiprocessing.cpu_count()

import pickle

# torch相关
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.nn as nn

# 数学库
import numpy as np
import random
import math
import statistics
import xlrd

# 函数
from Modules import *
from utils import get_geneid_symbol_mapping,get_union,get_intersection,get_adj,get_avg_length,get_link,read_str_recipes,read_str_drugs,\
                  get_hyperedge,get_over_HyperGraph,get_HyperGraph,generate_negative,train_batch_hyperedge,test_batch_hyperedge,train_epoch,eval_epoch,test_epoch,\
                  get_min_index,compute_k_nearest,count_cover,get_end_H,normalized,get_coverage,get_trainset,get_testset,evaluate


# 模型相关参数
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Run HyperGenerate model")
    parser.add_argument('--model_name',type=str,default='HyperGenerate')
    parser.add_argument('--input_dir', type=str, default='../Dataset')
    parser.add_argument('--save_path',type=str,default='../Output/checkpoints/HyperGenerate')
    parser.add_argument('--epoch',type=int,default=200)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--test_size',type=int,default=1)
    parser.add_argument('--dim',type=int,default=96)
    parser.add_argument('--neg_num',type=int,default=10)
    parser.add_argument('--pair_ratio',type=float,default=0.9)
    parser.add_argument('--minpts',type=float,default=12)
    parser.add_argument('--diag',type=str,default='True')
    parser.add_argument('--rw',type=float,default=0.0005)
    parser.add_argument('--alpha', type=float, default=0.0)
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args
args = parse_args()


# 生成节点的嵌入 
def generate_embeddings(H,nodes):
    H = np.array(H)
    HT = np.transpose(H)
    emb = np.dot(HT,H)
    idx = np.argwhere(np.all(emb[...,:]==0,axis=0))
    emb = np.delete(emb,idx,axis=1)
    emb=emb/np.max(emb,axis=0)
    return emb
    

# 训练超边筛选网络
def Train(args, model, loss, training_data,eval_data,optimizer, epochs, batch_size):
    eval_acu = -1
    for i in range(epochs):
        model.train()
        print('Epoch', i, 'of', epochs)
        bce_loss, recon_loss,acu = train_epoch(args, model, loss, training_data, optimizer, batch_size,neg_num=args.neg_num)
        print(bce_loss)
        print(recon_loss)
        print(acu)
        torch.cuda.empty_cache()
        acu = eval_epoch(args, model,eval_data,32,neg_num=args.neg_num)
        # 保存在验证集上最好的模型
        if  acu>eval_acu:
            eval_acu=acu
            torch.save(model.state_dict(), "model1.pth")
        print(acu)

# 校验超图结构
def test(args, model,testing_data,batch_size):
    model.load_state_dict(torch.load('model1.pth'))
    y = test_epoch(args,model,testing_data, batch_size)
    # 获取评分最高的若干条超边。
    a,b = testing_data
    H = get_end_H(a,y)
    return H   
        
    
# 控制总体流程
# 是否获取了节点的连接状态
linked= True
# 是否获取了供训练和筛选的超图结构
dataset_generated = True
# 是否已经训练好了超边筛选网络
trained = True
# 是否已经生成了超图
hyper_generated = True
# 组方评价网络是否已训练
net_trained = True


# 读取靶点ID
with open('../Dataset/ids1.txt','r') as f:
    lines = f.readlines()
    nodes = []
    for line in lines:
        line = line.strip('\n')
        nodes.append(int(line))


# 获取靶点连接状态
if not linked:
    # 读取靶点:社区下标的映射
    with open('../Output/cluster_results1.txt', 'rb') as handle:
        cluster_results=pickle.load(handle)
    handle.close()
    # 获取靶点的社区邻接矩阵
    adj = get_adj(cluster_results,nodes)
    # 建立高斯分布模型
    avg_length,u = get_avg_length(adj,10000)
    # 获取节点的直接相连邻接矩阵
    link = get_link(adj,avg_length-1.25*u)
    # 保存节点的邻接矩阵
    with open('../Output/BeforeH1.txt', 'wb') as handle:
        pickle.dump(link, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    exit()
else:
    # 读取节点的邻接矩阵
    with open('../Output/BeforeH1.txt', 'rb') as handle:
        link=pickle.load(handle)
    handle.close()

# 获取用于训练的数据集和用于筛选的超图结构
if not dataset_generated:
    # 读取节点的相似度矩阵
    with open('../Dataset/similarity1.txt','r') as f:
        lines = f.readlines()
        similarity = []
        for line in lines:
            line = line.strip("\n").split(' ')
            input_line=[]
            for i in range(len(line)):
                input_line.append(float(line[i]))
            similarity.append(input_line)
    # 计算k-近邻矩阵
    sim = compute_k_nearest(similarity,args.minpts)
    # 依据邻接矩阵和近邻矩阵获取超图结构
    over_H = get_over_HyperGraph(link,sim,min_pts=args.minpts)
    H = get_HyperGraph(link,sim,min_pts=args.minpts)
    # 存储over_H和H
    with open('../Output/OverH1.txt', 'wb') as handle:
        pickle.dump(over_H, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    with open('../Output/H1.txt', 'wb') as handle:
        pickle.dump(H, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    exit()
else:
    # 读取over_H和H
    with open('../Output/H1.txt', 'rb') as handle:
        H = pickle.load(handle)
    handle.close()
    with open('../Output/OverH1.txt', 'rb') as handle:
        over_H = pickle.load(handle)
    handle.close()
    
    
# 训练超边筛选网络
if not trained:
    # 节点特征嵌入相关
    embeddings_initial = generate_embeddings(over_H, nodes)
    node_embedding = MultipleEmbedding(embeddings_initial,args.dim,sparse=False).to(device)
    # 分类模型
    classifier_model = Classifier(n_head=8,d_model=args.dim,d_k=16,d_v=16,node_embedding=node_embedding,diag_mask=args.diag,bottle_neck=args.dim).to(device)
    # 打乱顺序，并划分训练集验证集
    over_H = np.array(over_H)
    p = np.random.permutation(range(len(over_H)))
    over_H = over_H[p]
    length_train = 7*len(over_H)//10
    train_H = over_H[:length_train]
    eval_H = over_H[length_train:]
    # 权重(可供后续扩展的)，目前全为1
    train_weight,eval_weight = np.ones(len(train_H), dtype='float32'),np.ones(len(eval_H), dtype='float32')
    # 损失函数和优化器
    loss = F.binary_cross_entropy
    params_list = list(set(list(classifier_model.parameters())))
    optimizer = torch.optim.Adam(params_list, lr=5e-5)
    # 训练
    Train(args, classifier_model,(loss,1.0),training_data=(train_H,train_weight),eval_data =(eval_H,eval_weight),optimizer=[optimizer], epochs=args.epoch, batch_size=args.batch_size)
    exit()
    

if not hyper_generated:
    # 获取超图结构
    embeddings_initial = generate_embeddings(H, nodes)
    H = np.array(H)
    test_weight = np.ones(len(H), dtype='float32')
    node_embedding = MultipleEmbedding(embeddings_initial,args.dim,sparse=False).to(device)
    print(len(H))
    classifier_model = Classifier(n_head=8,d_model=args.dim,d_k=16,d_v=16,node_embedding=node_embedding,diag_mask=args.diag,bottle_neck=args.dim).to(device)
    after_H = test(args,classifier_model,testing_data=(H,test_weight),batch_size=args.test_size)
    with open('../Output/afterH1.txt', 'wb') as handle:
        pickle.dump(after_H, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    print(len(after_H))
    H = after_H
else:
    # 读取超图结构
    with open('../Output/afterH1.txt', 'rb') as handle:
        H = pickle.load(handle)
    handle.close()


# 读取显著表达的靶点
file_name = '../Dataset/nodes1.txt'
f = open(file_name,'r')
lines = f.readlines()
nodes = list(map(lambda x: x.strip(), lines))
geneid_to_name, name_to_geneid = get_geneid_symbol_mapping("../Dataset/Homo_sapiens.gene_info")
for i in range(len(nodes)):
    nodes[i]=int(name_to_geneid[nodes[i]])

# 读取中药组方
recipes_dir =  '../Dataset/Stomache.xlsx'
recipes_id = read_str_recipes(recipes_dir,nodes,name_to_geneid)

# 读取背景中药数据集
drugs_dir = '../Dataset/StomacheDrugs.xlsx'
drugs_id = read_str_drugs(drugs_dir,nodes,name_to_geneid)
print(drugs_id)
      
# 获取超图结构(0,1表示->下标表示)
H = list(set([tuple(hyper) for hyper in H]))
hypergraph = []
for i in range(len(H)):
    hyper = []
    for j in range(len(H[i])):
        if H[i][j]==1:
            hyper.append(j)
    hypergraph.append(hyper)


# 组方评价网络结构
class jinyu_Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(jinyu_Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H, bias=True)
        self.linear2 = torch.nn.Linear(H, D_out, bias=False)
        #self.linear3 = torch.nn.Linear(H//10,D_out,bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        y_pred = self.linear2(self.relu(self.linear1(x)))
        y_pred = self.sigmoid(y_pred)
        return y_pred    


epoch = 1
best_score = -1
while not net_trained:
    for count in range(epoch):
        threshold = 0.5
        # 打乱数据集顺序
        p = np.random.permutation(range(len(recipes_id)))
        recipes_id = np.array(recipes_id)
        recipes_id = recipes_id[p]
        # 保证测试集样本平衡(大写是为了方便后续保存)
        Test_x,test_y =get_testset(recipes_id,drugs_id)
        test_x = normalized(Test_x,len(nodes))
        test_x = torch.tensor(get_coverage(test_x,hypergraph))
        # 训练集不需要样本均衡
        train_x,train_y = get_trainset(recipes_id,drugs_id)
        train_x = normalized(train_x,len(nodes))
        x = torch.tensor(get_coverage(train_x,hypergraph))
        y = torch.tensor(train_y,dtype=float).view(-1)
        
        # 模型相关参数
        D_in, H, D_out = len(hypergraph), 100, 1
        model = jinyu_Net(D_in, H, D_out)
        loss_fn = torch.nn.BCELoss()
        learning_rate = 4e-5
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 模型训练
        model.train()
        for t in range(30000):
            y_pred = model(x).view(-1) 
            loss = loss_fn(y_pred.float(), y.float())
            if t%1000==0:
                print(loss)
            if(loss<0.01):
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # 模型检验（保留在验证集上最好的一个模型）
        model.eval()
        with torch.no_grad():
            pred_y = model(test_x)
            yz_list = [0.1,0.01,1e-3,1e-4]
            score=evaluate(pred_y,test_y,yz_list)
            # 需要记住上面的阈值
            # 0.01,95.8%
            print(score)
            if(score>best_score):
                best_score = score
                with open('my_model1.pkl', 'wb') as f:
                    pickle.dump(model, f)
                f.close()
                # 获取用于检验结果的数据集
                with open('test_recipes1.txt','wb') as f:
                    pickle.dump(Test_x,f)
                f.close()
                with open('test_label1.txt','wb') as f:
                    pickle.dump(test_y,f)
                f.close()
        if count==epoch-1:
            net_trained = True
            exit()
        
        
# 对大批量中药组方进行筛选
with open('my_model1.pkl', 'rb') as f:
    model = pickle.load(f)
    
  
test_x = []   
test_y = []
res = []
for i in range(1000):
    recipe=set()
    marked = [0 for i in range(len(drugs_id))]
    # random 
    temp = set()
    drug_number = random.randint(4,6)
    for j in range(drug_number):
        drug_index = random.randint(0,len(drugs_id)-1)
        while marked[drug_index]==1:
            drug_index = random.randint(0,len(drugs_id)-1)
        marked[drug_index]=1
        for data in drugs_id[drug_index]:
            recipe.add(data)
        temp.add(drug_index)
    res.append(list(temp))
    test_x.append(list(recipe))
    test_y.append(0)
test_x,test_y = np.array(test_x),np.array(test_y)
test_x = normalized(test_x,len(nodes))
test_x = torch.tensor(get_coverage(test_x,hypergraph))
model.eval()
screened = []
score = []
with torch.no_grad():
    pred_y = model(test_x)
    for i in range(len(pred_y)):
        # 这里需要和上面的阈值对应:
        if pred_y[i]>0.01:
            score.append(pred_y[i])
            screened.append(res[i])
print(sum(score)/len(score))
f = open('../Output/screening1.txt', 'w')
for recipes in screened:
    f.write(str(recipes)+'\n')
f.close()
f = open('../Output/score1.txt', 'w')
for sc in score:
    f.write(str(sc)+'\n')
f.close()
