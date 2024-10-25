import numpy as np
import torch
import math
import torch.nn.functional as F
from optimizer import NAdam
import stopping
from GO import GO
import networkx as nx
import scipy.sparse as sp
import pickle
from func_associate import check_functional_enrichment
from sampler import get_edge_sampler
from model import PolyGCN
from loss import BerPossionLoss,l2_reg_loss
from utils import get_geneid_symbol_mapping,get_str_nodes,get_union,get_intersection,get_enrichment,save_key_nodes,save_sub_network,\
                  feature_generator,to_sparse_tensor,adj_polynomials,cluster_infer,community_to_node


import argparse
parser = argparse.ArgumentParser(description='PyTorch Implementation of Node community division')
# 数据集存储路径
parser.add_argument('--inputdir', type=str, default="../Dataset/", help='input file')
# 靶点按照度数切片的参数
parser.add_argument('--bin_num', type=int, default=20, help='node split by degree')
# 特征生成方式，目前仅支持svd降维
parser.add_argument('--preprocess', type=str, default='svd', help='feature preprocessing: svd')
# svd降为多少维
parser.add_argument('--n_components', type=int, default=100, help='n_components for SVD')
# 邻接矩阵最大阶数
parser.add_argument('--adjpow', type=int, default=2, help='element-wise power of adjacency matrix')
# 社区划分网络的dropout概率
parser.add_argument('--dropout', type=float, default=0, help='dropout')
# 用于构建数据集的处理器线程数
parser.add_argument('--num_workers', type=int, default=6, help='less than cpu cores')
# batchsize of gcn network
# must smaller than len(A)
parser.add_argument('--batch_size', type=int, default=100, help='batchsize')
# 模型&结果存储路径
parser.add_argument('--outputdir', type=str, default="../Output/", help='output file')
# 网络隐藏层尺寸
parser.add_argument('--hidden_size', default=[1024, 512], nargs='+', type=int, help='hidden size of gnn')
# 社区数量
parser.add_argument('--K', type=int, default=200, help='cluster_number')
# 是否对输入数据进行串标准化
parser.add_argument('--batch_norm', type=bool, default=True, help='whether to perform batch normalization')
# 随机数种子
parser.add_argument('--rand_seed', type=int, default=114514, help='random seed')
# 学习率相关
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_min', type=float, default=1e-5, help='min learning rate')
parser.add_argument('--lr_decay', type=float, default=0.9, help='decrease coefficient of learning rate')
# 早停超参数
parser.add_argument('--patience', type=int, default=2, help='patience for early stopping')
# 网络迭代次数
parser.add_argument('--epochs', type=int, default=1000, help='epoch')
# 每训练多少次验证
parser.add_argument('--val_step', type=int, default=50, help='validation step to evaluate loss')
#抑制过拟合的超参数
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
args = parser.parse_args()


# 靶点ID/名称映射文件
map_file = '../Dataset/Homo_sapiens.gene_info'
# 组方/疾病靶点文件
sheetname = '../Dataset/Stomache.xlsx'
# 保存富集功能的文件
function_file = "../Dataset/functions1.txt"
# 基因本体论相关数据集
go_fname='../Dataset/go.obo'
go_goa_fname='../Dataset/goa_human.gaf'
# 保存关键靶点的名称和ID
nodes_file = '../Dataset/nodes1.txt'
ids_file = '../Dataset/ids1.txt'
# 保存相似度矩阵和相互作用矩阵
similarity_file = '../Dataset/similarity1.txt'
interaction_file = '../Dataset/interaction1.txt'


def load_dataset(directory,bin_num):
    f1 = open(directory+'ids1.txt')
    f2 = open(directory+'interaction1.txt')
    ids = []
    line = f1.readline()
    while line:
        ids.append(line.strip('\n'))
        line = f1.readline()
    f1.close
    line = f2.readline()
    interaction = []
    while line:
        interaction.append(list(map(int,line.strip('\n').split(' '))))
        line = f2.readline()
    f2.close()
    net_nA = []
    net_nB = []
    for i in range(len(ids)):
        for j in range(len(ids)):
            if interaction[i][j]!=0:
                net_nA.append(ids[i])
                net_nB.append(ids[j])
    G = nx.Graph()  
    G.add_edges_from(list(zip(net_nA, net_nB)))
    G.remove_edges_from(nx.selfloop_edges(G))
    net_node = list(G)
    net_node = sorted(net_node)
    ID2NODE = dict()
    NODE2ID = dict()
    for idx in range(len(net_node)):
        ID2NODE[idx] = net_node[idx]
        NODE2ID[net_node[idx]] = idx
    row_idx = []
    col_idx = []
    val_idx = []  
    for n_a, n_b in zip(net_nA, net_nB):
        if n_a in net_node and n_b in net_node:
            row_idx.append(NODE2ID[n_a])
            col_idx.append(NODE2ID[n_b])
            val_idx.append(1.0)
            row_idx.append(NODE2ID[n_b])
            col_idx.append(NODE2ID[n_a])
            val_idx.append(1.0)
    A = sp.csr_matrix((np.array(val_idx), (np.array(row_idx), np.array(col_idx))), shape=(len(net_node), len(net_node)))
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    G_lcc = nx.Graph()
    G_lcc.add_edges_from(list(zip(row_idx, col_idx)))
    node_degree = {i: G_lcc.degree[i] for i in G_lcc.nodes}
    sorted_node_degree = dict(sorted(node_degree.items(), key=lambda item: item[1]))
    node2bin = np.array_split(list(sorted_node_degree.keys()), bin_num)
    return A,ID2NODE,node2bin


# 控制代码总体流程
# 是否已经完成功能富集
Enrichmented = True
# 是否完成数据预处理
PreTreatment = True
# 是否已经完成社区划分
CommunitySlash = False


# 获取靶点ID和靶点名称的映射
geneid_to_name, name_to_geneid = get_geneid_symbol_mapping(map_file)
# 读取该疾病涉及的全部靶点
names,nodes = get_str_nodes(sheetname,geneid_to_name,name_to_geneid)

# 功能富集
if not Enrichmented:
    check_functional_enrichment(names, None, "genesymbol", open(function_file, 'w').write, species="Homo sapiens")
    exit()

# 数据预处理
if not PreTreatment:
    # 读取显著表达的功能
    functions = get_enrichment(function_file)
    # 读取基因本体论数据集
    go = GO(go_fname, False, go_goa_fname)
    dict_nodes = {}
    # 将靶点本身的功能和功能富集到的功能交叉
    for i in range(len(nodes)):
        name = geneid_to_name[nodes[i]]
        dict_nodes.setdefault(name, get_intersection(go.get_go_terms_of_gene(name),functions))
    # 存储关键靶点（名称和ID）
    props,nodes = save_key_nodes(dict_nodes,name_to_geneid,nodes_file,ids_file)
    # 存储靶点网络（相似度和相互作用）
    similarity,interaction = save_sub_network(props,nodes,similarity_file,interaction_file)
    exit()


# 社区划分
if not CommunitySlash:
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    # 读取节点的邻接矩阵,靶点下标和ID的映射，靶点下标及其所属的切片
    A, ID2NODE,node2bin = load_dataset(args.inputdir,args.bin_num)
    # 靶点特征生成
    feat = feature_generator(A, n_comp=args.n_components, preprocess=args.preprocess)
    # 转换为稀疏矩阵
    feat = to_sparse_tensor(feat, device)
    # 获取高阶邻接矩阵
    adj_polys = adj_polynomials(A, args.adjpow, sparse=True)
    adj_polys = [to_sparse_tensor(p, device) for p in adj_polys]
    
    # 网络相关变量：drouput概率，损失函数，学习率，早停
    p_dropout = args.dropout if args.preprocess == "svd" else 0.0
    N = A.shape[0]
    criterion = BerPossionLoss(N, A.nnz)
    lr = args.lr
    val_loss = np.inf
    validation_fn = lambda: val_loss
    early_stopping = stopping.NoImprovementStopping(validation_fn, patience=args.patience)
    
    # 定义网络和优化器
    gnn = PolyGCN(input_dim=feat.shape[1], hidden_dims=args.hidden_size, output_dim=args.K, batch_norm=args.batch_norm, \
                                    adj_pow=(args.adjpow + 1), p_dropout=p_dropout,rand_seed=args.rand_seed).to(device)
    opt = NAdam(gnn.parameters(), lr=args.lr)
    
    # 记录模型及模型中间结果
    model_out = args.outputdir +"CommunityModel1" + ".pth"
    model_saver = stopping.ModelSaver(gnn, opt, model_out, device)
    f_out = open(args.outputdir + "ProcessRecords" + ".txt", "a")
    
    # 采样正边、负边、子靶点集合(数据集构建)
    sampler = get_edge_sampler(A, node2bin, args.bin_num, batchsize=args.batch_size, num_workers=args.num_workers)
    
    # 模型训练
    for epoch, batch in enumerate(sampler):
        # 调整学习率
        if (epoch + 1) % args.val_step == 0:
            lr = max(lr * args.lr_decay, args.lr_min)
            opt = NAdam(gnn.parameters(), lr=lr)
        
        # 取一个batch的数据 
        ones_idx, zeros_idx, batch_nodes = batch
        
        # 使用batch中包含的样本训练
        gnn.train()
        opt.zero_grad()
        A_batch = A[batch_nodes][:, batch_nodes]
        Z = F.relu(gnn(feat, adj_polys))
        Z_batch = Z[batch_nodes]
        criterion_batch = BerPossionLoss(len(batch_nodes), A_batch.nnz)
        loss = criterion_batch.loss_batch(Z_batch, ones_idx, zeros_idx)
        loss += l2_reg_loss(gnn, scale=args.weight_decay)
        loss.backward()
        opt.step()
        
        
        # 验证阶段
        if epoch % args.val_step == 0:
            with torch.no_grad():
                gnn.eval()
                # 在整个图上的Loss
                pos_full, neg_full, full_loss = criterion.loss_cpu(Z.cpu().detach().numpy(), A)
                # 在训练集上的Loss
                pos_batch, neg_batch, batch_loss = criterion_batch.loss_cpu(Z_batch.cpu().detach().numpy(), A_batch)
                
                # 计算在验证集上的Loss
                # 验证集上的正Loss
                pos_val = (pos_full * criterion.num_edges - pos_batch * criterion_batch.num_edges) / (criterion.num_edges - criterion_batch.num_edges)
                # 验证集上的负Loss
                neg_val = (neg_full * criterion.num_nonedges - neg_batch * criterion_batch.num_nonedges) / (criterion.num_nonedges - criterion_batch.num_nonedges)
                # 负样本/正样本概率
                val_ratio = (criterion.num_nonedges - criterion_batch.num_nonedges) / (criterion.num_edges - criterion_batch.num_edges)
                # 验证集上的损失函数
                val_loss = (pos_val + val_ratio * neg_val) / (1 + val_ratio)
                print(pos_val)
                print(neg_val)
                print('*' * 100)
                print(batch_loss)
                print(val_loss)
                f_out.write(f'Epoch {epoch:4d}, loss.train = {batch_loss:.4f}, loss.val = {val_loss:.4f}, loss.full = {full_loss:.4f}')
                f_out.write('\n')
                # Check if it's time for early stopping
                early_stopping.next_step()
                if early_stopping.should_save():
                    model_saver.save()
                if early_stopping.should_stop():
                    print(f'Breaking due to early stopping at epoch {epoch}')
                    break
                
        # 检查方法是否停止
        if (epoch + 1) > args.epochs:
            break
    f_out.close()
    
    
    # 获取节点的表示
    gnn.eval()
    Z = F.relu(gnn(feat, adj_polys))
    Z_cpu = Z.cpu().detach().numpy() if torch.cuda.is_available() else Z.detach().numpy()
    thresh = math.sqrt(-math.log(1 - 1 / N))
    Z_pred = Z_cpu > thresh
    # 进行社区划分，结果是靶点：靶点所在社区的字典
    clust_results = cluster_infer(Z_pred, ID2NODE)
    # 获取社区结果，结果是社区：社区上的节点的字典
    community_results = community_to_node(clust_results)
    
    # 结果保存
    with open(args.outputdir + 'cluster_results1.txt', 'wb') as handle:
        pickle.dump(clust_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    with open(args.outputdir + 'community_results1.txt', 'wb') as handle:
        pickle.dump(community_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    