import xlrd
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
import torch
import random
import network_utilities
import os
from tqdm import tqdm, trange
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from concurrent.futures import as_completed, ProcessPoolExecutor
import math
import statistics
import pickle



def get_geneid_symbol_mapping(file_name):
    """
    To parse Homo_sapiens.gene_info (trimmed to two colums) file from NCBI
    Creating the file
    wget ftp://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz
    zcat Homo_sapiens.gene_info.gz | cut -f 2,3,5 > geneid_to_symbol.txt
    Remove first line if need be (but keep header)
    """
    f = open(file_name)
    f.readline()
    geneid_to_name = {} # now contains only the official symbol
    name_to_geneid = {}
    for line in f:
        words = line.strip("\n").split("\t")
        geneid, symbol = words
        geneid = geneid.strip() # strip in case mal formatted input file
        symbol = symbol.strip()
        if geneid == "" or symbol == "":
            continue
        geneid_to_name[geneid] = symbol
        for symbol in [symbol]: # added for synonym parsing
            if symbol in name_to_geneid:
                if int(geneid) >= int(name_to_geneid[symbol]):
                    continue
                print ("Multiple geneids", name_to_geneid[symbol], geneid, symbol)
            name_to_geneid[symbol] = geneid
    f.close()
    return geneid_to_name, name_to_geneid
    

def get_nodes(sheetname,id_to_name,name_to_id):
    worksheet = xlrd.open_workbook(sheetname)
    sheet_names= worksheet.sheet_names()
    recipes = []
    for sheet_name in sheet_names:
        sheet = worksheet.sheet_by_name(sheet_name)
        rows = sheet.nrows
        cols = sheet.ncols
        recipe = set()
        for i in range(1,rows):
            cell = sheet.cell_value(i, 0)
            try:
                cell = str(int(cell))
                recipe.add(cell)
            except ValueError:
                pass
        for i in range(1,rows):
            try:
                cell = sheet.cell_value(i, 1)
                try:
                    cell = str(int(cell))
                    recipe.add(cell)
                except ValueError:
                    pass
            except IndexError:
                pass
        recipes.append(recipe)
    nodes = set()
    for recipe in recipes:
        for geneid in recipe:
            name = id_to_name[geneid]
            nodes.add(name)
    names = list(nodes)
    nodes = []
    for i in range(len(names)):
        nodes.append(name_to_id[names[i]])
    return names,nodes
    
def get_str_nodes(sheetname,id_to_name,name_to_id):
    worksheet = xlrd.open_workbook(sheetname)
    sheet_names= worksheet.sheet_names()
    recipes = []
    for sheet_name in sheet_names:
        sheet = worksheet.sheet_by_name(sheet_name)
        rows = sheet.nrows
        cols = sheet.ncols
        recipe = set()
        for i in range(rows):
            cell = sheet.cell_value(i, 0)
            try:
                cell = str(cell)
                recipe.add(cell)
            except ValueError:
                pass
        for i in range(rows):
            try:
                cell = sheet.cell_value(i, 1)
                try:
                    cell = str(cell)
                    recipe.add(cell)
                except ValueError:
                    pass
            except IndexError:
                pass
        recipes.append(recipe)
    nodes = set()
    for recipe in recipes:
        for genename in recipe:
            try:
                tmpid = name_to_id[genename]
                nodes.add(tmpid)
            except KeyError:
                continue
    nodes = list(nodes)
    names = []
    for i in range(len(nodes)):
        names.append(id_to_name[nodes[i]])
    return names,nodes


def save_key_nodes(dict_nodes,name_to_geneid,nodes_dir,ids_dir):
    props = {}
    nodes = []
    ids = []
    for key in dict_nodes:
        if len(dict_nodes[key])!=0:
            props.setdefault(key,dict_nodes[key])
            nodes.append(key)
            ids.append(name_to_geneid[key])
    f1 = open(nodes_dir,'w')
    for i in range(len(props)):
        f1.write(nodes[i]+'\n')
    f1.close()
    f2 = open(ids_dir,'w')
    for i in range(len(props)):
        f2.write(ids[i]+'\n')
    f2.close()                                                        
    return props,nodes
    
    
def save_sub_network(props,nodes,sim_dir,inter_dir):
    n = len(nodes)
    similarity = np.zeros((n,n))
    interaction = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            list1 = list(props[nodes[i]])
            list2 = list(props[nodes[j]])
            union = get_union(list1,list2)
            inter = get_intersection(list1,list2)
            if len(list1)==0 or len(list2)==0 or len(inter)==0:
                similarity[i][j]=0
                interaction[i][j]=0
            else:
                similarity[i][j] = 1.0*len(inter)/len(union)
                interaction[i][j]=1
    np.savetxt(sim_dir,similarity,fmt='%.5f')
    np.savetxt(inter_dir,interaction,fmt='%d')
    return similarity,interaction
    

def get_union(list_a,list_b):
    list_c = list(set(list_a).union(set(list_b)))
    return list_c

def get_intersection(list_a,list_b):
    list_c = list(set(list_a).intersection(set(list_b)))
    return list_c
 
   
def get_enrichment(file_path):
    f = open(file_path)
    firstline = f.readline()
    line = f.readline()
    res = []
    while line:
        temp = line.strip('\n').split('\t')
        line = f.readline()
        if temp[4]=='<0.00050':
            res.append(temp[5])
    f.close()
    return res
    
    
def load_dataset(directory,bin_num):
    f1 = open(directory+'ids.txt')
    f2 = open(directory+'interaction.txt')
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


def load_raw_dataset(dir_net, header = False):
    with open(dir_net, mode='r') as f:
        if header:
            next(f)
        net_nA = []
        net_nB = []
        for line in f:
            na, nb = line.strip("\n").split("\t")
            net_nA.append(na)
            net_nB.append(nb)
    G = nx.Graph()
    G.add_edges_from(list(zip(net_nA, net_nB)))
    net_node = list(max(nx.connected_components(G), key = len))
    net_node = sorted(net_node)
    NODE2ID = dict()
    ID2NODE = dict()
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
    return A, ID2NODE
    
    
    
def feature_generator(A, n_comp, preprocess = None):
    assert preprocess in [None, "svd"], "Only accept preprocess = None, svd."
    if preprocess == None:
        feat = A
    elif preprocess == "svd":
        svd = TruncatedSVD(n_components=n_comp)
        feat = svd.fit_transform(A)
        explained_var = svd.explained_variance_ratio_.sum()
        print("The current component number {0} with SVD can explain {1:.3f} of total variance!".format(n_comp,explained_var))
    return feat
    
    
def to_sparse_tensor(matrix,device):
    if sp.issparse(matrix):
        coo = matrix.tocoo()
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    elif torch.is_tensor(matrix):
        row, col = matrix.nonzero().t()
        indices = torch.stack([row, col])
        values = matrix[row, col]
        shape = torch.Size(matrix.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    elif isinstance(matrix, np.ndarray):
        sparse_tensor = torch.FloatTensor(matrix)
    else:
        raise ValueError(f"matrix must be scipy.sparse or torch.Tensor or numpy.array (got {type(matrix)} instead).")
    sparse_tensor = sparse_tensor.to(device)
    return sparse_tensor if isinstance(matrix, np.ndarray) else sparse_tensor.coalesce()


# 计算归一化邻接矩阵
def normalize_adj(adj, sparse=True):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    if sparse:
        return res.tocoo()
    else:
        return res.todense()

# 返回0，1，2...k阶邻接矩阵
def adj_polynomials(adj, k, sparse=True):
    adj_normalized = normalize_adj(adj, sparse=sparse)
    p_k = []
    if sparse:
        p_k.append(sp.eye(adj.shape[0]))
    else:
        p_k.append(np.eye(adj.shape[0]))
    p_k.append(adj_normalized)
    for p in range(2, k+1):
        p_k.append(sp.csr_matrix.power(adj_normalized, p))
    return p_k



def cluster_infer(Z_pred, ID2NODE):
    # 推测簇中心
    clust_results = dict()
    for idx in range(Z_pred.shape[0]):
        # 节点ID，及其所属的社区
        clust_results[ID2NODE[idx]] = []
        clust_sets = np.where(Z_pred[idx, ] > 0)[0]
        for cidx in clust_sets:
            clust_results[ID2NODE[idx]].append(cidx)
    return clust_results   


def community_to_node(input_dict):
    result = {}
    # 遍历每个节点和它所对应的社区列表
    for node, community_list in input_dict.items():
        # 遍历每个社区
        for community in community_list:
            # 如果该社区已有节点列表，直接加入节点
            if community in result:
                result[community].append(node)
            # 如果该社区没有节点列表，新建一个列表
            else:
                result[community] = [node]

    return result

def node_to_community(input_dict):
    result = {}
    # 遍历每个节点和它所对应的社区列表
    for community,nodes in input_dict.items():
        # 遍历每个社区
        for node in nodes:
            # 如果该社区已有节点列表，直接加入节点
            if node in result:
                result[node].append(community)
            # 如果该社区没有节点列表，新建一个列表
            else:
                result[node] = [community]

    return result
    
def excel_read(k,file_path):
    xlrd.xlsx.ensure_elementtree_imported(False, None)
    xlrd.xlsx.Element_has_iter = True
    worksheet = xlrd.open_workbook(file_path)
    sheet_names = worksheet.sheet_names()
    t_list = []
    n = 0
    for i in range(k):
        t_list.append([])
    for sheet_name in sheet_names:
        sheet = worksheet.sheet_by_name(sheet_name)
        rows = sheet.nrows
        cols = sheet.ncols
        nodes_from = []
        for i in range(1,rows):
            cell = sheet.cell_value(i, 0)
            try:
                cell = str(cell)
                nodes_from.append(cell)
            except ValueError:
                pass 
        nodes_from = list(nodes_from)
        if n < k:
            t_list[n] = nodes_from
            n += 1
    return t_list
    
   
def sample_nodes(List,num):
    t = []
    for i in range(len(List)):
        t.append([])
        if num==-1:
            n = len(List[i])//4
        else:
            n = num
        t[i] = np.random.choice(List[i],n,replace=True)
    return t
    
def jc_vector(nodes,clust_result,community_result):
    # 节点ID(str)：社区数字(int)
    # 社区数字(int)：节点ID(str)
    List = []
    length = max(list(community_result))
    for i in range(length):
        List.append(0)

    jc_list = []
    for i in range(length):
        jc_list.append(0)
    # 统计社区的出现次数
    for node,community_list in clust_result.items():
        for community in community_list:
            List[community-1] += 1
    # 获取靶点集合在社区上的映射
    keys =  clust_result.keys()
    for node in nodes:
        if node in keys:
            communitys = clust_result[node]
            for community in communitys:
                jc = List[community-1]/len(community_result)
                jc_list[community-1] += jc
    return jc_list
    
def euclidean_distance(List,jc_list_disease):
    t = []
    for j in List:
        t.append(j)
    t_drug = np.array(t)
    jc_list_disease = np.array(jc_list_disease)
    dist = np.sqrt(np.sum(np.square(jc_list_disease - t_drug)))/len(List)
    return dist
    
def get_network(network_file, only_lcc):
    network = network_utilities.create_network_from_sif_file(network_file, use_edge_data=False, delim=None,
                                                             include_unconnected=True)
    # print len(network.nodes()), len(network.edges())
    if only_lcc and not network_file.endswith(".lcc"):
        print
        "Shrinking network to its LCC", len(network.nodes()), len(network.edges())
        components = network_utilities.get_connected_components(network, False)
        network = network_utilities.get_subgraph(network, components[0])
        print
        "Final shape:", len(network.nodes()), len(network.edges())
        # print len(network.nodes()), len(network.edges())
        network_lcc_file = network_file + ".lcc"
        if not os.path.exists(network_lcc_file):
            f = open(network_lcc_file, 'w')
            for u, v in network.edges():
                f.write("%s 1 %s\n" % (u, v))
            f.close()
    return network

def get_random_nodes(nodes, network, bins=None, n_random=1000, min_bin_size=100, degree_aware=True, seed=None):
    if bins is None:
        # Get degree bins of the network
        bins = network_utilities.get_degree_binning(network, min_bin_size)
    nodes_random = network_utilities.pick_random_nodes_matching_selected(network, bins, nodes, n_random, degree_aware,
                                                                         seed=seed)
    return nodes_random

def calculate_closest_distance(network, nodes_from, nodes_to, lengths=None):
    values_outer = []
    if lengths is None:
        for node_from in nodes_from:
            values = []
            for node_to in nodes_to:
                val = network_utilities.get_shortest_path_length_between(network, node_from, node_to)
                values.append(val)
            d = min(values)
            # print d,
            values_outer.append(d)
    else:
        for node_from in nodes_from:
            values = []
            vals = lengths[node_from]
            for node_to in nodes_to:
                val = vals[node_to]
                values.append(val)
            d = min(values)
            values_outer.append(d)
    d = np.mean(values_outer)
    # print d
    return d
    
def calculate_proximity(network, nodes_from, nodes_to, nodes_from_random=None, nodes_to_random=None, bins=None,
                        n_random=1000, min_bin_size=100, seed=452456, lengths=None, distance="closest"):
    """
    Calculate proximity from nodes_from to nodes_to
    If degree binning or random nodes are not given, they are generated
    lengths: precalculated shortest path length dictionary
    """
    nodes_network = set(network.nodes())
    nodes_from = set(nodes_from) & nodes_network
    nodes_to = set(nodes_to) & nodes_network
    if len(nodes_from) == 0 or len(nodes_to) == 0:
        return None  # At least one of the node group not in network
    # 下面部分没有运行到
    # if distance != "closest":
    #     lengths = network_utilities.get_shortest_path_lengths(network, "temp_n%d_e%d.sif.pcl" % (len(nodes_network), network.number_of_edges()))
    #     d = network_utilities.get_separation(network, lengths, nodes_from, nodes_to, distance, parameters={})
    # else:
    # 获取两两节点之间的距离
    d = calculate_closest_distance(network, nodes_from, nodes_to, lengths)
    # 按照节点度数分箱
    if bins is None:
        bins = network_utilities.get_degree_binning(network, min_bin_size,lengths)
    if nodes_from_random is None:
        nodes_from_random = get_random_nodes(nodes_from, network, bins=bins, n_random=n_random,min_bin_size=min_bin_size, seed=seed)
    if nodes_to_random is None:
        nodes_to_random = get_random_nodes(nodes_to, network, bins=bins, n_random=n_random, min_bin_size=min_bin_size,seed=seed)
    random_values_list = zip(nodes_from_random, nodes_to_random)
    values = np.empty(len(nodes_from_random))
    for i, values_random in enumerate(random_values_list):
        nodes_from, nodes_to = values_random
        if distance != "closest":
            values[i] = network_utilities.get_separation(network, lengths, nodes_from, nodes_to, distance,
                                                         parameters={})
        else:
            values[i] = calculate_closest_distance(network, nodes_from, nodes_to, lengths)
    m, s = np.mean(values), np.std(values)
    if s == 0:
        z = 0.0
    else:
        z = (d - m) / s
    return d, z, (m, s)  # (z, pval)



def add_padding_idx(vec):
    if len(vec.shape) == 1:
		    return np.asarray([np.sort(np.asarray(v) + 1).astype('int') for v in tqdm(vec)])
    else:
		    vec = np.asarray(vec) + 1
		    vec = np.sort(vec, axis=-1)
    return vec.astype('int')


def np2tensor_hyper(vec, dtype):
	  vec = np.asarray(vec)
	  if len(vec.shape) == 1:
		    return [torch.as_tensor(v, dtype=dtype) for v in vec]
	  else:
		    return torch.as_tensor(vec, dtype = dtype)


def walkpath2str(walk):
	  return [list(map(str, w)) for w in tqdm(walk)]


def roc_auc_cuda(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy().reshape((-1, 1))
    y_pred = y_pred.cpu().detach().numpy().reshape((-1, 1))
    return accuracy(y_true,y_pred)
   

def accuracy(output, target):
    pred = output >= 0.1
    truth = target >= 0.1
    nums=0
    count=0
    for i in range(len(pred)): 
        if pred[i]==True:
            count+=1
            if truth[i]==True:
                nums+=1
    acc = torch.sum(pred.eq(truth))
    acc = float(acc) * 1.0 / (truth.shape[0] * 1.0)
    return acc


def build_hash(data):
	  dict1 = set()
	  for datum in data:
	      # We need sort here to make sure the order is right
		    datum.sort()
		    dict1.add(tuple(datum))
	  del data
	  return dict1

def parallel_build_hash(data, func,initial):
	import multiprocessing
	cpu_num = multiprocessing.cpu_count()
	data = np.array_split(data, cpu_num * 3)
	dict1 = initial.copy()
	pool = ProcessPoolExecutor(max_workers=cpu_num)
	process_list = []

	if func == 'build_hash':
		func = build_hash
	for datum in data:
		process_list.append(pool.submit(func, datum))

	for p in as_completed(process_list):
		a = p.result()
		dict1.update(a)

	pool.shutdown(wait=True)
	

	return dict1

def generate_negative_edge(x, length):
	pos = np.random.choice(len(pos_edges), length)
	pos = pos_edges[pos]
	negative = []

	temp_num_list = np.array([0] + list(num_list))

	id_choices = np.array([[0, 1], [1, 2], [0, 2]])
	id = np.random.choice([0, 1, 2], length * neg_num, replace=True)
	id = id_choices[id]

	start_1 = temp_num_list[id[:, 0]]
	end_1 = temp_num_list[id[:, 0] + 1]

	start_2 = temp_num_list[id[:, 1]]
	end_2 = temp_num_list[id[:, 1] + 1]

	if len(num_list) == 3:
		for i in range(neg_num * length):
			temp = [
				np.random.randint(
					start_1[i],
					end_1[i]) + 1,
				np.random.randint(
					start_2[i],
					end_2[i]) + 1]
			while tuple(temp) in dict2:
				temp = [
					np.random.randint(
						start_1[i],
						end_1[i]) + 1,
					np.random.randint(
						start_2[i],
						end_2[i]) + 1]
			negative.append(temp)

	return list(pos), negative
 

def get_union(list_a,list_b):
    list_c = list(set(list_a).union(set(list_b)))
    return list_c

def get_intersection(list_a,list_b):
    list_c = list(set(list_a).intersection(set(list_b)))
    return list_c
    
# 获取邻接矩阵
def get_adj(cluster_results,nodes,weight=False):
    adj = np.zeros((len(nodes),len(nodes)))
    # 根据community_results判断adj矩阵是否邻接
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            list1 = cluster_results[str(nodes[i])]
            list2 = cluster_results[str(nodes[j])]
            intersection = get_intersection(list1,list2)
            if weight==False:
                adj[i][j] = 0 if (len(intersection)==0) else 1
            else:
                union = get_union(list1,list2)
                if len(union)==0:
                    adj[i][j]=0
                else:
                    adj[i][j] = len(intersection)/len(union)
            if i==j:
                adj[i][j]=0
    return adj


# 统计学方法
def get_avg_length(adj,samp_num):
    temp = [0 for i in range(samp_num)]
    for i in range(samp_num):
        flag=False
        while not flag:
            samp_1 = random.randint(0,len(adj)-1)
            samp_2 = random.randint(0,len(adj)-1)
            if samp_1!=samp_2:
                flag=True
                dis = 0
                for j in range(len(adj)):
                    dis+=math.pow(abs(adj[samp_2][j]-adj[samp_1][j]),2)
                dis=math.sqrt(dis)
                temp[i]=dis
    mean = statistics.mean(temp)
    variance = statistics.stdev(temp)
    return mean,variance               


# 获取直接密度可达的点
def get_link(adj,threshold):
    # 计算一步可以直达的点
    link = []
    for node1 in range(len(adj)):
        print(node1)
        temp = []
        for node2 in range(len(adj)):
            dis = 0
            for j in range(len(adj)):
                dis+=math.pow(abs(adj[node1][j]-adj[node2][j]),2)
            dis=math.sqrt(dis)
            if node2!=node1 and dis<threshold:
                temp.append(node2)
        link.append(temp)
    with open('../Output/BeforeH.txt', 'wb') as handle:
        pickle.dump(link, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    return link
    

def get_min_index(list1):
    min_num = list1[0]
    min_index = 0
    for i in range(len(list1)):
        if list1[i]<min_num:
            min_index=i
            min_num=list1[i]
    return min_num,min_index
    
    
# 计算最相近的k个点
def compute_k_nearest(sim,minpts):
    sim_minpts = []
    for i in range(len(sim)): 
        max_index = [j for j in range(minpts)]
        max_data = [sim[i][j] for j in range(minpts)]
        for j in range(minpts,len(sim[i])):
            min_num,min_index=get_min_index(max_data)
            # 替换temp index中的最大数值
            if sim[i][j]>min_num:
                max_index[min_index]=j
                max_data[min_index]=sim[i][j]
        sim_minpts.append(max_index)
    return sim_minpts                


# 获取密度相连的点(还需要再改改)
def get_hyperedge(samp,link,sim,minpts):
    res = set()
    res.add(samp)
    # 记录新引入的节点
    new_node = link[samp]
    # 最多迭代minpts-1次
    for i in range(minpts-1):
        # 尝试将朋友的朋友也加入
        temp_node = set()
        for node in new_node:
            if node not in res:
                res.add(node)
                for node1 in link[node]:
                    temp_node.add(node1)
                    if len(res)>=minpts:
                        break
        if len(res)>=minpts:
            break
        new_node = list(temp_node)
    if len(res)<minpts:
        res = sim[samp]
    hyper_edge = [0 for i in range(len(link))]
    for node in res:
        hyper_edge[node]=1
    return hyper_edge
    

def generate_negative(x,weight,neg_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 负边列表
    edge_list = []
    # 新的列表？
    label_list = []
    # 修改整个列表，里面保存负边相对于正边哪些特征发生了变化
    for j, sample in enumerate(x):
        for i in range(neg_num):
            # temp是其中一条超边
            temp = np.copy(sample)
            # 用于判断如何调整超边
            simple_or_hard = np.random.random()
            temp = np.copy(sample)
            # 取一个和原有边相似(在前面dropout了)的边作为正边
            if simple_or_hard <= 0.5:
                label_list.append(1)
                id1 = random.randint(0,len(x)-1)
                temp = x[id1]
            # 否则随机生成一个不是正边的边
            else:
                id1 = random.randint(0,len(x)-1)
                # 生成一个和x[id]类似的负边
                label_list.append(0)
                random.shuffle(temp)
            # 生成的一条边
            edge_list.append(temp)     
    edge_list = np.array(edge_list)
    x = np2tensor_hyper(edge_list,dtype=torch.long)
    label_list = np.array(label_list)
    y = torch.zeros((len(x),1),device="cuda:0")
    for i in range(len(label_list)):
        y[i]=label_list[i]
    weight = torch.ones((len(x),1),device="cuda:0")
    return x,y,weight

    
def train_batch_hyperedge(model, loss_func, batch_data, batch_weight,neg_num):
    x = batch_data
    w = batch_weight
    # When label is not generated, prepare the data
    x, y, w = generate_negative(x,w,neg_num)
    index = torch.randperm(len(x))
    x, y, w = x[index], y[index], w[index]
    # forward
    pred, recon_loss = model(x, return_recon =True)
    loss = loss_func(pred, y, weight=w)
    return pred, y, loss, recon_loss   


def eval_batch_hyperedge(model, batch_data, batch_weight,neg_num):
    x = batch_data
    w = batch_weight
    # When label is not generated, prepare the data
    x, y, w = generate_negative(x,w,neg_num)
    index = torch.randperm(len(x))
    x, y, w = x[index], y[index], w[index]
    # forward
    pred, recon_loss = model(x, return_recon =True)
    return pred, y

def test_batch_hyperedge(model, batch_data, batch_weight):
    x = batch_data
    w = torch.ones((len(x),1),device="cuda:0")
    x = np2tensor_hyper(x,dtype=torch.long)
    # forward
    pred,_ = model(x, return_recon = True)
    return pred
    
    
def train_epoch(args, model, loss_func, training_data, optimizer, batch_size,neg_num):
    model_1 = model
    (loss_1, beta) = loss_func
    (edges,edge_weight) = training_data
    alpha = args.alpha
    y = torch.tensor([])
    # Permutate all the data
    index = torch.randperm(len(edges))
    edges = edges[index]
    model_1.train()
    _,length1 = edges.shape
    bce_total_loss = 0
    recon_total_loss = 0
    # 准确率，标签，预测标签
    acc_list, y_list, pred_list = [], [], []
    batch_num = int(math.floor(len(edges) / batch_size))
    bar = trange(batch_num, mininterval=0.1, desc='  - (Training) ', leave=False, )
    for i in bar:
        batch_edge = edges[i * batch_size:(i + 1) * batch_size].reshape(-1,length1)
        batch_edge_weight = edge_weight[i * batch_size:(i + 1) * batch_size]
        pred, batch_y, loss_bce, loss_recon = train_batch_hyperedge(model_1, loss_1, batch_edge, batch_edge_weight,neg_num)
        loss = beta * loss_bce + loss_recon * args.rw
        acc_list.append(accuracy(pred, batch_y))
        y_list.append(batch_y)
        pred_list.append(pred)
        # 令梯度归零
        for opt in optimizer:
            opt.zero_grad()
        
        # backward
        loss.backward()
        
        # update parameters
        for opt in optimizer:
            opt.step()
        
        bce_total_loss += loss_bce.item()
        recon_total_loss += loss_recon.item()
    y = torch.cat(y_list)
    pred = torch.cat(pred_list)
    return bce_total_loss / batch_num, recon_total_loss / batch_num, np.mean(acc_list)

def eval_epoch(args, model, eval_data, batch_size,neg_num):
    model_1 = model
    (edges,edge_weight) = eval_data
    alpha = args.alpha
    y = torch.tensor([])
    # Permutate all the data
    index = torch.randperm(len(edges))
    edges = edges[index]
    model_1.eval()
    _,length1 = edges.shape
    # 准确率，标签，预测标签
    acc_list, y_list, pred_list = [], [], []
    batch_num = int(math.floor(len(edges) / batch_size))
    bar = trange(batch_num, mininterval=0.1, desc='  - (Training) ', leave=False, )
    for i in bar:
        batch_edge = edges[i * batch_size:(i + 1) * batch_size].reshape(-1,length1)
        batch_edge_weight = edge_weight[i * batch_size:(i + 1) * batch_size]
        pred, batch_y = eval_batch_hyperedge(model_1, batch_edge, batch_edge_weight,neg_num)
        acc_list.append(accuracy(pred, batch_y))
    return np.mean(acc_list)


def test_epoch(args, model, testing_data, batch_size):
    model_1 = model
    (edges,edge_weight) = testing_data
    alpha = args.alpha
    # 用于保存结果
    res = []
    model_1.eval()
    _,length1 = edges.shape
    batch_num = int(math.floor(len(edges) / batch_size))
    bar = trange(batch_num, mininterval=0.1, desc='  - (Training) ', leave=False, )
    for i in bar:
        batch_edge = edges[i * batch_size:(i + 1) * batch_size].reshape(-1,length1)
        batch_edge_weight = edge_weight[i * batch_size:(i + 1) * batch_size]
        pred = test_batch_hyperedge(model_1, batch_edge, batch_edge_weight)
        res.append(pred)
    # 默认舍弃最后一个batchsize的数据
    res = torch.cat(res)
    print(res)
    return res   


def count_cover(res,len_H):
    res_set = set()
    for i in range(len(res)):
        for j in range(len(res[i])):
            if res[i][j]!=0:
                res_set.add(j)      
    return len(res_set)/len_H


# 每次削减一定数值，直到覆盖率满足要求
def get_end_H(H,y,coverage=0.95,shrink=0.001):
    cover = 0
    flag = 1+1e-6
    H_set = set()
    for i in range(len(H)):
        for j in range(len(H[i])):
            if H[i][j]!=0:
                H_set.add(j)
    len_H = len(H_set)       
    res = []
    while cover<coverage:
        print(flag)
        for i in range(len(y)):
            if flag>y[i]>flag-shrink and cover<coverage:
                cover = count_cover(res,len_H)
                res.append(H[i])
        flag-=shrink
    return res

def normalized(x,max_index):
    res = []
    for i in range(len(x)):
        temp = [0 for i in range(max_index)]
        for index in x[i]:
            temp[index]=1
        res.append(temp)
    return res
    

def get_coverage(list1,list2):
    result = []
    # for all recipe
    for i in range(len(list1)):
        temp = []
        # for all hypergraph
        for j in range(len(list2)):
            inter = set()
            for k in range(len(list1[i])):
                if list1[i][k]==1:
                    inter.add(k)
            len1 = len(inter)
            for index in list2[j]:
                inter.add(index)
            temp.append((len1+len(list2[j])-len(inter))/len(list2[j]))
        result.append(temp)
    return result
    
# 这个作为测试集
def get_HyperGraph(link,sim,min_pts,coverage=0.99):
    # 记录这个点是否访问过
    marked = [0 for i in range(len(link))]
    H = []
    # 每次以较大概率取一个在访问点之外的点，小概率取访问的点
    flag = False
    while not flag:
        # 记录没有访问过的点
        temp1 = []
        for i in range(len(marked)):
            if marked[i]==0:
                temp1.append(i)
        rand = random.random()
        if rand<0.9:
            # 从访问点之外取点
            samped = False
            while not samped:
                samp = random.randint(0,len(link)-1)
                samped = True if samp in temp1 else False
        else:
            # 取样一个点
            samp = random.randint(0,len(link)-1)
              
        # 获取这个点的超边
        edge = get_hyperedge(samp,link,sim,min_pts)
        H.append(edge)
        for i in range(len(edge)):
            if edge[i]==1:
                marked[i]=1
        # 判断flag是否为True
        flag = True if sum(marked)/len(marked)>coverage else False
    new_H = []
    for edge in H:
        if sum(edge)<min_pts:
            pass
        else:
            new_H.append(edge)
    return new_H
    
def read_recipes(recipes_dir,nodes):
    worksheet = xlrd.open_workbook(recipes_dir)
    sheet_names= worksheet.sheet_names()
    recipes = []
    for sheet_name in sheet_names:
        sheet = worksheet.sheet_by_name(sheet_name)
        rows = sheet.nrows
        cols = sheet.ncols
        recipe = []
        for i in range(1,rows):
            cell = sheet.cell_value(i, 0)
            try:
                cell = str(int(cell))
                recipe.append(cell)
            except ValueError:
                pass
        recipes.append(recipe)
    recipes_id = []
    for i in range(len(recipes)):
        recipe_id=[]
        for j in range(len(recipes[i])):
            for k in range(len(nodes)):
                if nodes[k]==int(recipes[i][j]):
                    recipe_id.append(k)
        recipes_id.append(recipe_id)
    return recipes_id
    
    
def read_str_recipes(recipes_dir,nodes,node2id):
    worksheet = xlrd.open_workbook(recipes_dir)
    sheet_names= worksheet.sheet_names()
    recipes = []
    for sheet_name in sheet_names:
        sheet = worksheet.sheet_by_name(sheet_name)
        rows = sheet.nrows
        cols = sheet.ncols
        recipe = []
        for i in range(1,rows):
            cell = sheet.cell_value(i, 0)
            try:
                cell = str(cell)
                recipe.append(node2id[cell])
            except:
                pass
        recipes.append(recipe)
    recipes_id = []
    for i in range(len(recipes)):
        recipe_id=[]
        for j in range(len(recipes[i])):
            for k in range(len(nodes)):
                if nodes[k]==int(recipes[i][j]):
                    recipe_id.append(k)
        recipes_id.append(recipe_id)
    return recipes_id

def read_drugs(drugs_dir,nodes):
    worksheet = xlrd.open_workbook(drugs_dir)
    sheet_names= worksheet.sheet_names()
    drugs = []
    for sheet_name in sheet_names:
        sheet = worksheet.sheet_by_name(sheet_name)
        rows = sheet.nrows
        cols = sheet.ncols
        drug = []
        for i in range(1,rows):
            cell = sheet.cell_value(i, 0)
            try:
                cell = str(int(cell))
                drug.append(cell)
            except ValueError:
                pass
        drugs.append(drug)
    drugs_id = []
    for i in range(len(drugs)):
        drug_id=[]
        for j in range(len(drugs[i])):
            for k in range(len(nodes)):
                if nodes[k]==int(drugs[i][j]):
                    drug_id.append(k)
        drugs_id.append(drug_id)
    return drugs_id

def read_str_drugs(drugs_dir,nodes,node2id):
    worksheet = xlrd.open_workbook(drugs_dir)
    sheet_names= worksheet.sheet_names()
    drugs = []
    for sheet_name in sheet_names:
        sheet = worksheet.sheet_by_name(sheet_name)
        rows = sheet.nrows
        cols = sheet.ncols
        drug = []
        for i in range(1,rows):
            cell = sheet.cell_value(i, 0)
            try:
                cell = str(cell)
                drug.append(node2id[cell])
            except:
                pass
        drugs.append(drug)
    drugs_id = []
    for i in range(len(drugs)):
        drug_id=[]
        for j in range(len(drugs[i])):
            for k in range(len(nodes)):
                if nodes[k]==int(drugs[i][j]):
                    drug_id.append(k)
        drugs_id.append(drug_id)
    return drugs_id

# 这个作为训练集
def get_over_HyperGraph(link,sim,min_pts,coverage=5):
    # 记录这个点的访问次数
    marked = [0 for i in range(len(link))]
    # 记录超边
    H = []
    # 每次以较大概率取一个在访问点之外的点，小概率取访问的点
    flag = False
    while not flag:
        # 记录没有充分覆盖的点
        temp1 = []
        for i in range(len(marked)):
            if marked[i]<coverage:
                temp1.append(i)
        rand = random.random()
        if rand<0.8:
            # 选择一个未充分覆盖的点
            samped = False
            while not samped:
                samp = random.randint(0,len(link)-1)
                samped = True if samp in temp1 else False
        else:
            # 随机取样一个点
            samp = random.randint(0,len(link)-1)
              
        # 获取这个点的超边
        edge = get_hyperedge(samp,link,sim,min_pts)
        H.append(edge)
        # 记录这条边上的点
        for i in range(len(edge)):
            if edge[i]==1:
                marked[i]+=1
        # 判断flag是否为True
        flag = True if min(marked)>=coverage else False
    # 遗忘一部分点
    for i in range(len(H)):
        for j in range(len(H[i])):
            if H[i][j]==1:
                rand = random.random()
                # 以一定的概率遗忘值1值点
                if rand<0.25:
                    H[i][j]=0
            else:
                rand = random.random()
                # 这里概率小是因为0值点数量较多
                if rand<0.001:
                    H[i][j]=1
    return H


def get_testset(recipes_id,drugs_id):
    test_x = []
    test_y = []
    for i in range(len(recipes_id)//5):
        test_x.append(recipes_id[i])
        test_y.append(1)
    for i in range(len(recipes_id)//5):
        recipe=set()
        marked = [0 for i in range(len(drugs_id))]
        drug_number = random.randint(4,6)
        for j in range(drug_number):
            drug_index = random.randint(0,len(drugs_id)-1)
            while marked[drug_index]==1:
                drug_index = random.randint(0,len(drugs_id)-1)
            marked[drug_index]=1
            for data in drugs_id[drug_index]:
                recipe.add(data)
        test_x.append(list(recipe))
        test_y.append(0)
    test_x,test_y = np.array(test_x),np.array(test_y)
    p = np.random.permutation(range(len(test_x)))
    test_x = test_x[p]
    test_y = test_y[p]
    return test_x,test_y


def get_trainset(recipes_id,drugs_id):
    train_x = []
    train_y = []
    for i in range(len(recipes_id)//5,len(recipes_id)):
        train_x.append(recipes_id[i])
        train_y.append(1)
    # get random recipes
    for i in range(200):
        recipe=set()
        marked = [0 for i in range(len(drugs_id))] 
        drug_number = random.randint(4,6)
        for j in range(drug_number):
            drug_index = random.randint(0,len(drugs_id)-1)
            while marked[drug_index]==1:
                drug_index = random.randint(0,len(drugs_id)-1)
            marked[drug_index]=1
            for data in drugs_id[drug_index]:
                recipe.add(data)
        train_x.append(list(recipe))
        train_y.append(0)
    train_x,train_y = np.array(train_x),np.array(train_y)
    p = np.random.permutation(range(len(train_x)))
    train_x = train_x[p]
    train_y = train_y[p]
    return train_x,train_y


def evaluate(pred_y,test_y,yz_list):
    best_score = 0
    for yz in yz_list:
        flag = [0 for i in range(len(pred_y))]
        score = 0
        for j in range(len(pred_y)):
            if pred_y[j]>yz:
                flag[j]=1
            else:
                flag[j]=0
        for i in range(len(pred_y)):
            if test_y[i]==flag[i]:
                score+=1/len(pred_y)
        if score>best_score:
            best_score = score
            print(yz)
    return best_score