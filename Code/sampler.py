import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import random


class EdgeSampler(Dataset):
    def __init__(self, A, node2bin, bin_num, num_pos):
        self.num_pos = num_pos
        self.A = A
        self.num_nodes = A.shape[0]
        self.node2bin = node2bin
        self.bin_num = bin_num
        self.sample_bin = int(self.num_pos // self.bin_num)

    def __getitem__(self, key):
        np.random.seed(key)
        
        # 获取本次采样的子节点
        sel_nodes_ls = []
        for arr in self.node2bin:
            sel_nodes_ls += list(np.random.choice(list(arr), size=self.sample_bin, replace = False))
        sel_nodes = np.array(sel_nodes_ls)
        
        # 获取正边edges
        self.A_sel = self.A[sel_nodes][:, sel_nodes]
        self.edges = np.transpose(sp.tril(self.A_sel, 1).nonzero())

        # 获取负边none_edges
        self.num_sel_nodes = len(sel_nodes)
        self.num_neg = self.edges.shape[0]
        generated = False
        while not generated:
            np.random.seed(key) 
            candidate_ne = np.random.randint(0, self.num_sel_nodes, size=(2*self.num_neg, 2), dtype=np.int64)
            cne1, cne2 = candidate_ne[:, 0], candidate_ne[:, 1]
            to_keep = (1 - self.A_sel[cne1, cne2]).astype(np.bool).A1 * (cne1 != cne2)
            next_nonedges = candidate_ne[to_keep][:self.num_neg]
            generated = to_keep.sum() >= self.num_neg
        next_edges = self.edges
        
        # 返回取样的正边，负边，子节点集合。
        return torch.LongTensor(next_edges), torch.LongTensor(next_nonedges), sel_nodes

    def __len__(self):
        return 2**32


def collate_fn(batch):
    edges, nonedges, batch_nodes = batch[0]
    return (edges, nonedges, batch_nodes)

def get_edge_sampler(A, node2bin, bin_num, batchsize, num_workers):
    data_source = EdgeSampler(A, node2bin, bin_num, batchsize)
    return DataLoader(data_source, num_workers=num_workers, collate_fn=collate_fn, pin_memory = True)