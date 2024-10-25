import numpy as np
import torch


class BerPossionLoss():
    def __init__(self, num_nodes, num_edges):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_possible_edges = num_nodes**2 - num_nodes
        self.num_nonedges = self.num_possible_edges - self.num_edges
        edge_prob = self.num_edges / (self.num_nodes ** 2 - self.num_nodes)
        self.eps = -np.log(1 - edge_prob)
        self.ratio = self.num_nonedges/self.num_edges

    def loss_batch(self, emb, ones_idx, zeros_idx):
        # 对于连接的节点，我们希望它们的嵌入向量之间的点积越大;而对于未连接的节点，我们希望它们的嵌入向量之间的点积越小。
        # 因为点积越大，两个向量之间的夹角越小，它们之间的相似度就越高。
        
        # Loss for edges
        e1, e2 = ones_idx[:, 0], ones_idx[:, 1]
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.mean(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Loss for non-edges
        ne1, ne2 = zeros_idx[:, 0], zeros_idx[:, 1]
        loss_nonedges = torch.mean(torch.sum(emb[ne1] * emb[ne2], dim=1))
        
        # 负边loss降低很快,主要需要降低正边的loss
        return (20*loss_edges + loss_nonedges) / (1+20)


    def loss_cpu(self, emb, adj):
        e1, e2 = adj.nonzero()
        edge_dots = np.sum(emb[e1] * emb[e2], axis=1)
        loss_edges = -np.sum(np.log(-np.expm1(-self.eps - edge_dots)))
        self_dots_sum = np.sum(emb * emb)
        correction = self_dots_sum + np.sum(edge_dots)
        sum_emb = np.transpose(np.sum(emb, axis = 0))
        loss_nonedges = np.sum(emb @ sum_emb) - correction
        pos_loss = loss_edges / self.num_edges
        neg_loss = loss_nonedges / self.num_nonedges
        return pos_loss, neg_loss, (pos_loss+neg_loss*self.ratio)/(1+self.ratio)
        
        
def l2_reg_loss(model, scale=1e-5):
    loss = 0.0
    for w in model.get_weights():
        loss += w.pow(2.).sum()
    return loss * scale