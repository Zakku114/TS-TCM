import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math




class PolyGraphConvolution(nn.Module):

    def __init__(self, adj_pow, in_features, out_features, rand_seed):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rand_seed = rand_seed
        self.weight = nn.Parameter(torch.empty(adj_pow, in_features, out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.manual_seed(self.rand_seed)
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, poly_ls):
        node_num = x.shape[0]
        out = torch.zeros(node_num, self.out_features).to(x.device)
        for idx in range(len(poly_ls)):
            out += poly_ls[idx] @ (x @ self.weight[idx,:,:])
        return out


class PolyGCN(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        input_dim = kwargs['input_dim']
        hidden_dims = kwargs['hidden_dims']
        output_dim = kwargs['output_dim']
        adj_pow = kwargs['adj_pow']
        p_dropout = kwargs['p_dropout']
        rand_seed = kwargs['rand_seed']
        batch_norm = kwargs['batch_norm']

        self.p = p_dropout
        self.dropout = nn.Dropout(p = self.p)
        self.rand_seed = rand_seed
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int32)
        self.layers = nn.ModuleList([PolyGraphConvolution(adj_pow, input_dim, layer_dims[0], self.rand_seed)])
        for idx in range(len(layer_dims) - 1):
            self.layers.append(PolyGraphConvolution(adj_pow, layer_dims[idx], layer_dims[idx + 1], self.rand_seed))
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None

    def forward(self, x, poly_ls):
        for idx, poly_gcn in enumerate(self.layers):
            if self.p != 0 and idx != 0:
                x = self.dropout(x)
            x = poly_gcn(x, poly_ls)
            if idx != len(self.layers) - 1:
                x = F.relu(x)
                if self.batch_norm is not None:
                    x = self.batch_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]