import math
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphAttentionLayer(nn.Module):
    """Simple GAT layer (https://arxiv.org/abs/1710.10903), device-agnostic."""

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.a, gain=math.sqrt(2.0))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        h = torch.mm(x, self.W)
        n = h.size(0)
        a_input = torch.cat([h.repeat(1, n).view(n * n, -1), h.repeat(n, 1)], dim=1).view(n, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        return F.elu(h_prime) if self.concat else h_prime


class GraphConvolution(Module):
    """Simple GCN layer with temporal residual connection (DataParallel-safe)."""

    def __init__(self, in_features, out_features, bias=False, residual=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_residual = residual
        self.weight = Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if not residual:
            self.residual = None
        elif in_features == out_features:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=5, padding=2)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.1)

    def forward(self, x, adj):
        support = x.matmul(self.weight)
        output = adj.matmul(support)

        if self.bias is not None:
            output = output + self.bias

        if self.has_residual and self.residual is not None:
            if self.in_features != self.out_features:
                res = self.residual(x.permute(0, 2, 1)).permute(0, 2, 1)
                output = output + res
            else:
                output = output + self.residual(x)

        return output

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features})"


class SimilarityAdj(Module):
    """Feature cosine-similarity graph adjacency (DataParallel-safe)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight0 = Parameter(torch.empty(in_features, out_features))
        self.weight1 = Parameter(torch.empty(in_features, out_features))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight0)
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, x, seq_len=None):
        theta = torch.matmul(x, self.weight0)
        phi = torch.matmul(x, self.weight0)
        sim_graph = torch.matmul(theta, phi.permute(0, 2, 1))

        theta_norm = torch.norm(theta, p=2, dim=2, keepdim=True)
        phi_norm = torch.norm(phi, p=2, dim=2, keepdim=True)
        sim_graph = sim_graph / (theta_norm.matmul(phi_norm.permute(0, 2, 1)) + 1e-20)

        output = torch.zeros_like(sim_graph)
        if seq_len is None:
            adj2 = F.threshold(sim_graph, 0.7, 0.0)
            output = F.softmax(adj2, dim=-1)
        else:
            for i in range(len(seq_len)):
                l_i = max(1, min(int(seq_len[i]), x.shape[1]))
                adj2 = F.threshold(sim_graph[i, :l_i, :l_i], 0.7, 0.0)
                output[i, :l_i, :l_i] = F.softmax(adj2, dim=1)
        return output


class DistanceAdj(Module):
    """Temporal proximity graph adjacency computed purely on target device (DataParallel-safe)."""

    def __init__(self):
        super().__init__()
        self.sigma = Parameter(torch.tensor([0.1], dtype=torch.float32))

    def forward(self, batch_size: int, max_seqlen: int, device: torch.device = None, dtype: torch.dtype = torch.float32):
        if device is None:
            device = self.sigma.device
        idx = torch.arange(max_seqlen, device=device, dtype=torch.float32)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        adj = torch.exp(-dist / math.e).to(dtype=dtype)
        return adj.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
