import math

import torch
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn
from torch_geometric.nn import GATConv, SAGEConv, GCNConv

#DEVICE = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')

class d3GraphConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 dropout=0,
                 time=True):
        super(d3GraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.root_weight = Parameter(torch.Tensor(in_channels, out_channels).float())
        self.s_weight = Parameter(torch.Tensor(in_channels, out_channels).float())
        self.dropout = dropout

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).float())
        else:
            self.register_parameter('bias', None)

        if time:
            self.t_weight = Parameter(torch.Tensor(in_channels, out_channels).float())
        else:
            self.register_parameter('t_weight', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_channels)
        if self.t_weight is not None:
            self.t_weight.data.uniform_(-stdv, stdv)
        self.root_weight.data.uniform_(-stdv, stdv)
        self.s_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):

        out = torch.matmul(edge_index, torch.matmul(x, self.s_weight)) \
              * torch.div(1, torch.clamp(torch.sum(edge_index,dim=-1).view(x.shape[0], x.shape[1], 1), min=1.))

        if self.t_weight is not None:
            mean1 = x[:-1].mean()
            mean2 = x[1:].mean()
            tem_attention = F.cosine_similarity(x[:-1]-mean1, x[1:]-mean2, dim=-1).view(x.shape[0] - 1, x.shape[1], 1)

            out = out + torch.cat(
                [(out[0] * 0).view(1, out.shape[1], out.shape[2]),
                 torch.matmul(tem_attention * x[:-1], self.t_weight)],
                dim=0) + torch.cat(
                [torch.matmul(tem_attention * x[1:], self.t_weight),
                 (out[0] * 0).view(1, out.shape[1], out.shape[2])],
                dim=0)

        out = torch.matmul(x, self.root_weight) + out

        # Add bias (if wished).
        if self.bias is not None:
            out = out + self.bias

        out = F.dropout(out, self.dropout, training=self.training)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class d3GraphConvat(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0.5):
        super(d3GraphConvat, self).__init__()
        self.dropout = dropout
        self.out_channels = out_channels
        self.in_channels = in_channels


        self.gat = GATConv(in_channels, out_channels)

        #self.conv = torch.nn.Conv1d(out_channels, out_channels, 3, padding=1)

        #self.t1_weight = Parameter(torch.Tensor(out_channels, out_channels).float())
        self.t2_weight = Parameter(torch.Tensor(out_channels, out_channels).float())
        self.t1_weight = Parameter(torch.Tensor(out_channels, out_channels).float())

        self.reset_parameters()

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()
        stdv = 1.0 / math.sqrt(self.in_channels)
        #self.t1_weight.data.uniform_(-stdv, stdv)
        self.t2_weight.data.uniform_(-stdv, stdv)
        self.t1_weight.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):
        out = torch.zeros((x.shape[0],x.shape[1],self.out_channels), dtype=torch.float).to(x.device)
        for i in range(x.shape[0]):
            edge = torch.nonzero(edge_index[i]).T
            out[i] = self.gat(x[i], edge)

        #out = self.conv(out.permute(1,2,0)).permute(2,0,1)

        tem_attention = F.cosine_similarity(out[:-1], out[1:], dim=-1).view(out.shape[0]-1, out.shape[1], 1)

        out = out + torch.cat(
            [(out[0] * 0).view(1, out.shape[1], out.shape[2]), torch.matmul(tem_attention * out[:-1], self.t2_weight)],
            dim=0) + torch.cat(
            [torch.matmul(tem_attention * out[1:], self.t1_weight),(out[0] * 0).view(1, out.shape[1], out.shape[2])],
            dim=0)  # (time t to time t+1) (time t+1 to time t)

        out = F.dropout(out, self.dropout, training=self.training) #+ out

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


