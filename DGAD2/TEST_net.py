import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io
from torch.autograd import Variable
import math
from functools import partial

from utils import *

from torch_geometric.nn import  GATConv as GCNConv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##################################################################################
# Model
##################################################################################
class Lis_autoencoder(nn.Module):
    def __init__(self, in_channels, conv_ch, dropout=0.5):
        super(Lis_autoencoder, self).__init__()
        self.conv1 = GCNConv(in_channels, conv_ch)
        #self.dropout1 = nn.Dropout(dropout)
        self.conv2 = GCNConv(conv_ch, conv_ch*2)
        #self.dropout2 = nn.Dropout(dropout)

        self.edge_decoder_conv = GCNConv(conv_ch*2, conv_ch)
        #self.dropout3 = nn.Dropout(dropout)

        self.x_decoder_conv1 = GCNConv(conv_ch*2, conv_ch)
        #self.dropout4 = nn.Dropout(dropout)
        self.x_decoder_conv2 = GCNConv(conv_ch, in_channels)
        #self.dropout5 = nn.Dropout(dropout)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, x, edge_index):
        edge_index = torch.nonzero(edge_index).T
        x = F.leaky_relu(self.conv1(x, edge_index))
        z = F.leaky_relu(self.conv2(x, edge_index))

        recon_edge = F.leaky_relu(self.edge_decoder_conv(z, edge_index))
        recon_edge = torch.sigmoid(torch.matmul(recon_edge, recon_edge.T))

        x = F.leaky_relu(self.x_decoder_conv1(z, edge_index))
        x = F.leaky_relu(self.x_decoder_conv2(x, edge_index))

        return recon_edge, x, z




