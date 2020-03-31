import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io
from torch.autograd import Variable
import math
from functools import partial

from utils import *

from d3_graph_conv import *


#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##################################################################################
# Model
##################################################################################
class autoencoder(nn.Module):
    def __init__(self, in_channels, conv_ch):
        super(autoencoder, self).__init__()
        self.conv1 = d3GraphConv(in_channels, conv_ch)
        self.conv2 = d3GraphConv(conv_ch, conv_ch*2)
        #self.conv3 = d3GraphConv(conv_ch*2, conv_ch*2)
        #self.conv4 = d3GraphConv(conv_ch*2, conv_ch*2)
        #self.conv5 = d3GraphConv(conv_ch*2, conv_ch*2)

        self.edge_decoder_conv = d3GraphConv(conv_ch*2, conv_ch)

        #self.x_decoder_conv1 = d3GraphConv(conv_ch*4, conv_ch*2)
        self.x_decoder_conv2 = d3GraphConv(conv_ch*2, conv_ch)
        self.x_decoder_conv3 = d3GraphConv(conv_ch, in_channels)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, x, edge_index):
        edge_index = edge_index.permute(0, 2, 1)
        x = F.leaky_relu(self.conv1(x, edge_index))
        z = F.leaky_relu(self.conv2(x, edge_index))
        #z = F.leaky_relu(self.conv3(z, edge_index)) + z
        #z = F.relu(self.conv4(z, edge_index)) + z
        #z = F.relu(self.conv5(z, edge_index)) + z

        recon_edge = F.leaky_relu(self.edge_decoder_conv(z,edge_index))
        recon_edge = torch.sigmoid(torch.matmul(recon_edge, recon_edge.permute(0, 2, 1)))

        #x = F.relu(self.x_decoder_conv1(z, edge_index))
        x = F.leaky_relu(self.x_decoder_conv2(z, edge_index))
        x = self.x_decoder_conv3(x, edge_index)

        return recon_edge, x, z


class autoencoderat(nn.Module):
    def __init__(self, in_channels, conv_ch):
        super(autoencoderat, self).__init__()
        self.conv1 = d3GraphConvat(in_channels, conv_ch)
        self.conv2 = d3GraphConvat(conv_ch, conv_ch*2)
        #self.conv3 = d3GraphConv(conv_ch*2, conv_ch*2)
        #self.conv4 = d3GraphConv(conv_ch*2, conv_ch*2)
        #self.conv5 = d3GraphConv(conv_ch*2, conv_ch*4)

        self.edge_decoder_conv = d3GraphConvat(conv_ch*2, conv_ch)

        #self.x_decoder_conv1 = d3GraphConv(conv_ch*4, conv_ch*2)
        self.x_decoder_conv2 = d3GraphConvat(conv_ch*2, conv_ch)
        self.x_decoder_conv3 = d3GraphConvat(conv_ch, in_channels)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, x, edge_index):

        x = F.leaky_relu(self.conv1(x, edge_index))
        z = F.leaky_relu(self.conv2(x, edge_index))


        recon_edge = F.leaky_relu(self.edge_decoder_conv(z,edge_index))
        recon_edge = torch.sigmoid(torch.matmul(recon_edge, recon_edge.permute(0, 2, 1)))

        x = F.leaky_relu(self.x_decoder_conv2(z, edge_index))
        x = F.leaky_relu(self.x_decoder_conv3(x, edge_index))

        return recon_edge, x, z


