import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from quant import *
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nf_bit, na_bit):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.nf_bit = nf_bit
        self.na_bit = na_bit

    def forward(self, x, adj, quant=False):
        if quant:
            x = quantize_activations(x, num_bits=self.nf_bit)
            adj = adj.coalesce()
            quantized_values = quantize_activations(adj.values(), num_bits=self.na_bit)
            quantized_adj_values = torch.tensor(quantized_values).clone().detach()
            adj = torch.sparse_coo_tensor(
                indices=adj.indices(),
                values=quantized_adj_values,
                size=adj.size())   
            print(adj)         
        x = F.dropout(x, self.dropout, training=self.training)
        # adj = adj.coalesce()
        # adj_values = adj.values()
        # print(adj_values)
        x = F.relu(self.gc1(x, adj))

        if quant:
            x = quantize_activations(x, num_bits=self.nf_bit)
            adj = adj.coalesce()
            quantized_values = quantize_activations(adj.values(), num_bits=self.na_bit)
            quantized_adj_values = torch.tensor(quantized_values).clone().detach()
            adj = torch.sparse_coo_tensor(
                indices=adj.indices(),
                values=quantized_adj_values,
                size=adj.size()) 
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        if quant:
            x = quantize_activations(x, num_bits=self.nf_bit)

        return F.log_softmax(x, dim=1)