from codes.fornode.config import *
from codes.layers import *
from codes.fornode.metricsHidden import *

import torch
import torch.nn as nn
from torch.nn import init

class GCN2(nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu', **kwargs):
        super(GCN2, self).__init__(**kwargs)
        usebias = args.bias
        self.bn = args.bn
        self.device = device

        try:
            hiddens = [int(s) for s in args.hiddens.split('-')]
        except:
            hiddens = []

        self.layers = []

        # Create first layer and append
        layer0 = GraphConvolution(input_dim=input_dim,
                                  output_dim=hiddens[0],
                                  activation=nn.ReLU,
                                  bias=usebias, device=self.device)
        self.layers.append(layer0)

        # Append all hidden layers
        for _ in range(1, len(hiddens)):

            # Append batch norm layer
            if self.bn:
                self.layers.append(nn.BatchNorm1d(hiddens[_]))

            self.layers.append(GraphConvolution(input_dim=hiddens[_-1],
                                                 output_dim=hiddens[_],
                                                 activation=nn.ReLU,
                                                 bias=usebias, device=self.device)
                                )

        self.layers_ = torch.nn.ModuleList(self.layers)

        # Create final linear layer
        self.pred_layer = nn.Linear(sum(hiddens), output_dim)

    def forward(self, inputs, training=None):
        # Run network
        emb = self.embedding(inputs, training)
        x = self.pred_layer(emb)
        return x

    def embedding(self, inputs, training=None):
        x, support = inputs
        x = x.to(self.device)
        support = support.to(self.device)

        x_all = []
        for layer in self.layers_:
            if isinstance(layer, nn.BatchNorm1d):# == type(nn.BatchNorm1d()):
                x = layer.forward(x)
            else:
                x = layer.forward((x, support), training)
                x_all.append(x)
        if args.concat:
            x = torch.cat(x_all, dim=-1)
        return x

if __name__ == '__main__':
    test = GCN2(10, 4)
