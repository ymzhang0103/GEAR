from config import *
from layers import *
from metrics import *

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
            hiddens =[args.hidden1]

        self.layers = []

        # Create first layer and append
        layer0 = GraphConvolution(input_dim=input_dim[1],
                                  output_dim=hiddens[0],
                                  activation=nn.ReLU,
                                  bias=usebias, device=self.device)
        self.layers.append(layer0)

        # Append all hidden layers
        for _ in range(1, len(hiddens)):

            # Append batch norm layer
            if self.bn:
                self.layers.append(nn.BatchNorm1d(input_dim[0])) #hiddens[_]))

            self.layers.append(GraphConvolution(input_dim=hiddens[_-1],
                                                 output_dim=hiddens[_],
                                                 activation=nn.ReLU,
                                                 bias=usebias, device=self.device)
                                )

        self.layers_ = torch.nn.ModuleList(self.layers)
        # Create final linear layer
        self.pred_layer = nn.Linear(input_dim[0]*2, output_dim)

        self.hiddens = hiddens

    def forward(self,inputs,training=None):
        inputs = (inputs[0].to(self.device), inputs[1].to(self.device))
        out = self.getNodeEmb(inputs,training)

        out1, _ = torch.max(out, dim=-1)
        out2 = torch.sum(out, dim=-1)

        out = torch.cat([out1, out2], dim=-1)

        out = self.pred_layer(out)
        return out

    def getNodeEmb(self, inputs, training=None):
        x, support = inputs
        x = x.to(self.device)
        support = support.to(self.device)
        x_all = []
        for layer in self.layers_:
            if isinstance(layer, nn.BatchNorm1d):
                x = layer.forward(x)
                x_all.append(x)
            else:
                x = layer.forward((x, support), training)
                if not args.bn:
                    x_all.append(x)
        if args.bn:
            x_all.append(x)
        if args.concat:
            x = torch.cat(x_all, dim=-1)
        out = x
        return out
