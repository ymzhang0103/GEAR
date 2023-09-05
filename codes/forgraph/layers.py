from inits import *
import torch
import torch.nn as nn
from config import args, dtype

class GraphConvolution(nn.Module):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, activation=nn.ReLU(), bias=False, device='cpu', **kwargs):
        super().__init__(**kwargs)

        self.act = activation
        self.bias = bias
        self.weight = torch.zeros((input_dim, output_dim), dtype=dtype)
        self.device = device

        if args.initializer == 'he':
            nn.init.kaiming_normal_(self.weight)
        else:
            nn.init.xavier_uniform_(self.weight)

        self.weight = nn.Parameter(self.weight)

        if self.bias:
            self.bias_weight = torch.zeros([output_dim], dtype=dtype)
            self.bias_weight = nn.Parameter(self.bias_weight)

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, inputs, training=None):

        x, support = inputs
        x = x.to(self.device)
        support = support.to(self.device)

        if training and args.dropout > 0:
            x = self.dropout(x)


        if args.order == 'AW':
            if isinstance(support, torch.Tensor):
                output = torch.matmul(support, x)
            else:
                output = torch.sparse.mm(support, x)
            output = torch.matmul(output, self.weight)

        else:
            pre_sup = torch.matmul(x, self.weight)
            if isinstance(support, torch.Tensor):
                output = torch.matmul(support, pre_sup)
            else:
                output = torch.sparse.mm(support, pre_sup)

        if self.bias:
            output = output + self.bias_weight

        if args.embnormlize:
            output = nn.functional.normalize(output, p=2, dim=-1)

        output = self.act()(output)

        return output

class Dense(nn.Module):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, activation=nn.ReLU(), bias=False, **kwargs):
        super().__init__(**kwargs)

        self.act = activation
        self.bias = bias

        self.weight = torch.zeros((input_dim, output_dim))

        if args.initializer == 'he':
            nn.init.kaiming_normal_(self.weight)
        else:
            nn.init.xavier_uniform_(self.weight)

        self.weight = nn.Parameter(self.weight)

        if self.bias:
            self.bias_weight = torch.zeros([output_dim], dtype=dtype)
            self.bias_weight = nn.Parameter(self.bias_weight)

        for p in self.parameters():
            p.to(self.device)


    def forward(self, inputs, training=None):
        x, support = inputs

        x = x.to(self.device)
        support = support.to(self.device)
        if training and args.dropout > 0:
            d = nn.Dropout(p=args.dropout)
            x = d(x)

        if args.order == 'AW':
            if isinstance(support, torch.Tensor):
                output = torch.matmul(support, x)
            else:
                output = torch.sparse.mm(support, x)
            output = torch.matmul(output.clone(), self.weight)

        else:
            pre_sup = torch.matmul(x, self.weight)
            if isinstance(support, torch.Tensor):
                output = torch.matmul(pre_sup, x)
            else:
                output = torch.sparse.mm(pre_sup, x)

        if self.bias:
            output = output.clone() + self.bias_weight

        if args.embnormlize:
            output = nn.functional.normalize(output, p=2, dim=-1)

        output = self.act()(output)

        return output


# if __name__ == "__main__":
#     test = GraphConvolution(10, 3)
#     test2 = Dense(10, 3)

