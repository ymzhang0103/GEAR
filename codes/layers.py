from codes.inits import *
import torch
import torch.nn as nn
from codes.fornode.config import args, dtype
import torch.nn.functional as F
import math
import scipy.stats as stats

class GraphConvolution(nn.Module):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, activation=nn.ReLU(), bias=False, device='cpu', **kwargs):
        super().__init__(**kwargs)

        self.act = activation
        self.bias = bias
        self.device = device

        self.weight = torch.zeros((input_dim, output_dim), dtype=dtype)
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
                output = torch.matmul(pre_sup, x)
            else:
                output = torch.sparse.mm(pre_sup, x)

        if self.bias:
           output = output + self.bias_weight

        if args.embnormlize:
            output = nn.functional.normalize(output, p=2, dim=-1)

        output = self.act()(output)

        return output

class Dense(nn.Module):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, activation=nn.ReLU(), bias=False, device='cpu', **kwargs):
        super().__init__(**kwargs)

        self.act = activation
        self.bias = bias
        self.device = device

        self.weight = torch.zeros((input_dim, output_dim))

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
                output = torch.matmul(pre_sup, x)
            else:
                output = torch.sparse.mm(pre_sup, x)

        if self.bias:
          output = output + self.bias_weight

        if args.embnormlize:
            output = nn.functional.normalize(output, p=2, dim=-1)

        output = self.act()(output)

        return output


class GraphConvolutionGem(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolutionGem, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = self.linear(input)
        output = torch.bmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNEdge(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        act,
        dropout=0.0,
        bias=False,
        haveweights=True,
        featureless=False,
        init='glorot',
        device = 'cpu'
    ):
        super(GCNEdge, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.haveweights = haveweights
        self.featureless = featureless
        self.bias = bias
        self.init = init
        self.device = device
        if dropout:
             self.dropout_layer = nn.Dropout(p=dropout)
        if self.init =='glorot':
            self.weight = nn.Parameter(self.glorot([input_dim, output_dim]).to(self.device))
        elif self.init =='trunc_normal':
            self.weight = nn.Parameter(self.trunc_normal([input_dim, output_dim]).to(self.device))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).to(self.device))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).to(self.device))
        else:
            self.bias = None

    def trunc_normal(self, shape, name=None, normalize=True):
        # np.random.seed(123)
        mu = 0.0
        sigma = 1.0 / math.sqrt(shape[0])
        lower, upper = mu - 2 * sigma, mu + 2 * sigma  # 截断在[μ-2σ, μ+2σ]
        X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(shape)
        initial = torch.tensor(X, dtype=torch.float)
        return torch.nn.functional.normalize(initial, 1)

    def glorot(self, shape, name=None):
        """Glorot & Bengio (AISTATS 2010) init."""
        # torch.manual_seed(123)
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = torch.Tensor(shape[0], shape[1]).uniform_(-init_range, init_range)
        return initial

    def forward(self, x, A):
        if self.dropout:
            x = self.dropout_layer(x)
        if self.haveweights:
            if not self.featureless:
                pre_sup = torch.matmul(x, self.weight)
            else:
                pre_sup = self.weight
        else:
            pre_sup=x
        y = torch.matmul(A.to(self.device), pre_sup.to(self.device))
        if self.bias is not None:
            y = y + self.bias
        y = self.act(y)
        return y, A


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none', device='cpu'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.device = device

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(self.device)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to(self.device)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

# if __name__ == "__main__":
#     test = GraphConvolution(10, 3)
#     test2 = Dense(10, 3)
