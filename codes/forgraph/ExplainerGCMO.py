import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import MessagePassing
from torch import Tensor
from math import sqrt

import torch.nn.functional as F

class ExplainerGCMO(nn.Module):
    def __init__(self, model, args, **kwargs):
        super(ExplainerGCMO, self).__init__(**kwargs)

        self.args = args
        # input dims for the MLP is defined by the concatenation of the hidden layers of the GCN
        try:
            hiddens = [int(s) for s in args.hiddens.split('-')]
        except:
            hiddens =[args.hidden1]
        
        if args.concat:
            input_dim = sum(hiddens) * 2 # or just times 3?
        else:
            input_dim = hiddens[-1] * 2
        self.device = model.device

        self.dropout = nn.Dropout(p=0.5)
        self.elayers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        #rc = torch.unsqueeze(torch.arange(0, nodesize), 0).repeat([nodesize,1]).to(torch.float32)
        # rc = torch.repeat(rc,[nodesize,1])
        #self.row = torch.reshape(rc.T,[-1]).to(self.device)
        #self.col = torch.reshape(rc,[-1]).to(self.device)
        # For masking diagonal entries
        #self.nodesize = nodesize
        self.model = model
        self.softmax = nn.Softmax(dim=-1)

        #ones = torch.ones((nodesize, nodesize))
        #self.diag_mask = ones.to(torch.float32) - torch.eye(nodesize)

        self.mask_act = 'sigmoid'
        self.init_bias = 0.0

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""

        if training:
            debug_var = 0.0
            bias = 0.0
            random_noise = bias + torch.FloatTensor(log_alpha.shape).uniform_(debug_var, 1.0-debug_var)
            random_noise = random_noise.to(self.device)
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs.clone() + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        return gate_inputs

    def __set_masks__(self, x: Tensor, edge_index: Tensor, edge_mask: Tensor = None):
        r""" Set the edge weights before message passing
        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            edge_mask (:obj:`torch.Tensor`): Edge weight matrix before message passing
              (default: :obj:`None`)

        The :attr:`edge_mask` will be randomly initialized when set to :obj:`None`.

        .. note:: When you use the :meth:`~OurExplainer.__set_masks__`,
          the explain flag for all the :class:`torch_geometric.nn.MessagePassing`
          modules in :attr:`model` will be assigned with :obj:`True`. In addition,
          the :attr:`edge_mask` will be assigned to all the modules.
          Please take :meth:`~OurExplainer.__clear_masks__` to reset.
        """
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        init_bias = self.init_bias
        #self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        #std = torch.nn.init.calculate_gain('sigmoid') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask.to(self.device)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module._explain = True
                module._edge_mask = self.edge_mask

    def __clear_masks__(self):
        """ clear the edge weights to None, and set the explain flag to :obj:`False` """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module._explain = False
                module._edge_mask = None
        #self.node_feat_masks = None
        self.edge_mask = None

        
    def forward(self, inputs, training=None):
        x, embed, adj, tmp, label = inputs
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device)
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        adj = adj.to(self.device)
        # embed = embed.to('cpu')
        #self.label = torch.argmax(label.to(torch.float32), dim=-1)
        self.label = label.to(self.device)
        self.tmp = tmp
        #row = self.row.type(torch.LongTensor).to(self.device)#('cpu')
        #col = self.col.type(torch.LongTensor).to(self.device)
        row = torch.nonzero(adj).T[0]
        col = torch.nonzero(adj).T[1]
        if not isinstance(embed[row], torch.Tensor):
            f1 = torch.Tensor(embed[row]).to(self.device)   # .to(self.device)  # <-- torch way to do tf.gather(embed, self.row)
            f2 = torch.Tensor(embed[col]).to(self.device)
        else:
            f1 = embed[row]  # .to(self.device)  # <-- torch way to do tf.gather(embed, self.row)
            f2 = embed[col]
        h = torch.cat([f1, f2], dim=-1)
        h = h.to(self.device)
        #h = self.dropout(h)
        for elayer in self.elayers:
            h = elayer(h)

        self.values = torch.reshape(h, [-1])
        values = self.concrete_sample(self.values, beta=tmp, training=training)
        nodesize = x.shape[0]
        sparsemask = torch.sparse_coo_tensor(
            indices=torch.nonzero(adj).T.to(torch.int64),
            values=values,
            size=[nodesize, nodesize]
        ).to(self.device)
        sym_mask = sparsemask.coalesce().to_dense().to(torch.float32)  #FIXME: again a reorder() is omitted, maybe coalesce
        self.mask = sym_mask

        # sym_mask = (sym_mask.clone() + sym_mask.clone().T) / 2      # Maybe needs a .clone()
        sym_mask = (sym_mask + sym_mask.T) / 2
        masked_adj = torch.mul(adj, sym_mask)
        self.masked_adj = masked_adj
        #x = torch.unsqueeze(x.detach().requires_grad_(True),0).to(torch.float32)        # Maybe needs a .clone()
        #adj = torch.unsqueeze(self.masked_adj,0).to(torch.float32)
        #x.to(self.device)
        #output = self.model((x,adj))

        # modify model
        edge_index = dense_to_sparse(masked_adj)[0]
        edge_mask = masked_adj[edge_index[0], edge_index[1]]
        self.__clear_masks__()
        #factual predict
        self.__set_masks__(x, edge_index, edge_mask)
        data = Data(x=x, edge_index=edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
        output, probs, __, emb, new_hidden_embs = self.model(data)
        res = self.softmax(output.squeeze())
        self.__clear_masks__()

        # contourfactual predict
        '''select_k = round(torch.mul(0.2, edge_index.shape[1]).item())
        selected_impedges_idx = edge_mask.sort(descending=True).indices[:select_k]
        edge_mask = edge_mask.clone()
        edge_mask[selected_impedges_idx] = 1-edge_mask[selected_impedges_idx]'''
        self.__set_masks__(x, edge_index, 1-edge_mask)
        data = Data(x=x, edge_index=edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
        cf_output, cf_probs, _, cf_emb, cf_new_hidden_embs = self.model(data)
        cf_res = self.softmax(cf_output.squeeze())
        self.__clear_masks__()

        return res, cf_res, emb, new_hidden_embs


    def loss(self, pred, label):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        #gt_label_node = self.label
        #logit = pred[gt_label_node]
        logit = pred[label]
        pred_loss = -torch.log(logit)
        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.functional.relu(self.mask)
        size_loss = self.args.coff_size * torch.sum(mask) #len(mask[mask > 0]) #torch.sum(mask)

        # entropy
        mask = mask *0.99+0.005     # maybe a .clone()
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.args.coff_ent * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        return loss


    def loss_ce_hidden(self, pred, label, new_hidden_embs, hidden_embs):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        #gt_label_node = self.label
        #logit = pred[gt_label_node]
        logit = pred[label]
        pred_loss = -torch.log(logit)

        hidden_loss=0
        if "hidden" in self.args.loss_flag:
            layer_num = len(new_hidden_embs)
            if self.args.hidden_layer == "alllayer":
                for i in range(layer_num):
                    #hidden_loss = hidden_loss + F.kl_div(new_hidden_emb[i].log(), sub_hidden_emb[i], reduction='sum')  #KL-divergence
                    #hidden_loss = hidden_loss + sum(torch.nn.CosineSimilarity(dim=0)(new_hidden_emb[i], sub_hidden_emb[i]))
                    hidden_loss = hidden_loss + sum(1-F.cosine_similarity(new_hidden_embs[i], hidden_embs[i], dim=1))  #cosine similarity
            else:
                layer_index = int(self.args.hidden_layer[-1])-1
                hidden_loss = layer_num * sum(1-F.cosine_similarity(new_hidden_embs[layer_index], hidden_embs[layer_index], dim=1))

        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.functional.relu(self.mask)
        size_loss = self.args.coff_size * torch.sum(mask) #len(mask[mask > 0]) #torch.sum(mask)

        # entropy
        mask = mask *0.99+0.005     # maybe a .clone()
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.args.coff_ent * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss + hidden_loss
        return loss, pred_loss, size_loss, mask_ent_loss, hidden_loss


    def loss_kl_hidden(self, new_pred, ori_pred, new_hidden_embs, hidden_embs):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        #KL loss
        pred_loss = F.kl_div(new_pred.log(), ori_pred, reduction="sum")
        #pred_loss = sum(ori_pred * torch.log(new_pred))

        hidden_loss=0
        if "hidden" in self.args.loss_flag:
            layer_num = len(new_hidden_embs)
            if self.args.hidden_layer == "alllayer":
                for i in range(layer_num):
                    #hidden_loss = hidden_loss + F.kl_div(new_hidden_emb[i].log(), sub_hidden_emb[i], reduction='sum')  #KL-divergence
                    #hidden_loss = hidden_loss + sum(torch.nn.CosineSimilarity(dim=0)(new_hidden_emb[i], sub_hidden_emb[i]))
                    hidden_loss = hidden_loss + sum(1-F.cosine_similarity(new_hidden_embs[i], hidden_embs[i], dim=1))  #cosine similarity
            else:
                layer_index = int(self.args.hidden_layer[-1])-1
                hidden_loss = layer_num * sum(1-F.cosine_similarity(new_hidden_embs[layer_index], hidden_embs[layer_index], dim=1))

        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.functional.relu(self.mask)
        size_loss = torch.sum(mask)/len(mask[mask > 0]) #torch.sum(mask)

        # entropy
        mask = mask *0.99+0.005     # maybe a .clone()
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = torch.mean(mask_ent)

        loss = pred_loss + self.args.coff_size * size_loss + self.args.coff_ent * mask_ent_loss + hidden_loss
        return loss, pred_loss, size_loss, mask_ent_loss, hidden_loss


    def loss_pdiff_hidden(self, new_pred, ori_pred, cf_pred, new_hidden_embs, hidden_embs):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        pdiff_loss = sum(torch.abs(ori_pred - new_pred))/2

        hidden_loss=0
        if "hidden" in self.args.loss_flag:
            layer_num = len(new_hidden_embs)
            if self.args.hidden_layer == "alllayer":
                for i in range(layer_num):
                    #hidden_loss = hidden_loss + F.kl_div(new_hidden_emb[i].log(), sub_hidden_emb[i], reduction='sum')  #KL-divergence
                    #hidden_loss = hidden_loss + sum(torch.nn.CosineSimilarity(dim=0)(new_hidden_emb[i], sub_hidden_emb[i]))
                    #hidden_loss = hidden_loss + sum(1-F.cosine_similarity(new_hidden_embs[i], hidden_embs[i], dim=1))  #cosine similarity
                    hidden_loss = hidden_loss + (1-F.cosine_similarity(new_hidden_embs[i], hidden_embs[i], dim=1)).mean()  #cosine similarity  mean
            else:
                layer_index = int(self.args.hidden_layer[-1])-1
                #hidden_loss = layer_num * sum(1-F.cosine_similarity(new_hidden_embs[layer_index], hidden_embs[layer_index], dim=1))
                hidden_loss = layer_num * (1-F.cosine_similarity(new_hidden_embs[layer_index], hidden_embs[layer_index], dim=1)).mean()
            hidden_loss = hidden_loss/layer_num
        
        #cf loss
        cf_loss = 0
        if "CF" in self.args.loss_flag:
            #pred_label = torch.argmax(pred)
            #cf_next = torch.max(torch.cat((cf_pred[:pred_label], cf_pred[pred_label+1:])))
            #cf_loss = nn.ReLU()(self.args.gam + cf_pred[pred_label] - cf_next)
            #cf_loss = self.sum_KL(ori_pred, cf_pred) - self.sum_KL(ori_pred, pred)       #KL
            cf_loss = -sum(torch.abs(ori_pred - cf_pred))/2

        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.functional.relu(self.mask)
        size_loss = torch.sum(mask)/len(mask[mask > 0]) #torch.sum(mask)

        # entropy
        mask = mask *0.99+0.005     # maybe a .clone()
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = torch.mean(mask_ent)

        #loss = pdiff_loss + self.args.coff_size * size_loss + self.args.coff_ent * mask_ent_loss + hidden_loss
        loss = self.args.coff_diff * pdiff_loss + self.args.coff_cf *  cf_loss + size_loss + mask_ent_loss + hidden_loss
        return loss, self.args.coff_diff * pdiff_loss, size_loss, mask_ent_loss, hidden_loss, self.args.coff_cf *  cf_loss


    def loss_neighbordiff_hidden(self, new_pred, ori_pred, cf_pred, new_hidden_embs, hidden_embs):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        #KL loss
        pdiff_loss = sum(torch.abs(ori_pred - new_pred))/2

        hidden_loss=0
        if "hidden" in self.args.loss_flag:
            layer_num = len(new_hidden_embs)
            if self.args.hidden_layer == "alllayer":
                for i in range(layer_num):
                    #hidden_loss = hidden_loss + F.kl_div(new_hidden_emb[i].log(), sub_hidden_emb[i], reduction='sum')  #KL-divergence
                    #hidden_loss = hidden_loss + sum(torch.nn.CosineSimilarity(dim=0)(new_hidden_emb[i], sub_hidden_emb[i]))
                    #hidden_loss = hidden_loss + sum(1-F.cosine_similarity(new_hidden_embs[i], hidden_embs[i], dim=1))  #cosine similarity
                    hidden_loss = hidden_loss + (1-F.cosine_similarity(new_hidden_embs[i], hidden_embs[i], dim=1)).mean()  #cosine similarity  mean
            else:
                layer_index = int(self.args.hidden_layer[-1])-1
                #hidden_loss = layer_num * sum(1-F.cosine_similarity(new_hidden_embs[layer_index], hidden_embs[layer_index], dim=1))
                hidden_loss = layer_num * (1-F.cosine_similarity(new_hidden_embs[layer_index], hidden_embs[layer_index], dim=1)).mean()
            hidden_loss = hidden_loss/layer_num
        
        #cf loss
        cf_loss = 0
        if "CF" in self.args.loss_flag:
            #pred_label = torch.argmax(pred)
            #cf_next = torch.max(torch.cat((cf_pred[:pred_label], cf_pred[pred_label+1:])))
            #cf_loss = nn.ReLU()(self.args.gam + cf_pred[pred_label] - cf_next)
            #cf_loss = self.sum_KL(ori_pred, cf_pred) - self.sum_KL(ori_pred, pred)       #KL
            cf_loss = -sum(torch.abs(ori_pred - cf_pred))/2

        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.functional.relu(self.mask)
        size_loss = torch.sum(mask)/len(mask[mask > 0]) #torch.sum(mask)

        # entropy
        mask = mask *0.99+0.005     # maybe a .clone()
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = torch.mean(mask_ent)

        loss = pdiff_loss + self.args.coff_size * size_loss + self.args.coff_ent * mask_ent_loss + hidden_loss
        return loss, pdiff_loss, size_loss, mask_ent_loss, hidden_loss, cf_loss
