import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from torch_geometric.nn import MessagePassing
from math import sqrt
from torch import Tensor

import torch.nn.functional as F


class ExplainerMO(nn.Module):
    def __init__(self, model, args, **kwargs):
        super(ExplainerMO, self).__init__(**kwargs)   # Not just super().__init__()?

        self.args = args
        # input dims for the MLP is defined by the concatenation of the hidden layers of the GCN
        try:
            hiddens = [int(s) for s in args.hiddens.split('-')]
        except:
            hiddens =[args.hidden1]

        if args.concat:
            input_dim = sum(hiddens) * 3
        else:
            input_dim = hiddens[0] * 3
        self.device = model.device
        self.elayers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 1)
        )

        self.model = model
        self.mask_act = 'sigmoid'
        # self.label = tf.argmax(tf.cast(label,tf.float32),axis=-1)
        self.params = []

        self.softmax = nn.Softmax(dim=0)

        self.coeffs = {
            "size": args.coff_size,
            "weight_decay": args.weight_decay,
            "ent": args.coff_ent
        }

        self.init_bias = 0.0
        self.eps = 1e-7

    def __set_masks__(self, x: Tensor, edge_index: Tensor, edge_mask: Tensor = None):
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

    def _masked_adj(self,mask,adj):
        mask = mask.to(self.device)
        sym_mask = mask
        sym_mask = (sym_mask.clone() + sym_mask.clone().T) / 2

        # Create sparse tensor TODO: test and "maybe" a transpose is needed somewhere
        sparseadj = torch.sparse_coo_tensor(
            indices=torch.transpose(torch.cat([torch.unsqueeze(torch.Tensor(adj.row),-1), torch.unsqueeze(torch.Tensor(adj.col),-1)], dim=-1), 0, 1).to(torch.int64),
            values=adj.data,
            size=adj.shape
        )

        adj = sparseadj.coalesce().to_dense().to(torch.float32).to(self.device) #FIXME: tf.sparse.reorder was also applied, but probably not necessary. Maybe it needs a .coalesce() too tho?
        self.adj = adj

        masked_adj = torch.mul(adj, sym_mask)

        num_nodes = adj.shape[0]
        ones = torch.ones((num_nodes, num_nodes))
        diag_mask = ones.to(torch.float32) - torch.eye(num_nodes)
        diag_mask = diag_mask.to(self.device)
        return torch.mul(masked_adj,diag_mask)


    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""
        if training:
            bias = self.args.sample_bias
            random_noise = torch.FloatTensor(log_alpha.shape).uniform_(bias, 1.0-bias)
            random_noise = random_noise.to(self.device)
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs.clone() + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        return gate_inputs


    def forward(self,inputs,training=False):
        x, adj, nodeid, embed, tmp = inputs
        x = x.to(self.device)

        self.tmp = tmp
        if not isinstance(embed[adj.row], torch.Tensor):
            f1 = torch.tensor(embed[adj.row]).to(self.device)  # .to(self.device)  # <-- torch way to do tf.gather(embed, self.row)
            f2 = torch.tensor(embed[adj.col]).to(self.device)
        else:
            f1 = embed[adj.row].to(self.device)  # .to(self.device)  # <-- torch way to do tf.gather(embed, self.row)
            f2 = embed[adj.col].to(self.device)

        selfemb = embed[nodeid] if isinstance(embed, torch.Tensor) else torch.tensor(embed[nodeid])
        selfemb = torch.unsqueeze(selfemb, 0).repeat([f1.shape[0], 1]).to(self.device)
        f12self = torch.cat([f1, f2, selfemb], dim=-1)

        h = f12self
        h = h.to(self.device)
        for elayer in self.elayers:
            h = elayer(h)

        self.values = torch.reshape(h, [-1,])

        values = self.concrete_sample(self.values,beta=tmp,training=training)

        sparse_edge_mask = torch.sparse_coo_tensor(
            indices=torch.transpose(torch.cat([torch.unsqueeze(torch.tensor(adj.row),-1), torch.unsqueeze(torch.tensor(adj.col),-1)], dim=-1), 0, 1).to(torch.int64).to(self.device),
            values=values,
            size=adj.shape
        )

        mask = sparse_edge_mask.coalesce().to_dense().to(torch.float32)  #FIXME: again a reorder() is omitted, maybe coalesce
        masked_adj = self._masked_adj(mask,adj)

        self.mask = mask
        self.masked_adj = masked_adj

        #output = self.model((x,masked_adj))

        # modify model
        edge_index = dense_to_sparse(masked_adj)[0]
        edge_mask = masked_adj[edge_index[0], edge_index[1]]
        self.__clear_masks__()
        #factual predict
        self.__set_masks__(x, edge_index, edge_mask)
        data = Data(x=x, edge_index=edge_index)
        output, probs, masked_embeds, hidden_emb = self.model(data)
        node_pred = output[nodeid, :]
        res = self.softmax(node_pred)
        self.__clear_masks__()

        #conterfactual predict
        '''select_k = round(torch.mul(0.2, edge_index.shape[1]).item())
        selected_impedges_idx = edge_mask.sort(descending=True).indices[:select_k]
        edge_mask = edge_mask.clone()
        edge_mask[selected_impedges_idx] = 1-edge_mask[selected_impedges_idx]'''
        #edge_mask[other_notimpedges_idx] = edge_mask[other_notimpedges_idx]
        #edge_mask = 1-edge_mask
        self.__set_masks__(x, edge_index, 1-edge_mask)
        data = Data(x=x, edge_index=edge_index)
        output, probs, cf_embed, _ = self.model(data)
        cf_node_pred = output[nodeid, :]
        cf_res = self.softmax(cf_node_pred)
        self.__clear_masks__()

        return res, cf_res, masked_embeds[nodeid], hidden_emb


    def mask_topk(self, nodeid, sub_feature, sub_adj, top_k):
        x = sub_feature.to(self.args.device)
        edge_mask = self.masked_adj[sub_adj.row, sub_adj.col].detach()
        sub_edge_index = torch.tensor([sub_adj.row, sub_adj.col], dtype=torch.int64).to(self.args.device)
        select_k = round(top_k/100 * len(sub_edge_index[0]))
        selected_impedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[:select_k]        #按比例选择top_k%的重要边
        sparsity_edges = 1- len(selected_impedges_idx) / sub_edge_index.shape[1]

        delimp_edge_mask = torch.ones(len(edge_mask), dtype=torch.float32).to(self.args.device) 
        #delimp_edge_mask[selected_impedges_idx] = 1-edge_mask[selected_impedges_idx]      #重要的top_k%边置为1-mask
        #delimp_edge_mask[selected_impedges_idx] = 0.001             #重要的top_k%边置为0.001
        delimp_edge_mask[selected_impedges_idx] = 0.0   #remove important edges
        self.__clear_masks__()
        self.__set_masks__(x, sub_edge_index, delimp_edge_mask)    
        data = Data(x=x, edge_index=sub_edge_index)
        self.model.eval()
        _, maskimp_preds, embed, maskimp_h_all= self.model(data)
        maskimp_pred = maskimp_preds[nodeid]
        self.__clear_masks__()

        retainimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(self.args.device) 
        other_notimpedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[select_k:]        #按比例选择top_k%的重要边
        #retainimp_edge_mask[other_notimpedges_idx] = edge_mask[other_notimpedges_idx]      #除了重要的top_k%之外的其他边置为mask
        #retainimp_edge_mask[other_notimpedges_idx] = 0.001
        retainimp_edge_mask[other_notimpedges_idx] =  0.0   #remove not important edges
        self.__clear_masks__()
        self.__set_masks__(x, sub_edge_index, retainimp_edge_mask)    
        data = Data(x=x, edge_index=sub_edge_index)
        self.model.eval()
        _, retainimp_preds, embed, retainimp_h_all = self.model(data)
        retainimp_pred = retainimp_preds[nodeid]
        self.__clear_masks__()
        return maskimp_pred, maskimp_h_all, retainimp_pred, retainimp_h_all


    def loss(self, pred, pred_label, label, node_idx, adj_tensor=None):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        label = torch.argmax(label.clone().to(torch.float32), dim=-1)

        pred_label_node = pred_label[node_idx]
        logit = pred[pred_label_node]

        if self.args.miGroudTruth:
            gt_label_node = label[node_idx]
            logit = pred[gt_label_node]

        logit = logit + 1e-6
        pred_loss = -torch.log(logit)

        if self.args.budget<=0:
            size_loss = torch.sum(self.mask)#len(self.mask[self.mask > 0]) #
        else:
            relu = nn.ReLU()
            size_loss = relu(torch.sum(self.mask)-self.args.budget) #torch.sum(self.mask)
            
        scale=0.99
        mask = self.mask*(2*scale-1.0)+(1.0-scale)
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = torch.mean(mask_ent)

        l2norm = 0
        for name, parameter in self.elayers.named_parameters():
            if "weight" in name:
                l2norm = l2norm + torch.norm(parameter)
        l2norm = self.coeffs['weight_decay']*l2norm.clone()

        loss = pred_loss + self.coeffs["size"] * size_loss + l2norm + self.coeffs["ent"] * mask_ent_loss

        # Code block for budget constraint, not used
        # if args.budget>0 and args.coff_connect>0:
        #     # sample args.connect_sample adjacency pairs
        #     adj_tensor_dense = tf.sparse.to_dense(adj_tensor,validate_indices=False) # need to figure out
        #     noise = tf.random.uniform(adj_tensor_dense.shape,minval=0, maxval=0.001)
        #     adj_tensor_dense += noise
        #     cols = tf.argsort(adj_tensor_dense,direction='DESCENDING',axis=-1)
        #     sampled_rows = tf.expand_dims(tf.range(adj_tensor_dense.shape[0]),-1)
        #     sampled_cols_0 = tf.expand_dims(cols[:,0],-1)
        #     sampled_cols_1 = tf.expand_dims(cols[:,1],-1)
        #     sampled0 = tf.concat((sampled_rows,sampled_cols_0),-1)
        #     sampled1 = tf.concat((sampled_rows,sampled_cols_1),-1)
        #     sample0_score = tf.gather_nd(mask,sampled0)
        #     sample1_score = tf.gather_nd(mask,sampled1)
        #     connect_loss = tf.reduce_sum(-(1.0-sample0_score)*tf.math.log(1.0-sample1_score)-sample0_score*tf.math.log(sample1_score))
        #     connect_loss = connect_loss* args.coff_connect
        #     loss += connect_loss
        return loss,pred_loss,size_loss,mask_ent_loss


    def loss_ce_hidden(self, new_hidden_emb, sub_hidden_emb, pred, pred_label, label, node_idx, adj_tensor=None):
        """
        Args:
            new_hidden_emb: embedding of hidden layes by current model.
            sub_hidden_emb: embedding of hidden layers by original model.
            pred: prediction made by current model.
            pred_label: the label predicted by the original model.
        """
        label = torch.argmax(label.clone().to(torch.float32), dim=-1)

        pred_label_node = pred_label[node_idx]
        logit = pred[pred_label_node]

        if self.args.miGroudTruth:
            gt_label_node = label[node_idx]
            logit = pred[gt_label_node]

        logit = logit + 1e-6
        pred_loss = -torch.log(logit)

        hidden_loss=0
        if self.args.loss_flag == "ce_and_hidden":
            layer_num = len(new_hidden_emb)
            if self.args.hidden_layer == "alllayer":
                for i in range(layer_num):
                    #hidden_loss = hidden_loss + F.kl_div(new_hidden_emb[i].log(), sub_hidden_emb[i], reduction='sum')  #KL-divergence
                    hidden_loss = hidden_loss + sum(1-F.cosine_similarity(new_hidden_emb[i], sub_hidden_emb[i], dim=1))  #cosine similarity
            else:
                layer_index = int(self.args.hidden_layer[-1])-1
                hidden_loss = layer_num * sum(1-F.cosine_similarity(new_hidden_emb[layer_index], sub_hidden_emb[layer_index], dim=1))
        
        if self.args.budget<=0:
            size_loss = torch.sum(self.mask)#len(self.mask[self.mask > 0])
        else:
            relu = nn.ReLU()
            size_loss = relu(torch.sum(self.mask)-self.args.budget) #torch.sum(self.mask)
        
        scale=0.99
        mask = self.mask*(2*scale-1.0)+(1.0-scale)
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = torch.mean(mask_ent)

        loss =  pred_loss + hidden_loss + self.coeffs["size"] * size_loss + self.coeffs["ent"] * mask_ent_loss

        return loss, pred_loss, size_loss, mask_ent_loss, hidden_loss


    def deterministic_NeuralSort(self, s, tau=0.1, hard=False):
        """s: input elements to be sorted. 
        Shape: batch_size x n x 1
        tau: temperature for relaxation. Scalar."""
        n = s.size()[1]
        bsize = s.size()[0]
        one = torch.ones((n, 1), dtype = torch.float32).to(self.device)
        A_s = torch.abs(s - s.permute(0, 2, 1))
        B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))
        scaling = (n + 1 - 2*(torch.arange(n) + 1)).type(torch.float32).to(self.device)
        C = torch.matmul(s, scaling.unsqueeze(0))
        P_max = (C-B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / tau)
        if hard==True:
            P = torch.zeros_like(P_hat)
            b_idx = torch.arange(bsize).repeat([1, n]).view(n, bsize).transpose(dim0=1, dim1=0).flatten().type(torch.float32).to(self.device)
            r_idx = torch.arange(n).repeat([bsize, 1]).flatten().type(torch.float32).to(self.device)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            #P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P[brc_idx[0].type(torch.int32).cpu().numpy(), brc_idx[1].type(torch.int32).cpu().numpy(), brc_idx[2].type(torch.int32).cpu().numpy()] = 1
            P_hat = (P-P_hat).detach() + P_hat
        return P_hat

    def sum_KL(self, P,Q):
        return F.kl_div(P.log(), Q, reduction="sum") + F.kl_div(Q.log(), P, reduction="sum")

    def loss_ce_neuralsort(self, pred, ori_pred, label, node_idx, adj_tensor=None):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        label = torch.argmax(label.clone().to(torch.float32), dim=-1)
        pred_label_node = torch.argmax(ori_pred)
        logit = pred[pred_label_node]

        if self.args.miGroudTruth:
            gt_label_node = label[node_idx]
            logit = pred[gt_label_node]

        logit = logit + 1e-6
        ce_loss = -torch.log(logit)

        #neuralsort
        P = self.deterministic_NeuralSort(pred.unsqueeze(0).unsqueeze(-1), 0.00001)
        ori_pred_ranked = torch.matmul(P, ori_pred.unsqueeze(0).t())[0].t()[0]
        pl_loss = 0
        for i in range(len(ori_pred_ranked)):
            s = sum(ori_pred_ranked[i:])
            pl_loss = pl_loss - torch.log(ori_pred_ranked[i] / s)
        
        pred_loss = pl_loss + ce_loss

        if self.args.budget<=0:
            size_loss = self.coeffs["size"] * torch.sum(self.mask)#len(self.mask[self.mask > 0]) #
        else:
            relu = nn.ReLU()
            size_loss = self.coeffs["size"] * relu(torch.sum(self.mask)-self.args.budget) #torch.sum(self.mask)
        scale=0.99
        mask = self.mask*(2*scale-1.0)+(1.0-scale)
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        l2norm = 0
        for name, parameter in self.elayers.named_parameters():
            if "weight" in name:
                l2norm = l2norm + torch.norm(parameter)
        l2norm = self.coeffs['weight_decay']*l2norm.clone()
        loss = pred_loss +size_loss+l2norm+mask_ent_loss

        return loss, pred_loss, size_loss,mask_ent_loss,ce_loss, pl_loss


    def loss_kl_hidden(self, new_hidden_emb, sub_hidden_emb, pred, cf_pred, ori_pred, adj_tensor=None):
        """
        Args:
            pred: prediction made by current model
            ori_pred: prediction made by the original model.
        """
       #KL loss
        pred_loss = F.kl_div(pred.log(), ori_pred, reduction="sum")
        #pred_loss = sum(ori_pred * torch.log(pred))

        hidden_loss=0
        if "hidden" in self.args.loss_flag:
            layer_num = len(new_hidden_emb)
            if self.args.hidden_layer == "alllayer":
                for i in range(layer_num):
                    #hidden_loss = hidden_loss + F.kl_div(new_hidden_emb[i].log(), sub_hidden_emb[i], reduction='sum')  #KL-divergence
                    #hidden_loss = hidden_loss + sum(torch.nn.CosineSimilarity(dim=0)(new_hidden_emb[i], sub_hidden_emb[i]))
                    hidden_loss = hidden_loss + sum(1-F.cosine_similarity(new_hidden_emb[i], sub_hidden_emb[i], dim=1))  #cosine similarity
            else:
                layer_index = int(self.args.hidden_layer[-1])-1
                hidden_loss = layer_num * sum(1-F.cosine_similarity(new_hidden_emb[layer_index], sub_hidden_emb[layer_index], dim=1))
            hidden_loss = hidden_loss / layer_num
        #cf loss
        cf_loss = 0
        if "CF" in self.args.loss_flag:
            pred_label = torch.argmax(pred)
            cf_next = torch.max(torch.cat((cf_pred[:pred_label], cf_pred[pred_label+1:])))
            cf_loss = nn.ReLU()(self.args.gam + cf_pred[pred_label] - cf_next)

            kl_loss = self.sum_KL(ori_pred, cf_pred) - self.sum_KL(ori_pred, pred)
            #number = len(subgraph) - len(sub)
            #cf_loss = kl/number
            cf_loss = kl_loss

        if self.args.budget<=0:
            size_loss =torch.sum(self.mask)/len(self.mask[self.mask > 0]) #
        else:
            relu = nn.ReLU()
            size_loss = relu(torch.sum(self.mask)-self.args.budget) #torch.sum(self.mask)
        scale=0.99
        mask = self.mask*(2*scale-1.0)+(1.0-scale)
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = torch.mean(mask_ent)
        #mask_ent_loss = 0

        loss = pred_loss + hidden_loss + self.coeffs["size"] * size_loss + self.coeffs["ent"] * mask_ent_loss + cf_loss

        return loss, pred_loss, size_loss, mask_ent_loss, hidden_loss, cf_loss


    def loss_pl_hidden(self, new_hidden_emb, sub_hidden_emb, pred, cf_pred, ori_pred, adj_tensor=None):
        """
        Args:
            pred: prediction made by current model
            ori_pred: prediction made by the original model.
        """
       #PL loss
        P = self.deterministic_NeuralSort(pred.unsqueeze(0).unsqueeze(-1), 0.00001)
        ori_pred_ranked = torch.matmul(P, ori_pred.unsqueeze(0).t())[0].t()[0]
        pl_loss = 0
        for i in range(len(ori_pred_ranked)):
            s = sum(ori_pred_ranked[i:])
            pl_loss = pl_loss - torch.log(ori_pred_ranked[i] / s)

        # value loss
        pre_rp, r = torch.sort(ori_pred, descending=True)
        pred_ranked = pred[r]
        value_loss = sum(torch.abs(pred_ranked - pre_rp))
        pred_loss = pl_loss + value_loss

        hidden_loss=0
        if "hidden" in self.args.loss_flag:
            layer_num = len(new_hidden_emb)
            if self.args.hidden_layer == "alllayer":
                for i in range(layer_num):
                    #hidden_loss = hidden_loss + F.kl_div(new_hidden_emb[i].log(), sub_hidden_emb[i], reduction='sum')  #KL-divergence
                    #hidden_loss = hidden_loss + sum(torch.nn.CosineSimilarity(dim=0)(new_hidden_emb[i], sub_hidden_emb[i]))
                    hidden_loss = hidden_loss + sum(1-F.cosine_similarity(new_hidden_emb[i], sub_hidden_emb[i], dim=1))  #cosine similarity
            else:
                layer_index = int(self.args.hidden_layer[-1])-1
                hidden_loss = layer_num * sum(1-F.cosine_similarity(new_hidden_emb[layer_index], sub_hidden_emb[layer_index], dim=1))
        
        #cf loss
        cf_loss = 0
        if "cf" in self.args.loss_flag:
            pred_label = torch.argmax(pred)
            cf_next = torch.max(torch.cat((cf_pred[:pred_label], cf_pred[pred_label+1:])))
            cf_loss = nn.ReLU()(self.args.gam + cf_pred[pred_label] - cf_next)

        if self.args.budget<=0:
            size_loss =torch.sum(self.mask)#len(self.mask[self.mask > 0]) #
        else:
            relu = nn.ReLU()
            size_loss = relu(torch.sum(self.mask)-self.args.budget) #torch.sum(self.mask)
        scale=0.99
        mask = self.mask*(2*scale-1.0)+(1.0-scale)
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = torch.mean(mask_ent)

        loss = pred_loss + hidden_loss + self.coeffs["size"] * size_loss + self.coeffs["ent"] * mask_ent_loss + cf_loss

        return loss, pred_loss, size_loss, mask_ent_loss, value_loss, pl_loss, hidden_loss, cf_loss


    def loss_diff_hidden(self, new_hidden_emb, sub_hidden_emb, pred, cf_pred, ori_pred, random_cf_node_pred):
        """
        Args:
            pred: prediction made by current model
            ori_pred: prediction made by the original model.
        """
       #pred loss
        pdiff_loss = sum(torch.abs(ori_pred - pred) + self.eps)/2

        hidden_loss=0
        if "hidden" in self.args.loss_flag:
            layer_num = len(new_hidden_emb)
            if self.args.hidden_layer == "alllayer":
                for i in range(layer_num):
                    #hidden_loss = hidden_loss + F.kl_div(new_hidden_emb[i].log(), sub_hidden_emb[i], reduction='sum')  #KL-divergence
                    #hidden_loss = hidden_loss + sum(torch.nn.CosineSimilarity(dim=0)(new_hidden_emb[i], sub_hidden_emb[i]))
                    #hidden_loss = hidden_loss + sum(1-F.cosine_similarity(new_hidden_emb[i], sub_hidden_emb[i], dim=1))  #cosine similarity
                    hidden_loss = hidden_loss + (1-F.cosine_similarity(new_hidden_emb[i], sub_hidden_emb[i], dim=1)).mean()  #cosine similarity  mean
            else:
                layer_index = int(self.args.hidden_layer[-1])-1
                #hidden_loss = layer_num * sum(1-F.cosine_similarity(new_hidden_emb[layer_index], sub_hidden_emb[layer_index], dim=1))
                hidden_loss = layer_num * (1-F.cosine_similarity(new_hidden_emb[layer_index], sub_hidden_emb[layer_index], dim=1)).mean()
            hidden_loss = (hidden_loss + self.eps)/layer_num
        
        #cf loss
        cf_loss = 0
        if "CF" in self.args.loss_flag:
            #pred_label = torch.argmax(pred)
            #cf_next = torch.max(torch.cat((cf_pred[:pred_label], cf_pred[pred_label+1:])))
            #cf_loss = nn.ReLU()(self.args.gam + cf_pred[pred_label] - cf_next)
            
            #cf_loss = self.sum_KL(ori_pred, cf_pred) - self.sum_KL(ori_pred, pred)       #KL

            cf_loss = -sum(torch.abs(ori_pred - cf_pred)+ self.eps)/2
            #cf_loss = sum(torch.abs(ori_pred-random_cf_node_pred)+ self.eps)/2 - sum(torch.abs(ori_pred - cf_pred) + self.eps)/2

        if self.args.budget<=0:
            size_loss =torch.sum(self.mask)/len(self.mask[self.mask > 0]) 
        else:
            relu = nn.ReLU()
            size_loss = relu(torch.sum(self.mask)-self.args.budget)   #torch.sum(self.mask)
        
        mask_ent_loss = 0
        if "conn" in self.args.loss_flag:
            scale=0.99
            mask = self.mask*(2*scale-1.0)+(1.0-scale)
            mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
            mask_ent_loss = torch.mean(mask_ent)

        #loss = pdiff_loss + hidden_loss + self.coeffs["size"] * size_loss + self.coeffs["ent"] * mask_ent_loss + cf_loss
        loss = self.args.coff_diff * pdiff_loss  + self.args.coff_cf * cf_loss + hidden_loss + size_loss + mask_ent_loss

        return loss, pdiff_loss, size_loss, mask_ent_loss, hidden_loss, cf_loss


    def loss_cf_hidden(self, new_hidden_emb, sub_hidden_emb, pred, cf_pred, ori_pred, adj_tensor=None):
        """
        Args:
            pred: prediction made by current model
            ori_pred: prediction made by the original model.
        """
       #pred loss
        pdiff_loss = sum(torch.abs(ori_pred - pred))
        #pdiff_loss=0
        hidden_loss=0
        if "hidden" in self.args.loss_flag:
            layer_num = len(new_hidden_emb)
            if self.args.hidden_layer == "alllayer":
                for i in range(layer_num):
                    #hidden_loss = hidden_loss + F.kl_div(new_hidden_emb[i].log(), sub_hidden_emb[i], reduction='sum')  #KL-divergence
                    #hidden_loss = hidden_loss + sum(torch.nn.CosineSimilarity(dim=0)(new_hidden_emb[i], sub_hidden_emb[i]))
                    hidden_loss = hidden_loss + sum(1-F.cosine_similarity(new_hidden_emb[i], sub_hidden_emb[i], dim=1))  #cosine similarity
            else:
                layer_index = int(self.args.hidden_layer[-1])-1
                hidden_loss = layer_num * sum(1-F.cosine_similarity(new_hidden_emb[layer_index], sub_hidden_emb[layer_index], dim=1))
        hidden_loss = 0
        #cf loss
        cf_loss = 0
        if "CF" in self.args.loss_flag:
            #pred_label = torch.argmax(pred)
            #cf_next = torch.max(torch.cat((cf_pred[:pred_label], cf_pred[pred_label+1:])))
            #cf_loss = nn.ReLU()(self.args.gam + cf_pred[pred_label] - cf_next)

            #cf_loss = self.sum_KL(ori_pred, cf_pred) - self.sum_KL(ori_pred, pred)

            cf_loss = -sum(torch.abs(ori_pred - cf_pred))
        
        if self.args.budget<=0:
            size_loss =torch.sum(self.mask)#len(self.mask[self.mask > 0]) #
        else:
            relu = nn.ReLU()
            size_loss = relu(torch.sum(self.mask)-self.args.budget) #torch.sum(self.mask)
        size_loss=0
        scale=0.99
        mask = self.mask*(2*scale-1.0)+(1.0-scale)
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = torch.mean(mask_ent)
        mask_ent_loss=0
        loss = pdiff_loss + hidden_loss + self.coeffs["size"] * size_loss + self.coeffs["ent"] * mask_ent_loss + cf_loss

        return loss, pdiff_loss, size_loss, mask_ent_loss, hidden_loss, cf_loss