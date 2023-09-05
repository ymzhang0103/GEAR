from config import dtype
import torch
import torch.nn.functional as F
from typing import List, Union
import numpy as np
from torch import Tensor
from math import sqrt
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from typing import Union
from torch_geometric.utils import dense_to_sparse
from scipy.sparse import coo_matrix
import random

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    preds = preds.to(device)
    labels = labels.to(device)
    mask = mask.to(device)

    loss = F.cross_entropy(preds, torch.argmax(labels, dim=1), reduction='none')
    mask = mask.type(torch.float32)
    mask = mask / torch.mean(mask)
    loss = loss * mask
    return torch.mean(loss)

def softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    preds = preds.to(device)
    labels = labels.to(device)

    loss = F.cross_entropy(preds, torch.argmax(labels, dim=1), reduction='none')
    return torch.mean(loss)

def masked_sigmoid_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    preds = preds.to(device)
    labels = labels.to(device)
    mask = mask.to(device)

    logits = torch.tensor(preds)
    p = torch.tensor(labels)
    loss = p*-torch.log(torch.sigmoid(logits)) + (1-p)*-torch.log(1-torch.sigmoid(logits))
    mask = mask.type(torch.float32)
    mask = mask / torch.mean(mask)
    loss = loss * mask
    return torch.mean(loss)

def sigmoid_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    preds = preds.to(device)
    labels = labels.to(device)

    logits = torch.tensor(preds)
    p = torch.tensor(labels)
    loss = p*-torch.log(torch.sigmoid(logits)) + (1-p)*-torch.log(1-torch.sigmoid(logits))
    return torch.mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    preds = preds.to(device)
    labels = labels.to(device)
    mask = mask.to(device)

    correct_prediction = torch.eq(torch.argmax(preds, 1), torch.argmax(labels, 1))
    accuracy_all = correct_prediction.type(torch.float32)
    mask = mask.type(torch.float32)
    mask = mask / torch.mean(mask)
    accuracy_all = accuracy_all * mask
    return torch.mean(accuracy_all)

def accuracy(preds, labels):
    """Accuracy with masking."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    preds = preds.to(device)
    labels = labels.to(device)

    correct_prediction = torch.eq(torch.argmax(preds, 1), torch.argmax(labels, 1))
    accuracy_all = correct_prediction.type(torch.float32)
    return torch.mean(accuracy_all)

def acc(adj, mask_adj, insert, skip = 5):
        real = []
        pred = []
        adj = coo_matrix(adj)
        for r,c in list(zip(adj.row,adj.col)):
            if r>=insert and r<insert+skip and c>=insert and c<insert+skip:
                real.append(1)
            else:
                real.append(0)
            pred.append(mask_adj[r][c])
        return real, pred

def attack_graph(edge_index, edge_mask, top_k, perturb_edge_num):
    select_k = round(top_k/100 * len(edge_index[0]))
    selected_impedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[:select_k]
    sub_edge_index = edge_index[:, selected_impedges_idx]
    sub_nodes = torch.unique(torch.flatten(sub_edge_index))
    #print("edge_index",edge_index)
    attacked_edge_index = edge_index.clone()
    edge_num = edge_index.shape[1]
    #all nodes is selected
    if len(sub_nodes) == edge_index.max().item()+1:
        return attacked_edge_index
    #add edges 
    while 1==1:
        start_node = random.randint(0, attacked_edge_index.max())
        end_node = random.randint(0, attacked_edge_index.max())
        if len(sub_nodes)==0 or (start_node in sub_nodes and end_node not in sub_nodes) or (start_node not in sub_nodes and end_node in sub_nodes):
            attacked_edge_index = torch.cat((attacked_edge_index, torch.tensor([[start_node], [end_node]]).to(attacked_edge_index.device)), axis=1)
        if attacked_edge_index.shape[1]== edge_num + perturb_edge_num:
            break
    #delete edges 
    while 1==1:
        start_node = random.randint(0, attacked_edge_index.max())
        end_node = random.randint(0, attacked_edge_index.max())
        attacked_edge_index_list = list(zip(attacked_edge_index[0], attacked_edge_index[1]))
        if (start_node, end_node) in attacked_edge_index_list:
            if len(sub_nodes)==0 or (start_node in sub_nodes and end_node not in sub_nodes) or (start_node not in sub_nodes and end_node in sub_nodes):
                del_edge_idx = attacked_edge_index_list.index((start_node, end_node))
                attacked_edge_index = torch.cat((attacked_edge_index[:, :del_edge_idx], attacked_edge_index[:, del_edge_idx+1:]), axis=1)
            if attacked_edge_index.shape[1] == edge_num:
                break
    #print("attacked edge_index", edge_index)
    return attacked_edge_index

def generate_fake_graph(generator, features, edge_index, edge_mask, top_k):
    features = features.cpu()
    edge_index = edge_index.cpu()
    edge_atrr = edge_mask.unsqueeze(1).cpu()
    mu, log_var, z = generator.encode(
        x=features, in_edge_index=edge_index, 
        in_edge_attr=edge_atrr, reparameterize=1
        )
    _, _, cond_z = generator.encode(
        x=features, in_edge_index=edge_index, 
        in_edge_attr=torch.ones([edge_index.shape[1],1]), reparameterize=1
        )
    z = torch.cat([z, cond_z], dim=1)#torch.cat([z, cond_z, cond_z-z], dim=1)
    # mu, log_var, z = G.encode(x=g.x, in_edge_index=g.edge_index, in_edge_attr=g.edge_attr)

    # 2. Generate fake graphs by merging positive edges (i.e., broken_edge_index) & negative edges
    # .... pos_edge_prob for full edges (g.edge_index), neg_edge_prob for neg_edge_index
    sortededge_mask = edge_mask.sort(descending = True)
    out_edge_ratio = top_k/100
    preserve_idx = sortededge_mask.indices[: round(len(edge_mask)* out_edge_ratio)]
    broken_edge_index = edge_index[:, preserve_idx]
    broken_edge_attr = edge_atrr[preserve_idx]
    fake_edge_index, fake_edge_prob, fake_edge_attr, neg_prob2reg = \
        generator.fill_soft(
            z=z, preserved_edge_index=broken_edge_index,  preserved_edge_mask=broken_edge_attr, preserved_edge_ratio=np.array([out_edge_ratio]),
            batch=torch.zeros(features.shape[0], dtype=torch.int64, device=features.device), pos=None, neg_edge_index=None, threshold=False
            )
    return fake_edge_index, fake_edge_attr.squeeze(1)

def test_robust(indices, dataset, model, explainer, top_k, perturb_edge_num):
    attack_acc = 0
    for graphid in indices:
        node_features = dataset.data.x[dataset.slices['x'][graphid].item():dataset.slices['x'][graphid+1].item(), :]
        edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][graphid].item():dataset.slices['edge_index'][graphid+1].item()]
        data = Data(x=node_features, edge_index=edge_index, batch = torch.zeros(node_features.shape[0], dtype=torch.int64, device=node_features.device))
        logits, prob, node_embs, _, h_all = model(data)
        label = dataset.data.y[graphid]
        adj = torch.sparse_coo_tensor(indices=edge_index, values=torch.ones(edge_index.shape[1]), size=(node_features.shape[0], node_features.shape[0])).to_dense().to(edge_index.device)
        explainer.eval()
        masked_pred, cf_pred, masked_graph_emb, cf_graph_emb, masked_node_embs, cf_node_embs, new_hidden_embs  = explainer((node_features, node_embs, adj, 1.0, label))
        edge_mask = explainer.masked_adj[edge_index[0], edge_index[1]]
        attacked_edge_index = attack_graph(edge_index, edge_mask, top_k, perturb_edge_num)
        data = Data(x=node_features, edge_index=attacked_edge_index, batch = torch.zeros(node_features.shape[0], dtype=torch.int64, device=node_features.device))
        attack_logits, attack_prob, attack_node_embs, attack_graph_emb, _ = model(data)
        #print("graphid=", graphid, ", ori_prob=",prob, ", masked_prob",masked_pred,", attack_prob= ",attack_prob)
        if prob.argmax() == attack_prob.argmax():
            attack_acc = attack_acc + 1
    attack_acc = attack_acc / len(indices)
    #print("attack_acc=", attack_acc)
    return attack_acc

def rho_ndcg(dists, new_dists, hitsn):
    _, pre_r = torch.sort(dists,descending=True)
    sub_r = pre_r[:hitsn]
    dists = dists[sub_r]
    new_dists = new_dists[sub_r]
    r_dists, r_new = torch.sort(dists,descending=True)
    r_new_dists, r_mask = torch.sort(new_dists,descending=True)
    r_new = len(dists) - r_new
    r_mask = len(new_dists) - torch.IntTensor([list(r_new_dists).index(d) for d in new_dists])

    #spearman评价
    '''
    d = (r_new.cpu() - r_mask.cpu())**2
    rho = 1-(np.true_divide(np.float(sum(d)),(hitsn**3-hitsn)))
    '''
    #ndcg评价
    '''
    dcg = np.sum(np.true_divide(r_mask.cpu().detach().numpy(), np.log2(np.arange(2, r_mask.cpu().detach().numpy().size + 2))))
    dcg_pf = np.sum(np.true_divide(r_new.cpu().detach().numpy(), np.log2(np.arange(2, r_new.cpu().detach().numpy().size + 2))))
    ndcg = np.true_divide(dcg, dcg_pf)
    '''
    #print("hits@",hitsn, "ndcg",ndcg, "\ndists",dists, "\nnew_dists",new_dists, "\nr_new",r_new,"\n r_mask",r_mask)
    dcg = torch.sum(torch.divide(r_mask, torch.log2(torch.arange(2, len(r_mask) + 2).float())))
    dcg_pf = torch.sum(torch.divide(r_new.cpu(), torch.log2(torch.arange(2, len(r_new.cpu()) + 2).float())))
    ndcg = torch.divide(dcg, dcg_pf).item()
    return ndcg, r_mask, dists, new_dists

def compute_pn(exp_dict, pred_label_dict, args, model, dataset):
    pn_count = 0
    for gid, masked_adj in exp_dict.items():
        #masked_adj = torch.tensor(masked_adj)
        #graph = self.G_dataset.graphs[gid]
        if args.fix_exp:
            if len(masked_adj.flatten().sort(descending=True)[0])>args.fix_exp:
                thresh = masked_adj.flatten().sort(descending=True)[0][args.fix_exp+1]
            else:
                thresh = masked_adj.flatten().sort(descending=True)[0][-1]
        else:
            thresh = args.mask_thresh
        ps_adj = (masked_adj > thresh).float()
        
        sub_features = dataset.data.x[dataset.slices['x'][gid].item():dataset.slices['x'][gid+1].item(), :]
        sub_edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][gid].item():dataset.slices['edge_index'][gid+1].item()]
        sub_adj = torch.sparse_coo_tensor(indices=sub_edge_index, values=torch.ones(sub_edge_index.shape[1]), size=(sub_features.shape[0], sub_features.shape[0])).to_dense().to(args.device)
        pn_adj = sub_adj - ps_adj.reshape(sub_features.shape[0], sub_features.shape[0])
        sub_edge_index = torch.nonzero(pn_adj).T
        data = Data(x=sub_features.to(args.device), edge_index=sub_edge_index.to(args.device))
        _, sub_preds, _, _, _ = model(data)
        pred_label = pred_label_dict[gid]
        new_pre = sub_preds.squeeze(0)[pred_label]
        
        if new_pre < 0.5:
            pn_count += 1
    pn = pn_count / len(exp_dict.keys())
    return pn

def compute_ps(exp_dict, pred_label_dict, args, model, dataset):
    ps_count = 0
    size_dict={}
    for gid, masked_adj in exp_dict.items():
        #masked_adj = torch.tensor(masked_adj)
        if args.fix_exp:
            if len(masked_adj.flatten().sort(descending=True)[0])>args.fix_exp:
                thresh = masked_adj.flatten().sort(descending=True)[0][args.fix_exp+1]
            else:
                thresh = masked_adj.flatten().sort(descending=True)[0][-1]
        else:
            thresh = args.mask_thresh
        ps_adj = (masked_adj > thresh).float()

        size_dict[gid] = torch.nonzero(ps_adj).shape[0]/2

        sub_features = dataset.data.x[dataset.slices['x'][gid].item():dataset.slices['x'][gid+1].item(), :]
        #sub_edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][gid].item():dataset.slices['edge_index'][gid+1].item()]
        #sub_adj = torch.sparse_coo_tensor(indices=sub_edge_index, values=torch.ones(sub_edge_index.shape[1]), size=(sub_features.shape[0], sub_features.shape[0])).to_dense()
        sub_edge_index = torch.nonzero(ps_adj.reshape(sub_features.shape[0], sub_features.shape[0])).T
        data = Data(x=sub_features.to(args.device), edge_index=sub_edge_index.to(args.device))
        _, sub_preds, _, _, _ = model(data)
        pred_label = pred_label_dict[gid]
        new_pre = sub_preds.squeeze(0)[pred_label]
    
        if new_pre > 0.5:
            ps_count += 1

    ps = ps_count / len(exp_dict.keys())
    ave_size = sum(size_dict.values()) / len(size_dict.keys())
    return ps, ave_size

def compute_precision_recall(exp_dict, args, dataset):
    pres = []
    recalls = []
    f1s = []
    accs = []
    for gid, masked_adj in exp_dict.items():
        #masked_adj = torch.tensor(masked_adj)
        if args.fix_exp:
            if len(masked_adj.flatten().sort(descending=True)[0])>args.fix_exp:
                thresh = masked_adj.flatten().sort(descending=True)[0][args.fix_exp+1]
            else:
                thresh = masked_adj.flatten().sort(descending=True)[0][-1]
        else:
            thresh = args.mask_thresh
        e_labels = dataset.data.edge_label_gt[dataset.slices['edge_label_gt'][gid].item() : dataset.slices['edge_label_gt'][gid+1].item()]
        sub_edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][gid].item():dataset.slices['edge_index'][gid+1].item()]
                 
        new_edges = [masked_adj > thresh][0].cpu().numpy()
        '''old_edges = [self.G_dataset[gid][0].edata['weight'] > thresh][0].numpy()
        int_map = map(int, new_edges)
        new_edges = list(int_map)
        int_map = map(int, old_edges)
        old_edges = list(int_map)
        exp_list = np.array(new_edges)
        exp_list = new_edges.astype(int).reshape(1,-1)[0]'''
        exp_list = new_edges.astype(int)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(e_labels)):
            if exp_list[sub_edge_index[0][i], sub_edge_index[1][i]] == 1:
                if e_labels[i] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if e_labels[i] == 1:
                    FN += 1
                else:
                    TN += 1
        if TP != 0:
            pre = TP / (TP + FP)
            rec = TP / (TP + FN)
            acc = (TP + TN) / (TP + FP + TN + FN)
            f1 = 2 * pre * rec / (pre + rec)
        else:
            pre = 0
            rec = 0
            f1 = 0
            acc = (TP + TN) / (TP + FP + TN + FN)
        pres.append(pre)
        recalls.append(rec)
        f1s.append(f1)
        accs.append(acc)
    return np.mean(accs), np.mean(pres), np.mean(recalls), np.mean(f1s)

def fidelity_bak(ori_probs: torch.Tensor, maskout_probs: torch.Tensor) -> float:
    drop_probability = ori_probs - maskout_probs
    return drop_probability.mean().item()

def fidelity(ori_probs: torch.Tensor, maskout_probs: torch.Tensor) -> float:
    drop_probability = abs(ori_probs - maskout_probs)
    return drop_probability.mean().item()

def fidelity_complete(ori_probs, maskout_probs):
    drop_prob_complete = [ori_probs[i] - maskout_probs[i] for i in range(len(ori_probs))]
    result = np.mean([sum(abs(i)).item() for i in drop_prob_complete])
    return result

def fidelity_complete_bak(origin_preds, mask_out_preds, label):
    '''
    preds_label = [torch.argmax(pred) for pred in origin_preds]
    correct = sum([preds_label[i]==label[i] for i in range(len(label))]).true_divide(len(label))
    mask_out_label = [torch.argmax(pred) for pred in mask_out_preds]
    mask_out_correct = sum([mask_out_label[i]==label[i] for i in range(len(label))]).true_divide(len(label))
    return correct.item()-mask_out_correct.item()
    '''
    fidelity_label = np.mean([(origin_preds[i][label[i]] - mask_out_preds[i][label[i]]).item() for i in range(len(origin_preds))])
    fidelity_other = np.mean([sum(np.delete(mask_out_preds[i].tolist(), label[i].item()) - np.delete(origin_preds[i].tolist(), label[i].item())) for i in range(len(origin_preds))])
    return fidelity_label + fidelity_other


class XCollector_bak(object):
    def __init__(self, sparsity=None):
        self.__related_preds, self.__targets = {'zero': [], 'origin': [], 'masked': [], 'maskimp': [], 'retainimp':[], 'sparsity': [], 'masknotimp': [], 'delnotimp_sparsity': [], 'masknodes':[], 'sparsity_nodes':[], \
       'origin_l': [], 'masked_l': [], 'maskimp_l': [], 'masknotimp_l': [], 'masknodes_l':[], 'origin_ol': [], 'masked_ol': [], 'maskimp_ol': [], 'masknotimp_ol': [], 'masknodes_ol':[], 'label': [], 'origin_label': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__sparsity = sparsity
        self.__sparsity_delnotimp = sparsity
        self.__fidelity, self.__fidelity_ol, self.__fidelity_complete, self.__fidelityminus_complete, self.__fidelity_delnotimp, self.__fidelity_ol_delnotimp, self.__fidelity_complete_delnotimp, self.__simula, self.__simula_ol, self.__simula_complete, self.__fidelity_nodes, self.__fidelity_origin_nodes, self.__fidelity_complete_nodes, self.__sparsity_nodes = None, None, None, None, None, None, None, None, None, None, None, None, None, None

    @property
    def targets(self) -> list:
        return self.__targets

    def new(self):
        self.__related_preds, self.__targets = {'zero': [], 'origin': [], 'masked': [], 'maskimp': [], 'retainimp':[], 'sparsity': [], 'masknotimp': [], 'delnotimp_sparsity': [], 'masknodes':[], 'sparsity_nodes':[], \
       'origin_l': [], 'masked_l': [], 'maskimp_l': [], 'masknotimp_l': [], 'masknodes_l':[], 'origin_ol': [], 'masked_ol': [], 'maskimp_ol': [], 'masknotimp_ol': [], 'masknodes_ol':[], 'label': [], 'origin_label': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__fidelity, self.__fidelity_ol, self.__fidelity_complete, self.__fidelityminus_complete, self.__sparsity, self.__fidelity_delnotimp, self.__fidelity_ol_delnotimp, self.__fidelity_complete_delnotimp, self.__sparsity_delnotimp, self.__simula, self.__simula_ol, self.__simula_complete, self.__fidelity_nodes, self.__fidelity_origin_nodes, self.__fidelity_complete_nodes, self.__sparsity_nodes = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    def collect_data(self,
                     masks: List[Tensor],
                     related_preds: dir,
                     label: int = 0) -> None:
        r"""
        The function is used to collect related data. After collection, we can call fidelity, fidelity_inv, sparsity
        to calculate their values.

        Args:
            masks (list): It is a list of edge-level explanation for each class.
            related_preds (list): It is a list of dictionary for each class where each dictionary
            includes 4 type predicted probabilities and sparsity.
            label (int): The ground truth label. (default: 0)
        """
        if not np.isnan(label):
            for key, value in related_preds[label].items():
                self.__related_preds[key].append(value)
            for key in self.__related_preds.keys():
                if key not in related_preds[0].keys():
                    self.__related_preds[key].append(None)
            self.__targets.append(label)
            self.masks.append(masks)


    @property
    def fidelity(self):
        if self.__fidelity:
            return self.__fidelity
        elif None in self.__related_preds['maskimp_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['maskimp_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelity = fidelity(origin_preds, maskout_preds)
            return self.__fidelity

    @property
    def fidelity_origin(self):
        if self.__fidelity_ol:
            return self.__fidelity_ol
        elif None in self.__related_preds['maskimp_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['maskimp_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelity_ol = fidelity(origin_preds, maskout_preds)
            return self.__fidelity_ol

    @property
    def fidelity_complete(self):
        if self.__fidelity_complete:
            return self.__fidelity_complete
        elif None in self.__related_preds['maskimp'] or None in self.__related_preds['origin']:
            return None
        else:
            maskout_preds, origin_preds = self.__related_preds['maskimp'], self.__related_preds['origin']
            self.__fidelity_complete = fidelity_complete(origin_preds, maskout_preds)
            return self.__fidelity_complete

    @property
    def fidelityminus_complete(self):
        if self.__fidelityminus_complete:
            return self.__fidelityminus_complete
        elif None in self.__related_preds['retainimp'] or None in self.__related_preds['origin']:
            return None
        else:
            retain_preds, origin_preds = self.__related_preds['retainimp'], self.__related_preds['origin']
            self.__fidelityminus_complete = fidelity_complete(origin_preds, retain_preds)
            return self.__fidelityminus_complete

    @property
    def sparsity(self):
        r"""
        Return the Sparsity value.
        """
        if self.__sparsity:
            return self.__sparsity
        elif None in self.__related_preds['sparsity']:
            return None
        else:
            return torch.tensor(self.__related_preds['sparsity']).mean().item()

    @property
    def fidelity_delnotimp(self):
        if self.__fidelity_delnotimp:
            return self.__fidelity_delnotimp
        elif None in self.__related_preds['masknotimp_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['masknotimp_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelity_delnotimp = fidelity(origin_preds, maskout_preds)
            return self.__fidelity_delnotimp

    @property
    def fidelity_origin_delnotimp(self):
        if self.__fidelity_ol_delnotimp:
            return self.__fidelity_ol_delnotimp
        elif None in self.__related_preds['masknotimp_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['masknotimp_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelity_ol_delnotimp = fidelity(origin_preds, maskout_preds)
            return self.__fidelity_ol_delnotimp

    @property
    def fidelity_complete_delnotimp(self):
        if self.__fidelity_complete_delnotimp:
            return self.__fidelity_complete_delnotimp
        elif None in self.__related_preds['masknotimp'] or None in self.__related_preds['origin']:
            return None
        else:
            maskout_preds, origin_preds = self.__related_preds['masknotimp'], self.__related_preds['origin']
            self.__fidelity_complete_delnotimp = fidelity_complete(origin_preds, maskout_preds)
            return self.__fidelity_complete_delnotimp

    @property
    def sparsity_delnotimp(self):
        r"""
        Return the Sparsity value.
        """
        if self.__sparsity_delnotimp:
            return self.__sparsity_delnotimp
        elif None in self.__related_preds['delnotimp_sparsity']:
            return None
        else:
            return torch.tensor(self.__related_preds['delnotimp_sparsity']).mean().item()

    @property
    def simula(self):
        if self.__simula:
            return self.__simula
        elif None in self.__related_preds['masked_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            masked_preds, origin_preds = torch.tensor(self.__related_preds['masked_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__simula = fidelity(origin_preds, masked_preds)
            return self.__simula

    @property
    def simula_origin(self):
        if self.__simula_ol:
            return self.__simula_ol
        elif None in self.__related_preds['masked_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            masked_preds, origin_preds = torch.tensor(self.__related_preds['masked_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__simula_ol = fidelity(origin_preds, masked_preds)
            return self.__simula_ol

    @property
    def simula_complete(self):
        if self.__simula_complete:
            return self.__simula_complete
        elif None in self.__related_preds['masked'] or None in self.__related_preds['origin']:
            return None
        else:
            masked_preds, origin_preds= self.__related_preds['masked'], self.__related_preds['origin']
            self.__simula_complete = fidelity_complete(origin_preds, masked_preds)
            return self.__simula_complete

    @property
    def fidelity_nodes(self):
        if self.__fidelity_nodes:
            return self.__fidelity_nodes
        elif None in self.__related_preds['masknodes_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            mask_out_preds, origin_preds = torch.tensor(self.__related_preds['masknodes_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelity_nodes = fidelity(origin_preds, mask_out_preds)
            return self.__fidelity_nodes

    @property
    def fidelity_origin_nodes(self):
        if self.__fidelity_origin_nodes:
            return self.__fidelity_origin_nodes
        elif None in self.__related_preds['masknodes_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            mask_out_preds, origin_preds = torch.tensor(self.__related_preds['masknodes_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelity_origin_nodes = fidelity(origin_preds, mask_out_preds)
            return self.__fidelity_origin_nodes

    @property
    def fidelity_complete_nodes(self):
        if self.__fidelity_complete_nodes:
            return self.__fidelity_complete_nodes
        elif None in self.__related_preds['masknodes'] or None in self.__related_preds['origin']:
            return None
        else:
            mask_out_preds, origin_preds = self.__related_preds['masknodes'], self.__related_preds['origin']
            self.__fidelity_complete_nodes = fidelity_complete(origin_preds, mask_out_preds)
            return self.__fidelity_complete_nodes

    @property
    def sparsity_nodes(self):
        if self.__sparsity_nodes:
            return self.__sparsity_nodes
        elif None in self.__related_preds['sparsity_nodes']:
            return None
        else:
            return torch.tensor(self.__related_preds['sparsity_nodes']).mean().item()


class XCollector(object):
    def __init__(self, sparsity=None):
        self.__related_preds, self.__targets = {'zero': [], 'origin': [], 'masked': [], 'maskimp': [], 'masknotimp':[], 'delimp':[], 'retainimp':[], 'sparsity_edges': [], 'maskimp_nodes':[], 'retainimp_nodes':[], 'sparsity_nodes':[], \
       'origin_l': [], 'masked_l': [], 'maskimp_l': [], 'masknotimp_l':[], 'delimp_l':[], 'retainimp_l':[], 'maskimp_nodes_l':[], 'retainimp_nodes_l':[], 'origin_ol': [], 'masked_ol': [], 'maskimp_ol': [], 'masknotimp_ol':[], 'delimp_ol':[], 'retainimp_ol':[], 'maskimp_nodes_ol':[], 'retainimp_nodes_ol':[], 'label': [], 'origin_label': [], 'attack_status':[], 'fake_pred':[], 'fake_pred_l':[], 'fake_pred_ol':[]}, []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__sparsity_edges = sparsity
        self.__simula, self.__simula_origin, self.__simula_complete = None, None, None
        self.__fidelity, self.__fidelity_origin, self.__fidelity_complete, self.__fidelityminus, self.__fidelityminus_origin, self.__fidelityminus_complete= None, None, None, None, None, None
        self.__del_fidelity, self.__del_fidelity_origin, self.__del_fidelity_complete, self.__del_fidelityminus, self.__del_fidelityminus_origin, self.__del_fidelityminus_complete = None, None, None, None, None, None
        self.__fake_fidelityminus, self.__fake_fidelityminus_origin, self.__fake_fidelityminus_complete = None, None, None
        self.__fidelity_nodes, self.__fidelity_origin_nodes, self.__fidelity_complete_nodes, self.__fidelityminus_nodes, self.__fidelityminus_origin_nodes, self.__fidelityminus_complete_nodes = None, None, None, None, None, None
        self.__sparsity_nodes = None
        self.__attack_acc = None
    @property
    def targets(self) -> list:
        return self.__targets

    def new(self):
        self.__related_preds = {'zero': [], 'origin': [], 'masked': [], 'maskimp': [], 'masknotimp':[], 'delimp':[], 'retainimp':[], 'sparsity_edges': [], 'maskimp_nodes':[], 'retainimp_nodes':[], 'sparsity_nodes':[], \
       'origin_l': [], 'masked_l': [], 'maskimp_l': [], 'masknotimp_l':[], 'delimp_l':[], 'retainimp_l':[], 'maskimp_nodes_l':[], 'retainimp_nodes_l':[], 'origin_ol': [], 'masked_ol': [], 'maskimp_ol': [], 'masknotimp_ol':[], 'delimp_ol':[], 'retainimp_ol':[], 'maskimp_nodes_ol':[], 'retainimp_nodes_ol':[], 'label': [], 'origin_label': [], 'attack_status':[], 'fake_pred':[], 'fake_pred_l':[], 'fake_pred_ol':[]}
        self.__targets = []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__simula, self.__simula_origin, self.__simula_complete = None, None, None
        self.__fidelity, self.__fidelity_origin, self.__fidelity_complete, self.__fidelityminus, self.__fidelityminus_origin, self.__fidelityminus_complete= None, None, None, None, None, None
        self.__del_fidelity, self.__del_fidelity_origin, self.__del_fidelity_complete, self.__del_fidelityminus, self.__del_fidelityminus_origin, self.__del_fidelityminus_complete = None, None, None, None, None, None
        self.__fake_fidelityminus, self.__fake_fidelityminus_origin, self.__fake_fidelityminus_complete = None, None, None
        self.__fidelity_nodes, self.__fidelity_origin_nodes, self.__fidelity_complete_nodes, self.__fidelityminus_nodes, self.__fidelityminus_origin_nodes, self.__fidelityminus_complete_nodes = None, None, None, None, None, None
        self.__attack_acc = None

    def collect_data(self,
                     masks: List[Tensor],
                     related_preds: List[dict],
                     label: int = 0) -> None:
        r"""
        The function is used to collect related data. After collection, we can call fidelity, fidelity_inv, sparsity
        to calculate their values.

        Args:
            masks (list): It is a list of edge-level explanation for each class.
            related_preds (list): It is a list of dictionary for each class where each dictionary
            includes 4 type predicted probabilities and sparsity.
            label (int): The ground truth label. (default: 0)
        """
        if not np.isnan(label):
            for key, value in related_preds[label].items():
                self.__related_preds[key].append(value)
            for key in self.__related_preds.keys():
                if key not in related_preds[0].keys():
                    self.__related_preds[key].append(None)
            self.__targets.append(label)
            self.masks.append(masks)

    @property
    def attack_acc(self):
        if self.__attack_acc:
            return self.__attack_acc
        elif None in self.__related_preds['attack_status']:
            return None
        else:
            #use prop of correct label
            attack_status = torch.tensor(self.__related_preds['attack_status'])
            self.__attack_acc = attack_status.sum().item()/attack_status.shape[0]
            return self.__attack_acc

    @property
    def simula(self):
        if self.__simula:
            return self.__simula
        elif None in self.__related_preds['masked_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            masked_preds, origin_preds = torch.tensor(self.__related_preds['masked_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__simula = fidelity(origin_preds, masked_preds)
            return self.__simula

    @property
    def simula_origin(self):
        if self.__simula_origin:
            return self.__simula_origin
        elif None in self.__related_preds['masked_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            masked_preds, origin_preds = torch.tensor(self.__related_preds['masked_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__simula_origin = fidelity(origin_preds, masked_preds)
            return self.__simula_origin

    @property
    def simula_complete(self):
        if self.__simula_complete:
            return self.__simula_complete
        elif None in self.__related_preds['masked'] or None in self.__related_preds['origin']:
            return None
        else:
            masked_preds, origin_preds= self.__related_preds['masked'], self.__related_preds['origin']
            self.__simula_complete = fidelity_complete(origin_preds, masked_preds)
            return self.__simula_complete

    @property
    def fidelity(self):
        if self.__fidelity:
            return self.__fidelity
        elif None in self.__related_preds['maskimp_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['maskimp_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelity = fidelity(origin_preds, maskout_preds)
            return self.__fidelity

    @property
    def fidelity_origin(self):
        if self.__fidelity_origin:
            return self.__fidelity_origin
        elif None in self.__related_preds['maskimp_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['maskimp_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelity_origin = fidelity(origin_preds, maskout_preds)
            return self.__fidelity_origin

    @property
    def fidelity_complete(self):
        if self.__fidelity_complete:
            return self.__fidelity_complete
        elif None in self.__related_preds['maskimp'] or None in self.__related_preds['origin']:
            return None
        else:
            maskout_preds, origin_preds = self.__related_preds['maskimp'], self.__related_preds['origin']
            self.__fidelity_complete = fidelity_complete(origin_preds, maskout_preds)
            return self.__fidelity_complete

    @property
    def fidelityminus(self):
        if self.__fidelityminus:
            return self.__fidelityminus
        elif None in self.__related_preds['masknotimp_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            masknotimp_preds, origin_preds = torch.tensor(self.__related_preds['masknotimp_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelityminus = fidelity(origin_preds, masknotimp_preds)
            return self.__fidelityminus

    @property
    def fidelityminus_origin(self):
        if self.__fidelityminus_origin:
            return self.__fidelityminus_origin
        elif None in self.__related_preds['masknotimp_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            masknotimp_preds, origin_preds = torch.tensor(self.__related_preds['masknotimp_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelityminus_origin = fidelity(origin_preds, masknotimp_preds)
            return self.__fidelityminus_origin

    @property
    def fidelityminus_complete(self):
        if self.__fidelityminus_complete:
            return self.__fidelityminus_complete
        elif None in self.__related_preds['masknotimp'] or None in self.__related_preds['origin']:
            return None
        else:
            masknotimp_preds, origin_preds = self.__related_preds['masknotimp'], self.__related_preds['origin']
            self.__fidelityminus_complete = fidelity_complete(origin_preds,masknotimp_preds)
            return self.__fidelityminus_complete
    
    @property
    def del_fidelity(self):
        if self.__del_fidelity:
            return self.__del_fidelity
        elif None in self.__related_preds['delimp_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            delimp_preds, origin_preds = torch.tensor(self.__related_preds['delimp_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__del_fidelity = fidelity(origin_preds, delimp_preds)
            return self.__del_fidelity

    @property
    def del_fidelity_origin(self):
        if self.__del_fidelity_origin:
            return self.__del_fidelity_origin
        elif None in self.__related_preds['delimp_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            delimp_preds, origin_preds = torch.tensor(self.__related_preds['delimp_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__del_fidelity_origin = fidelity(origin_preds, delimp_preds)
            return self.__del_fidelity_origin

    @property
    def del_fidelity_complete(self):
        if self.__del_fidelity_complete:
            return self.__del_fidelity_complete
        elif None in self.__related_preds['delimp'] or None in self.__related_preds['origin']:
            return None
        else:
            delimp_preds, origin_preds = self.__related_preds['delimp'], self.__related_preds['origin']
            self.__del_fidelity_complete = fidelity_complete(origin_preds, delimp_preds)
            return self.__del_fidelity_complete

    @property
    def del_fidelityminus(self):
        if self.__del_fidelityminus:
            return self.__del_fidelityminus
        elif None in self.__related_preds['retainimp_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            retainimp_preds, origin_preds = torch.tensor(self.__related_preds['retainimp_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__del_fidelityminus = fidelity(origin_preds, retainimp_preds)
            return self.__del_fidelityminus

    @property
    def del_fidelityminus_origin(self):
        if self.__del_fidelityminus_origin:
            return self.__del_fidelityminus_origin
        elif None in self.__related_preds['retainimp_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            retainimp_preds, origin_preds = torch.tensor(self.__related_preds['retainimp_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__del_fidelityminus_origin = fidelity(origin_preds, retainimp_preds)
            return self.__del_fidelityminus_origin

    @property
    def del_fidelityminus_complete(self):
        if self.__del_fidelityminus_complete:
            return self.__del_fidelityminus_complete
        elif None in self.__related_preds['retainimp'] or None in self.__related_preds['origin']:
            return None
        else:
            retainimp_preds, origin_preds = self.__related_preds['retainimp'], self.__related_preds['origin']
            self.__del_fidelityminus_complete = fidelity_complete(origin_preds, retainimp_preds)
            return self.__del_fidelityminus_complete

    @property
    def fake_fidelityminus(self):
        if self.__fake_fidelityminus:
            return self.__fake_fidelityminus
        elif None in self.__related_preds['fake_pred_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            fake_preds, origin_preds = torch.tensor(self.__related_preds['fake_pred_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fake_fidelityminus = fidelity(origin_preds, fake_preds)
            return self.__fake_fidelityminus

    @property
    def fake_fidelityminus_origin(self):
        if self.__fake_fidelityminus_origin:
            return self.__fake_fidelityminus_origin
        elif None in self.__related_preds['fake_pred_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            fake_preds, origin_preds = torch.tensor(self.__related_preds['fake_pred_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fake_fidelityminus_origin = fidelity(origin_preds, fake_preds)
            return self.__fake_fidelityminus_origin

    @property
    def fake_fidelityminus_complete(self):
        if self.__fake_fidelityminus_complete:
            return self.__fake_fidelityminus_complete
        elif None in self.__related_preds['fake_pred'] or None in self.__related_preds['origin']:
            return None
        else:
            fake_preds, origin_preds = self.__related_preds['fake_pred'], self.__related_preds['origin']
            self.__fake_fidelityminus_complete = fidelity_complete(origin_preds, fake_preds)
            return self.__fake_fidelityminus_complete

    @property
    def sparsity_edges(self):
        r"""
        Return the Sparsity value.
        """
        if self.__sparsity_edges:
            return self.__sparsity_edges
        elif None in self.__related_preds['sparsity_edges']:
            return None
        else:
            return torch.tensor(self.__related_preds['sparsity_edges']).mean().item()

    @property
    def fidelity_nodes(self):
        if self.__fidelity_nodes:
            return self.__fidelity_nodes
        elif None in self.__related_preds['maskimp_nodes_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['maskimp_nodes_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelity_nodes = fidelity(origin_preds, maskout_preds)
            return self.__fidelity_nodes

    @property
    def fidelity_origin_nodes(self):
        if self.__fidelity_origin_nodes:
            return self.__fidelity_origin_nodes
        elif None in self.__related_preds['maskimp_nodes_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            maskout_preds, origin_preds = torch.tensor(self.__related_preds['maskimp_nodes_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelity_origin_nodes = fidelity(origin_preds, maskout_preds)
            return self.__fidelity_origin_nodes

    @property
    def fidelity_complete_nodes(self):
        if self.__fidelity_complete_nodes:
            return self.__fidelity_complete_nodes
        elif None in self.__related_preds['maskimp_nodes'] or None in self.__related_preds['origin']:
            return None
        else:
            maskout_preds, origin_preds = self.__related_preds['maskimp_nodes'], self.__related_preds['origin']
            self.__fidelity_complete_nodes = fidelity_complete(origin_preds, maskout_preds)
            return self.__fidelity_complete_nodes
    
    @property
    def fidelityminus_nodes(self):
        if self.__fidelityminus_nodes:
            return self.__fidelityminus_nodes
        elif None in self.__related_preds['retainimp_nodes_l'] or None in self.__related_preds['origin_l']:
            return None
        else:
            #use prop of correct label
            retainimp_preds, origin_preds = torch.tensor(self.__related_preds['retainimp_nodes_l']), torch.tensor(self.__related_preds['origin_l'])
            self.__fidelityminus_nodes = fidelity(origin_preds, retainimp_preds)
            return self.__fidelityminus_nodes

    @property
    def fidelityminus_origin_nodes(self):
        if self.__fidelityminus_origin_nodes:
            return self.__fidelityminus_origin_nodes
        elif None in self.__related_preds['retainimp_nodes_ol'] or None in self.__related_preds['origin_ol']:
            return None
        else:
            # use prop of origin label
            retainimp_preds, origin_preds = torch.tensor(self.__related_preds['retainimp_nodes_ol']), torch.tensor(self.__related_preds['origin_ol'])
            self.__fidelityminus_origin_nodes = fidelity(origin_preds, retainimp_preds)
            return self.__fidelityminus_origin_nodes

    @property
    def fidelityminus_complete_nodes(self):
        if self.__fidelityminus_complete_nodes:
            return self.__fidelityminus_complete_nodes
        elif None in self.__related_preds['retainimp_nodes'] or None in self.__related_preds['origin']:
            return None
        else:
            retainimp_preds, origin_preds = self.__related_preds['retainimp_nodes'], self.__related_preds['origin']
            self.__fidelityminus_complete_nodes = fidelity_complete(origin_preds, retainimp_preds)
            return self.__fidelityminus_complete_nodes
     
    @property
    def sparsity_nodes(self):
        if self.__sparsity_nodes:
            return self.__sparsity_nodes
        elif None in self.__related_preds['sparsity_nodes']:
            return None
        else:
            return torch.tensor(self.__related_preds['sparsity_nodes']).mean().item()



class MaskoutMetric:
    def __init__(
        self,
        model, 
        prog_args
    ):
        self.model = model
        self.model.eval()
        self.prog_args= prog_args

    def GnnNets_NC2value_func_new(self, gnnNets_NC, node_idx):
        def value_func(data):
            with torch.no_grad():
                logits, probs, _, _ = gnnNets_NC(data=data)
                # select the corresponding node prob through the node idx on all the sampling graphs
                batch_size = data.batch.max() + 1
                probs = probs.reshape(batch_size, -1, probs.shape[-1])
                scores = probs[:, node_idx]
                return scores
        return value_func

    def GnnNets_GC2value_func_new(self, gnnNets):
        def value_func(batch):
            with torch.no_grad():
                logits, probs, _, _, _ = gnnNets(data=batch)
                #probs = F.softmax(logits, dim=-1)
                score = probs.squeeze()
            return score
        return value_func

    def gnn_prob(self, coalition: list, data: Data, value_func: str, subgraph_building_method='zero_filling') -> torch.Tensor:
        """ the value of subgraph with selected nodes """
        num_nodes = data.num_nodes
        subgraph_build_func = self.get_graph_build_func(subgraph_building_method)
        mask = torch.zeros(num_nodes).type(torch.float32).to(data.x.device)
        mask[coalition] = 1.0
        ret_x, ret_edge_index = subgraph_build_func(data.x, data.edge_index, mask)
        mask_data = Data(x=ret_x, edge_index=ret_edge_index)
        mask_data = Batch.from_data_list([mask_data])
        score = value_func(mask_data)
        # get the score of predicted class for graph or specific node idx
        return score

    def get_graph_build_func(self, build_method):
        if build_method.lower() == 'zero_filling':
            return self.graph_build_zero_filling
        elif build_method.lower() == 'split':
            return self.graph_build_split
        else:
            raise NotImplementedError

    def graph_build_zero_filling(self, X, edge_index, node_mask: np.array):
        """ subgraph building through masking the unselected nodes with zero features """
        ret_X = X * node_mask.unsqueeze(1)
        return ret_X, edge_index

    def graph_build_split(X, edge_index, node_mask: np.array):
        """ subgraph building through spliting the selected nodes from the original graph """
        row, col = edge_index
        edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
        ret_edge_index = edge_index[:, edge_mask]
        return X, ret_edge_index

    def calculate_selected_nodes(self, edge_index, edge_mask, top_k):
        threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0]-1)])
        hard_mask = (edge_mask > threshold).cpu()
        edge_idx_list = torch.where(hard_mask == 1)[0]
        selected_nodes = []
        #edge_index = data.edge_index.cpu().numpy()
        for edge_idx in edge_idx_list:
            selected_nodes += [edge_index[0][edge_idx].item(), edge_index[1][edge_idx].item()]
        selected_nodes = list(set(selected_nodes))
        return selected_nodes

    def __set_masks__(self, x: Tensor, edge_index: Tensor, edge_mask: Tensor = None):
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        init_bias = 0.0
        #self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        #std = torch.nn.init.calculate_gain('sigmoid') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

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

    def evaluate_adj_new(self, node_idx_new, features, adj):
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.prog_args.device)
            sub_edge_index = torch.nonzero(adj).t()
            new_edge_mask = adj[sub_edge_index[0], sub_edge_index[1]]
            self.__clear_masks__()
            self.__set_masks__(x, sub_edge_index, new_edge_mask) 
            data = Data(x=features, edge_index=sub_edge_index)
            logit, pred, _ = self.model(data.to(self.prog_args.device))
            _, pred_label = torch.max(pred, 1)
            self.__clear_masks__()
        return pred_label[node_idx_new], pred[node_idx_new]

    def evaluate_adj_new_GC(self, features, adj):
        with torch.no_grad():
            sub_edge_index = torch.nonzero(adj).t()
            new_edge_mask = adj[sub_edge_index[0], sub_edge_index[1]]
            self.__clear_masks__()
            self.__set_masks__(features, sub_edge_index, new_edge_mask) 
            data = Data(x=features, edge_index=sub_edge_index, batch = torch.zeros(features.shape[0], dtype=torch.int64, device=features.device))
            _, pred, _ = self.model(data)
            _, pred_label = torch.max(pred, 1)
            self.__clear_masks__()
        return pred_label, pred[0]

    def predict(self, features, edge_index, edge_mask):
        self.__clear_masks__()
        self.__set_masks__(features, edge_index, edge_mask)    
        data = Data(x=features, edge_index=edge_index, batch = torch.zeros(features.shape[0], dtype=torch.int64, device=features.device))
        self.model.eval()
        _, mask_preds, embed, _, _ = self.model(data)
        mask_pred = mask_preds.squeeze()
        self.__clear_masks__()
        return mask_pred

    '''def evaluate_adj_new(self, node_idx_new, features, adj):
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.prog_args.device)
            sub_edge_index = torch.nonzero(adj).t()
            new_edge_mask = adj[sub_edge_index[0], sub_edge_index[1]]
            data = Data(x=features, edge_index=sub_edge_index, edge_attr=new_edge_mask)
            logit, pred, _ = self.model(data.to(self.prog_args.device))
            _, pred_label = torch.max(pred, 1)
        return pred_label[node_idx_new], pred[node_idx_new]
    
    def evaluate_adj_new_GC(self, features, adj):
        with torch.no_grad():
            sub_edge_index = torch.nonzero(adj).t()
            new_edge_mask = adj[sub_edge_index[0], sub_edge_index[1]]
            data = Data(x=features, edge_index=sub_edge_index, edge_attr=new_edge_mask, batch = torch.zeros(features.shape[0], dtype=torch.int64, device=features.device))
            _, pred, _ = self.model(data)
            _, pred_label = torch.max(pred, 1)
        return pred_label, pred[0]

    def predict(self, features, edge_index, edge_mask):
        data = Data(x=features, edge_index=edge_index, edge_attr=edge_mask, batch = torch.zeros(features.shape[0], dtype=torch.int64, device=features.device))
        self.model.eval()
        _, mask_preds, embed, _, _ = self.model(data)
        mask_pred = mask_preds.squeeze()
        return mask_pred'''

    def metric_del_edges_GC(self, topk_arr, sub_feature, mask, sub_edge_index, origin_pred, masked_pred, label):
        origin_label = torch.argmax(origin_pred)
        edge_mask = mask[sub_edge_index[0], sub_edge_index[1]]

        x = sub_feature.to(self.prog_args.device)
        related_preds_dict = dict()
        attack_status = 0
        for top_k in topk_arr:
            select_k = round(top_k/100 * len(sub_edge_index[0]))

            selected_impedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[:select_k]#按比例选择top_k%的重要边
            other_notimpedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[select_k:]        #按比例选择top_k%的重要边
            sparsity_edges = 1- len(selected_impedges_idx) / sub_edge_index.shape[1]

            maskimp_edge_mask = torch.ones(len(edge_mask)).to(self.prog_args.device) 
            maskimp_edge_mask[selected_impedges_idx] = 1-edge_mask[selected_impedges_idx]#重要的边，权重置为1-mask
            maskimp_pred = self.predict(x, sub_edge_index, maskimp_edge_mask)

            masknotimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(self.prog_args.device) 
            masknotimp_edge_mask[other_notimpedges_idx] = edge_mask[other_notimpedges_idx]      #除了重要的top_k%之外的其他边置为mask
            masknotimp_pred = self.predict(x, sub_edge_index, masknotimp_edge_mask)

            delimp_edge_mask = torch.ones(len(edge_mask)).to(self.prog_args.device) 
            delimp_edge_mask[selected_impedges_idx] = 0.0    #remove important edges
            delimp_pred = self.predict(x, sub_edge_index, delimp_edge_mask)

            retainimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(self.prog_args.device) 
            retainimp_edge_mask[other_notimpedges_idx] = 0.0   #remove not important edges
            retainimp_pred = self.predict(x, sub_edge_index, retainimp_edge_mask)

            #delete nodes
            selected_nodes = self.calculate_selected_nodes(sub_edge_index, edge_mask, select_k)
            maskout_nodes_list = [node for node in range(sub_feature.shape[0]) if node not in selected_nodes]
            value_func = self.GnnNets_GC2value_func_new(self.model)
            data = Data(x=sub_feature, edge_index=sub_edge_index, edge_attr=edge_mask, batch = torch.zeros(sub_feature.shape[0], dtype=torch.int64, device=sub_feature.device))
            maskimp_pred_nodes = self.gnn_prob(maskout_nodes_list, data, value_func, subgraph_building_method='zero_filling')
            retainimp_pred_nodes = self.gnn_prob(selected_nodes, data, value_func, subgraph_building_method='zero_filling')
            sparsity_nodes = 1 - len(selected_nodes) / sub_feature.shape[0]

            related_preds = [{
                'label': label,
                'origin_label': origin_label,
                'origin': origin_pred,
                'origin_l': origin_pred[label],
                'origin_ol': origin_pred[origin_label],
                'masked': masked_pred,
                'masked_l': masked_pred[label],
                'masked_ol': masked_pred[origin_label],
                'maskimp': maskimp_pred,
                'maskimp_l': maskimp_pred[label],
                'maskimp_ol': maskimp_pred[origin_label],
                'masknotimp': masknotimp_pred,
                'masknotimp_l': masknotimp_pred[label],
                'masknotimp_ol': masknotimp_pred[origin_label],
                'delimp':delimp_pred,
                'delimp_l':delimp_pred[label],
                'delimp_ol':delimp_pred[origin_label],
                'retainimp':retainimp_pred,
                'retainimp_l':retainimp_pred[label],
                'retainimp_ol':retainimp_pred[origin_label],
                'sparsity_edges': sparsity_edges,
                'maskimp_nodes': maskimp_pred_nodes,
                'maskimp_nodes_l':maskimp_pred_nodes[label],
                'maskimp_nodes_ol':maskimp_pred_nodes[origin_label],
                'retainimp_nodes': retainimp_pred_nodes,
                'retainimp_nodes_l':retainimp_pred_nodes[label],
                'retainimp_nodes_ol':retainimp_pred_nodes[origin_label],
                'sparsity_nodes': sparsity_nodes
            }]
            related_preds_dict[top_k] = related_preds

        pred_mask = [edge_mask.cpu().detach().numpy()]
        return pred_mask, related_preds_dict
    
    def metric_del_edges_GC_distribution(self, topk_arr, sub_feature, mask, sub_edge_index, origin_pred, masked_pred, label):
        origin_label = torch.argmax(origin_pred)
        edge_mask = mask[sub_edge_index[0], sub_edge_index[1]]

        x = sub_feature.to(self.prog_args.device)
        related_preds_dict = dict()
        attack_status = 0
        for top_k in topk_arr:
            select_k = round(top_k/100 * len(sub_edge_index[0]))

            selected_impedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[:select_k]#按比例选择top_k%的重要边
            other_notimpedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[select_k:]        #按比例选择top_k%的重要边
            sparsity_edges = 1- len(selected_impedges_idx) / sub_edge_index.shape[1]

            maskimp_edge_mask = torch.ones(len(edge_mask)).to(self.prog_args.device) 
            maskimp_edge_mask[selected_impedges_idx] = 1-edge_mask[selected_impedges_idx]#重要的边，权重置为1-mask
            maskimp_pred = self.predict(x, sub_edge_index, maskimp_edge_mask)

            masknotimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(self.prog_args.device) 
            masknotimp_edge_mask[other_notimpedges_idx] = edge_mask[other_notimpedges_idx]      #除了重要的top_k%之外的其他边置为mask
            masknotimp_pred = self.predict(x, sub_edge_index, masknotimp_edge_mask)

            delimp_edge_mask = torch.ones(len(edge_mask)).to(self.prog_args.device) 
            delimp_edge_mask[selected_impedges_idx] = 0.0    #remove important edges
            delimp_pred = self.predict(x, sub_edge_index, delimp_edge_mask)

            retainimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(self.prog_args.device) 
            retainimp_edge_mask[other_notimpedges_idx] = 0.0   #remove not important edges
            retainimp_pred = self.predict(x, sub_edge_index, delimp_edge_mask)

            #delete nodes
            selected_nodes = self.calculate_selected_nodes(sub_edge_index, edge_mask, select_k)
            maskout_nodes_list = [node for node in range(sub_feature.shape[0]) if node not in selected_nodes]
            value_func = self.GnnNets_GC2value_func_new(self.model)
            data = Data(x=sub_feature, edge_index=sub_edge_index, edge_attr=edge_mask, batch = torch.zeros(sub_feature.shape[0], dtype=torch.int64, device=sub_feature.device))
            maskimp_pred_nodes = self.gnn_prob(maskout_nodes_list, data, value_func, subgraph_building_method='zero_filling')
            retainimp_pred_nodes = self.gnn_prob(selected_nodes, data, value_func, subgraph_building_method='zero_filling')
            sparsity_nodes = 1 - len(selected_nodes) / sub_feature.shape[0]

            #attack_acc
            attacked_edge_index = attack_graph(sub_edge_index, edge_mask, top_k, self.prog_args.perturb_edge_num)
            data = Data(x=x, edge_index=attacked_edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
            attack_logits, attack_prob, attack_node_embs, attack_graph_emb, _ = self.model(data)
            if origin_label == attack_prob.argmax():
                attack_status = 1
            else:
                attack_status = 0

            related_preds = [{
                'label': label,
                'origin_label': origin_label,
                'origin': origin_pred,
                'origin_l': origin_pred[label],
                'origin_ol': origin_pred[origin_label],
                'masked': masked_pred,
                'masked_l': masked_pred[label],
                'masked_ol': masked_pred[origin_label],
                'maskimp': maskimp_pred,
                'maskimp_l': maskimp_pred[label],
                'maskimp_ol': maskimp_pred[origin_label],
                'masknotimp': masknotimp_pred,
                'masknotimp_l': masknotimp_pred[label],
                'masknotimp_ol': masknotimp_pred[origin_label],
                'delimp':delimp_pred,
                'delimp_l':delimp_pred[label],
                'delimp_ol':delimp_pred[origin_label],
                'retainimp':retainimp_pred,
                'retainimp_l':retainimp_pred[label],
                'retainimp_ol':retainimp_pred[origin_label],
                'sparsity_edges': sparsity_edges,
                'maskimp_nodes': maskimp_pred_nodes,
                'maskimp_nodes_l':maskimp_pred_nodes[label],
                'maskimp_nodes_ol':maskimp_pred_nodes[origin_label],
                'retainimp_nodes': retainimp_pred_nodes,
                'retainimp_nodes_l':retainimp_pred_nodes[label],
                'retainimp_nodes_ol':retainimp_pred_nodes[origin_label],
                'sparsity_nodes': sparsity_nodes,
                'attack_status': attack_status,
            }]
            related_preds_dict[top_k] = related_preds

        pred_mask = [edge_mask.cpu().detach().numpy()]
        return pred_mask, related_preds_dict


    def metric_del_edges_fakeGC(self, generator, topk_arr, sub_feature, mask, sub_edge_index, origin_pred, masked_pred, label):
        origin_label = torch.argmax(origin_pred)
        edge_mask = mask[sub_edge_index[0], sub_edge_index[1]]

        x = sub_feature.to(self.prog_args.device)
        related_preds_dict = dict()
        attack_status = 0
        for top_k in topk_arr:
            select_k = round(top_k/100 * len(sub_edge_index[0]))

            selected_impedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[:select_k]#按比例选择top_k%的重要边
            other_notimpedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[select_k:]        #按比例选择top_k%的重要边
            sparsity_edges = 1- len(selected_impedges_idx) / sub_edge_index.shape[1]

            maskimp_edge_mask = torch.ones(len(edge_mask)).to(self.prog_args.device) 
            maskimp_edge_mask[selected_impedges_idx] = 1-edge_mask[selected_impedges_idx]#重要的边，权重置为1-mask
            maskimp_pred = self.predict(x, sub_edge_index, maskimp_edge_mask)

            masknotimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(self.prog_args.device) 
            masknotimp_edge_mask[other_notimpedges_idx] = edge_mask[other_notimpedges_idx]      #除了重要的top_k%之外的其他边置为mask
            masknotimp_pred = self.predict(x, sub_edge_index, masknotimp_edge_mask)

            delimp_edge_mask = torch.ones(len(edge_mask)).to(self.prog_args.device) 
            delimp_edge_mask[selected_impedges_idx] = 0.0    #remove important edges
            delimp_pred = self.predict(x, sub_edge_index, delimp_edge_mask)

            retainimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(self.prog_args.device) 
            retainimp_edge_mask[other_notimpedges_idx] = 0.0   #remove not important edges
            retainimp_pred = self.predict(x, sub_edge_index, delimp_edge_mask)

            #delete nodes
            selected_nodes = self.calculate_selected_nodes(sub_edge_index, edge_mask, select_k)
            maskout_nodes_list = [node for node in range(sub_feature.shape[0]) if node not in selected_nodes]
            value_func = self.GnnNets_GC2value_func_new(self.model)
            data = Data(x=sub_feature, edge_index=sub_edge_index, edge_attr=edge_mask, batch = torch.zeros(sub_feature.shape[0], dtype=torch.int64, device=sub_feature.device))
            maskimp_pred_nodes = self.gnn_prob(maskout_nodes_list, data, value_func, subgraph_building_method='zero_filling')
            retainimp_pred_nodes = self.gnn_prob(selected_nodes, data, value_func, subgraph_building_method='zero_filling')
            sparsity_nodes = 1 - len(selected_nodes) / sub_feature.shape[0]

            #attack_acc
            attacked_edge_index = attack_graph(sub_edge_index, edge_mask, top_k, self.prog_args.perturb_edge_num)
            data = Data(x=x, edge_index=attacked_edge_index, batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device))
            attack_logits, attack_prob, attack_node_embs, attack_graph_emb, _ = self.model(data)
            if origin_label == attack_prob.argmax():
                attack_status = 1
            else:
                attack_status = 0

            #fake_graph
            fake_edge_index, fake_edge_atrr = generate_fake_graph(generator, x, sub_edge_index, edge_mask, top_k)
            fake_edge_index = fake_edge_index.to(self.prog_args.device)
            fake_edge_atrr = fake_edge_atrr.to(self.prog_args.device)
            fake_pred = self.predict(x, sub_edge_index, fake_edge_atrr)
            
            related_preds = [{
                'label': label,
                'origin_label': origin_label,
                'origin': origin_pred,
                'origin_l': origin_pred[label],
                'origin_ol': origin_pred[origin_label],
                'masked': masked_pred,
                'masked_l': masked_pred[label],
                'masked_ol': masked_pred[origin_label],
                'maskimp': maskimp_pred,
                'maskimp_l': maskimp_pred[label],
                'maskimp_ol': maskimp_pred[origin_label],
                'masknotimp': masknotimp_pred,
                'masknotimp_l': masknotimp_pred[label],
                'masknotimp_ol': masknotimp_pred[origin_label],
                'delimp':delimp_pred,
                'delimp_l':delimp_pred[label],
                'delimp_ol':delimp_pred[origin_label],
                'retainimp':retainimp_pred,
                'retainimp_l':retainimp_pred[label],
                'retainimp_ol':retainimp_pred[origin_label],
                'sparsity_edges': sparsity_edges,
                'maskimp_nodes': maskimp_pred_nodes,
                'maskimp_nodes_l':maskimp_pred_nodes[label],
                'maskimp_nodes_ol':maskimp_pred_nodes[origin_label],
                'retainimp_nodes': retainimp_pred_nodes,
                'retainimp_nodes_l':retainimp_pred_nodes[label],
                'retainimp_nodes_ol':retainimp_pred_nodes[origin_label],
                'sparsity_nodes': sparsity_nodes,
                'attack_status': attack_status,
                'fake_pred': fake_pred,
                'fake_pred_l': fake_pred[label],
                'fake_pred_ol': fake_pred[origin_label],
            }]
            related_preds_dict[top_k] = related_preds

        pred_mask = [edge_mask.cpu().detach().numpy()]
        return pred_mask, related_preds_dict