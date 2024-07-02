from config import *
import numpy as np
import pickle as pkl
import networkx as nx
from scipy.sparse import coo_matrix
import sys
import torch
from codes.dataset import MoleculeDataset, SynGraphDataset, SentiGraphDataset, BA_LRP
from torch.utils.data import random_split, Subset
from torch_geometric.data import DataLoader
from codes.load_dataset import MutagenicityDataset, NCI1Dataset
import matplotlib.pyplot as plt
from textwrap import wrap
import rdkit.Chem as Chem
from torch_geometric.datasets import MoleculeNet

def get_dataset(dataset_root, dataset_name):
    if dataset_name.lower() == 'Mutagenicity'.lower():
        return MutagenicityDataset(root=dataset_root, name= dataset_name)
    elif dataset_name.lower() == 'NCI1'.lower():
        return NCI1Dataset(root=dataset_root, name= dataset_name)
    elif dataset_name.lower() in list(MoleculeDataset.names.keys()):
        return MoleculeDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in ['graph_sst2', 'graph_sst5', 'graph_twitter']:
        return SentiGraphDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in list(SynGraphDataset.names.keys()):
        return SynGraphDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in ['ba_lrp']:
        return BA_LRP(root=dataset_root)
    else:
        raise ValueError(f"{dataset_name} is not defined.")


def get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=2):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(dataset,
                                         lengths=[num_train, num_eval, num_test],
                                         generator=torch.Generator().manual_seed(seed))

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=False)
    dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=False)
    return dataloader


def get_graph_data(dataset):
    pri = '/home/liuli/zhangym/torch_projects/datasets/'+dataset+'/raw/'+dataset+'_'

    file_edges = pri+'A.txt'
    file_edge_labels = pri+'edge_labels.txt'
    file_edge_gt = pri+'edge_gt.txt'
    file_graph_indicator = pri+'graph_indicator.txt'
    file_graph_labels = pri+'graph_labels.txt'
    file_node_labels = pri+'node_labels.txt'

    edges = np.loadtxt( file_edges,delimiter=',').astype(np.int32)
    try:
        edge_labels = np.loadtxt(file_edge_labels,delimiter=',').astype(np.int32)
    except Exception as e:
        print(e)
        print('use edge label 0')
        edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

    try:
        edge_labels_gt = np.loadtxt(file_edge_gt,delimiter=',').astype(np.int32)
    except Exception as e:
        print(e)
        print('use edge gt label 0')
        edge_labels_gt = np.zeros(edges.shape[0]).astype(np.int32)

    graph_indicator = np.loadtxt(file_graph_indicator,delimiter=',').astype(np.int32)
    graph_labels = np.loadtxt(file_graph_labels,delimiter=',').astype(np.int32)

    try:
        node_labels = np.loadtxt(file_node_labels,delimiter=',').astype(np.int32)
    except Exception as e:
        print(e)
        print('use node label 0')
        node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

    graph_id = 1
    starts = [1]
    node2graph = {}   #key=node, value= graphid
    for i in range(len(graph_indicator)):
        if graph_indicator[i]!=graph_id:
            graph_id = graph_indicator[i]
            starts.append(i+1)
        node2graph[i+1]=len(starts)-1

    graphid  = 0
    edge_lists = []
    edge_label_lists = []
    edge_list = []
    edge_label_list = []
    edge_label_gt_lists = []
    edge_label_gt_list = []
    for (s,t), l, gt in list(zip(edges,edge_labels, edge_labels_gt)):
        sgid = node2graph[s]
        tgid = node2graph[t]
        if sgid!=tgid:
            print('edges connecting different graphs, error here, please check.')
            print(s,t,'graph id',sgid,tgid)
            exit(1)
        gid = sgid
        if gid !=  graphid:
            edge_lists.append(edge_list)
            edge_label_lists.append(edge_label_list)
            edge_label_gt_lists.append(edge_label_gt_list)
            edge_list = []
            edge_label_list = []
            edge_label_gt_list = []
            graphid = gid
        start = starts[gid]
        edge_list.append((s-start,t-start))
        edge_label_list.append(l)
        edge_label_gt_list.append(gt)

    edge_lists.append(edge_list)
    edge_label_lists.append(edge_label_list)
    edge_label_gt_lists.append(edge_label_gt_list)

    # node labels
    node_label_lists = []
    graphid = 0
    node_label_list = []
    for i in range(len(node_labels)):
        nid = i+1
        gid = node2graph[nid]
        # start = starts[gid]
        if gid!=graphid:
            node_label_lists.append(node_label_list)
            graphid = gid
            node_label_list = []
        node_label_list.append(node_labels[i])
    node_label_lists.append(node_label_list)

    return edge_lists, graph_labels, edge_label_lists, edge_label_gt_lists, node_label_lists


def load_real_dataset(dataset):
    try:
        with open('./'+dataset+'.pkl','rb') as fin:
            return pkl.load(fin)
    except:
        edge_lists, graph_labels, edge_label_lists, node_label_lists = get_graph_data(dataset)

        graph_labels[graph_labels == -1] = 0

        max_node_nmb = np.max([len(node_label) for node_label in node_label_lists]) + 1  # add nodes for each graph

        edge_label_nmb = np.max([np.max(l) for l in edge_label_lists]) + 1
        node_label_nmb = np.max([np.max(l) for l in node_label_lists]) + 1

        for gid in range(len(edge_lists)):
            node_nmb = len(node_label_lists[gid])
            for nid in range(node_nmb, max_node_nmb):
                edge_lists[gid].append((nid, nid))  # add self edges
                node_label_lists[gid].append(node_label_nmb)  # the label of added node is node_label_nmb
                edge_label_lists[gid].append(edge_label_nmb)

        adjs = []
        for edge_list in edge_lists:
            row = np.array(edge_list)[:, 0]
            col = np.array(edge_list)[:, 1]
            data = np.ones(row.shape)
            adj = coo_matrix((data, (row, col))).toarray()
            if args.normadj:
                degree = np.sum(adj, axis=0, dtype=float).squeeze()
                degree[degree == 0] = 1
                sqrt_deg = np.diag(1.0 / np.sqrt(degree))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            adjs.append(np.expand_dims(adj, 0))

        labels = graph_labels

        adjs = np.concatenate(adjs, 0)
        labels = np.array(labels).astype(int)
        feas = []

        for node_label in node_label_lists:
            fea = np.zeros((len(node_label), node_label_nmb + 1))
            rows = np.arange(len(node_label))
            fea[rows, node_label] = 1
            fea = fea[:, :-1]  # remove the added node feature

            if node_label_nmb < 3:
                const_features = np.ones([fea.shape[0], 10])
                fea = np.concatenate([fea, const_features], -1)
            feas.append(fea)

        feas = np.array(feas)

        # print(max_node_nmb)

        b = np.zeros((labels.size, labels.max() + 1))
        b[np.arange(labels.size), labels] = 1
        labels = b
        with open('./'+dataset+'.pkl','wb') as fout:
            pkl.dump((adjs, feas,labels),fout)
        return adjs, feas,labels


class PlotUtils():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    '''
    def plot(self, graph, nodelist, figname, **kwargs):
        """ plot function for different dataset """
        if self.dataset_name.lower() == 'BA_2motifs'.lower():
            self.plot_ba2motifs(graph, nodelist, figname=figname)
        elif self.dataset_name.lower() in ['bbbp', 'mutag']:
            x = kwargs.get('x')
            self.plot_molecule(graph, nodelist, x, figname=figname)
        elif self.dataset_name.lower() == 'ba_shapes':
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            self.plot_bashapes(graph, nodelist, y, node_idx, figname=figname)
        elif self.dataset_name.lower() == 'ba_community':
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            self.plot_bacommunity(graph, nodelist, y, node_idx, figname=figname)
        elif self.dataset_name.lower() in ['grt_sst2_BERT_Identity'.lower()]:
            words = kwargs.get('words')
            self.plot_sentence(graph, nodelist, words=words, figname=figname)
        else:
            raise NotImplementedError
    '''
    def plot(self, graph, nodelist, figname, **kwargs):
        """ plot function for different dataset """
        if self.dataset_name.lower() in ['ba_2motifs', 'ba_lrp']:
            self.plot_ba2motifs(graph, nodelist, figname=figname)
        elif self.dataset_name.lower() in ['mutag'] + list(MoleculeNet.names.keys()):
            x = kwargs.get('x')
            self.plot_molecule(graph, nodelist, x, figname=figname)
        elif self.dataset_name.lower() in ['ba_shapes', 'ba_community', 'tree_grid', 'tree_cycle']:
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            self.plot_bashapes(graph, nodelist, y, node_idx, figname=figname)
        elif self.dataset_name.lower() in ['graph-sst2', 'graph-sst5', 'graph-twitter']:
            words = kwargs.get('words')
            self.plot_sentence(graph, nodelist, words=words, figname=figname)
        else:
            raise NotImplementedError

    def plot_new(self, graph, nodelist, edgelist, figname, **kwargs):
        """ plot function for different dataset """
        if self.dataset_name.lower() in ['ba_shapes', 'ba_community']:
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
            if self.dataset_name.lower() == 'ba_shapes':
                node_color = ['#FFA500', '#4970C6', '#FE0000', 'green']
            elif self.dataset_name.lower() == 'ba_community':
                node_color = ['#FFA500', '#4970C6', '#FE0000', 'green', '#B08B31', '#00F5FF', '#EE82EE', 'blue']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
            self.plot_subgraph_new(graph, node_idx, nodelist=None, edgelist=edgelist, colors=colors, figname=figname, subgraph_edge_color='black')
        elif self.dataset_name.lower() in ['graph_sst2', 'graph_sst5', 'graph_twitter']:
            words = kwargs.get('words')
            self.plot_sentence_new(graph, nodelist=nodelist, edgelist=edgelist, words=words, figname=figname)
        elif self.dataset_name.lower() == 'mutagenicity':
            x = kwargs.get('x')
            self.plot_mutagenicity(graph, nodelist=nodelist, edgelist=edgelist, x=x, figname=figname)
        elif self.dataset_name.lower() == 'nci1':
            x = kwargs.get('x')
            self.plot_NCI1(graph, nodelist=nodelist, edgelist=edgelist, x=x, figname=figname)
        elif self.dataset_name.lower() in ['mutag'] + list(MoleculeNet.names.keys()):
            x = kwargs.get('x')
            self.plot_molecule(graph, nodelist, x, figname=figname)
        else:
            raise NotImplementedError


    def plot_mutagenicity(self, graph, nodelist, edgelist, x, figname):
        node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
                        8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
        node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
        node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
        node_color = ['orange','red','lime','green','blue','orchid','darksalmon','darkslategray','gold','bisque','tan','lightseagreen','indigo','navy']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color='gray', arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color='black',
                               arrows=False)

        if node_labels is not None:
            nx.draw_networkx_labels(graph, pos, node_labels, font_size=14)

        plt.axis('off')
        if figname is not None:
            plt.savefig(figname)
        # plt.show()
        plt.close('all')
    

    def plot_NCI1(self, graph, nodelist, edgelist, x, figname):
        #node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
        #                8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
        node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
        #node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
        node_labels = {k: Chem.rdchem.PeriodicTable.GetElementSymbol(Chem.rdchem.GetPeriodicTable(), int(v))
                           for k, v in node_idxs.items()}
        node_color = ['white', 'green', 'maroon',  '#4970C6', 'brown', 'indigo', 'orange', 'blue', 'red', 'orchid', '#F0EA00', 'tan','lime','blue','#E49D1C','darksalmon','darkslategray','gold','bisque','lightseagreen','navy']
        #node_color = ['orange','red','lime','green','blue','orchid','darksalmon','darkslategray','gold','bisque','tan','lightseagreen','indigo','navy']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color='gray', arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color='black',
                               arrows=False)

        if node_labels is not None:
            nx.draw_networkx_labels(graph, pos, node_labels, font_size=14)

        plt.axis('off')
        if figname is not None:
            plt.savefig(figname)
        # plt.show()
        plt.close('all')


    def plot_subgraph_new(self, graph, node_idx, nodelist=None, edgelist=None, colors='#FFA500', labels=None, edge_color='gray',
                                subgraph_edge_color='black', title_sentence=None, figname=None):
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]
        if nodelist is None:
            nodelist=[]
            for (n_frm, n_to) in edgelist:
                nodelist.append(n_frm)
                nodelist.append(n_to)
            nodelist = list(set(nodelist))

        pos = nx.kamada_kawai_layout(graph) # calculate according to graph.nodes()
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        node_idx = int(node_idx)
        if isinstance(colors, list):
            list_indices = int(np.where(np.array(graph.nodes()) == node_idx)[0])
            node_idx_color = colors[list_indices]
        else:
            node_idx_color = colors
        nx.draw_networkx_nodes(graph, pos=pos,
                            nodelist=[node_idx],
                            node_color=node_idx_color,
                            node_size=600)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        plt.close('all')

    def plot_sentence_new(self, graph, nodelist, words, edgelist=None, figname=None):
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        nx.draw_networkx_nodes(graph, pos, node_color="#C4F9FF", nodelist=list(graph.nodes()), node_size=300) #D4F1EF
        nx.draw_networkx_labels(graph, pos, words_dict, font_size=14)
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_nodes(graph, pos_coalition,
                                   nodelist=nodelist,
                                   node_color='#96EEC6',  #6B8BF5, #C0C2DE
                                   node_shape='o',
                                   node_size=500)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]

        nx.draw_networkx_edges(graph, pos, width=3, edge_color='grey')
        nx.draw_networkx_edges(graph, pos=pos_coalition, edgelist=edgelist, width=3, edge_color='black')

        plt.axis('off')
        plt.title('\n'.join(wrap(' '.join(words), width=50)))
        if figname is not None:
            plt.savefig(figname)
        plt.close('all')


    def plot_subgraph(self, graph, nodelist, colors='#FFA500', labels=None, edge_color='gray',
                    edgelist=None, subgraph_edge_color='black', title_sentence=None, figname=None):

        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=6,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        # plt.show()
        plt.close('all')

    def plot_subgraph_with_nodes(self, graph, nodelist, node_idx, colors='#FFA500', labels=None, edge_color='gray',
                                edgelist=None, subgraph_edge_color='black', title_sentence=None, figname=None):
        node_idx = int(node_idx)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                                  n_frm in nodelist and n_to in nodelist]

        pos = nx.kamada_kawai_layout(graph) # calculate according to graph.nodes()
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)
        if isinstance(colors, list):
            list_indices = int(np.where(np.array(graph.nodes()) == node_idx)[0])
            node_idx_color = colors[list_indices]
        else:
            node_idx_color = colors

        nx.draw_networkx_nodes(graph, pos=pos,
                               nodelist=[node_idx],
                               node_color=node_idx_color,
                               node_size=600)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        plt.close('all')

    def plot_ba2motifs(self, graph, nodelist, edgelist=None, figname=None):
        return self.plot_subgraph(graph, nodelist, edgelist=edgelist, figname=figname)

    def plot_molecule(self, graph, nodelist, x, edgelist=None, figname=None):
        # collect the text information and node color
        if self.dataset_name.lower() == 'mutag':
            node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
            node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        elif self.dataset_name.lower() == 'bbbp':
            element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
            node_idxs = element_idxs
            node_labels = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
                           for k, v in element_idxs.items()}
            node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
            colors = [node_color[(v - 1) % len(node_color)] for k, v in node_idxs.items()]
        else:
            raise NotImplementedError

        self.plot_subgraph(graph, nodelist, colors=colors, labels=node_labels,
                           edgelist=edgelist, edge_color='gray',
                           subgraph_edge_color='black',
                           title_sentence=None, figname=figname)

    def plot_sentence(self, graph, nodelist, words, edgelist=None, figname=None):
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_nodes(graph, pos_coalition,
                                   nodelist=nodelist,
                                   node_color='yellow',
                                   node_shape='o',
                                   node_size=500)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]

        nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_size=300)
        nx.draw_networkx_labels(graph, pos, words_dict)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color='grey')
        nx.draw_networkx_edges(graph, pos=pos_coalition, edgelist=edgelist, width=3, edge_color='black')

        plt.axis('off')
        plt.title('\n'.join(wrap(' '.join(words), width=50)))
        if figname is not None:
            plt.savefig(figname)
        plt.close('all')

    def plot_bashapes(self, graph, nodelist, y, node_idx, edgelist=None, figname=None):
        node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
        node_color = ['#FFA500', '#4970C6', '#FE0000', 'green']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        self.plot_subgraph_with_nodes(graph, nodelist, node_idx, colors, edgelist=edgelist, figname=figname,
                           subgraph_edge_color='black')

    def plot_bacommunity(self, graph, nodelist, y, node_idx, edgelist=None, figname=None):
        node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
        node_color = ['#FFA500', '#4970C6', '#FE0000', 'green', '#B08B31', '#00F5FF', '#EE82EE', 'blue']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        self.plot_subgraph_with_nodes(graph, nodelist, node_idx, colors, edgelist=edgelist, figname=figname,
                           subgraph_edge_color='black')
