import os
import glob
import json
import torch
import pickle
import numpy as np
import os.path as osp
import sys
sys.path.append('./codes')
from Configures import data_args
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import random_split, Subset
import traceback
import scipy.sparse as sp
from torch_geometric.data.dataset import Dataset
from torch_geometric.datasets import Planetoid, Actor, Amazon


def undirected_graph(data):
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0),
                                 data.edge_index], dim=1)
    return data


def split(data, batch):
    # i-th contains elements from slice[i] to slice[i+1]
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()

    slices = dict()
    slices['x'] = node_slice
    slices['edge_index'] = edge_slice
    slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices

def read_file(folder, prefix, name):
    file_path = osp.join(folder, prefix + f'_{name}.txt')
    return np.genfromtxt(file_path, dtype=np.int64)

def read_sentigraph_data(folder: str, prefix: str):
    txt_files = glob.glob(os.path.join(folder, "{}_*.txt".format(prefix)))
    json_files = glob.glob(os.path.join(folder, "{}_*.json".format(prefix)))
    txt_names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in txt_files]
    json_names = [f.split(os.sep)[-1][len(prefix) + 1:-5] for f in json_files]
    names = txt_names + json_names

    with open(os.path.join(folder, prefix+"_node_features.pkl"), 'rb') as f:
        x = pickle.load(f)
    x = torch.from_numpy(x)
    edge_index = read_file(folder, prefix, 'edge_index')
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    batch = read_file(folder, prefix, 'node_indicator') - 1     # from zero
    y = read_file(folder, prefix, 'graph_labels')
    y = torch.tensor(y, dtype=torch.long)

    supplement = dict()
    if 'split_indices' in names:
        split_indices = read_file(folder, prefix, 'split_indices')
        split_indices = torch.tensor(split_indices, dtype=torch.long)
        supplement['split_indices'] = split_indices
    if 'sentence_tokens' in names:
        with open(os.path.join(folder, prefix + '_sentence_tokens.json')) as f:
            sentence_tokens: dict = json.load(f)
        supplement['sentence_tokens'] = sentence_tokens

    data = Data(x=x, edge_index=edge_index, y=y)
    data, slices = split(data, batch)

    return data, slices, supplement


def read_syn_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pickle.load(f)

    x = torch.from_numpy(features).float()
    y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
    y = torch.from_numpy(np.where(y)[1])
    edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_mask = torch.from_numpy(train_mask)
    data.val_mask = torch.from_numpy(val_mask)
    data.test_mask = torch.from_numpy(test_mask)
    return data


def read_ba2motif_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
        dense_edges, node_features, graph_labels = pickle.load(f)

    data_list = []
    for graph_idx in range(dense_edges.shape[0]):
        data_list.append(Data(x=torch.from_numpy(node_features[graph_idx]).float(),
                              edge_index=dense_to_sparse(torch.from_numpy(dense_edges[graph_idx]))[0],
                              y=torch.from_numpy(np.where(graph_labels[graph_idx])[0])))
    return data_list



def get_dataset(data_args):
    sync_dataset_dict = {
        'BA_2Motifs'.lower(): 'BA_2Motifs',
        'BA_Shapes'.lower(): 'BA_shapes',
        'BA_Community'.lower(): 'BA_community',
        'BA_LRP'.lower(): "ba_lrp"
    }
    sentigraph_names = ['Graph_SST2', 'Graph_SST5', 'Graph_Twitter']
    sentigraph_names = [name.lower() for name in sentigraph_names]
    molecule_net_dataset_names = [name.lower() for name in MoleculeNet.names.keys()]

    if data_args.dataset_name.lower() == 'MUTAG'.lower():
        return load_MUTAG(data_args)
    elif data_args.dataset_name.lower() == 'Mutagenicity'.lower():
        return load_MutagenicityDataset(data_args)
    elif data_args.dataset_name.lower() == 'NCI1'.lower():
        return load_NCI1Dataset(data_args)
    elif data_args.dataset_name.lower() in sync_dataset_dict.keys():
        return load_syn_data(data_args)
    elif data_args.dataset_name.lower() in molecule_net_dataset_names:
        return load_MolecueNet(data_args)
    elif data_args.dataset_name.lower() in sentigraph_names:
        return load_SeniGraph(data_args)
    elif data_args.dataset_name.lower() == "computers":
        return Amazon(root=data_args.dataset_dir, name = data_args.dataset_name)
    else:
        raise NotImplementedError


class MUTAGDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(MUTAGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.slices['x']) - 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return [self.name+'_A', self.name+'_graph_labels', self.name+'_graph_indicator', self.name+'_node_labels']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        with open(os.path.join(self.raw_dir, self.name+'_node_labels.txt'), 'r') as f:
            nodes_all_temp = f.read().splitlines()
            nodes_all = [int(i) for i in nodes_all_temp]

        adj_all = np.zeros((len(nodes_all), len(nodes_all)))
        with open(os.path.join(self.raw_dir, self.name+'_A.txt'), 'r') as f:
            adj_list = f.read().splitlines()
        for item in adj_list:
            lr = item.split(', ')
            l = int(lr[0])
            r = int(lr[1])
            adj_all[l - 1, r - 1] = 1

        with open(os.path.join(self.raw_dir, self.name+'_graph_indicator.txt'), 'r') as f:
            graph_indicator_temp = f.read().splitlines()
            graph_indicator = [int(i) for i in graph_indicator_temp]
            graph_indicator = np.array(graph_indicator)

        with open(os.path.join(self.raw_dir, self.name+'_graph_labels.txt'), 'r') as f:
            graph_labels_temp = f.read().splitlines()
            graph_labels = [int(i) for i in graph_labels_temp]

        data_list = []
        for i in range(1, 189):
            idx = np.where(graph_indicator == i)
            graph_len = len(idx[0])
            adj = adj_all[idx[0][0]:idx[0][0] + graph_len, idx[0][0]:idx[0][0] + graph_len]
            label = int(graph_labels[i - 1] == 1)
            feature = nodes_all[idx[0][0]:idx[0][0] + graph_len]
            nb_clss = 7
            targets = np.array(feature).reshape(-1)
            one_hot_feature = np.eye(nb_clss)[targets]
            data_example = Data(x=torch.from_numpy(one_hot_feature).float(),
                                edge_index=dense_to_sparse(torch.from_numpy(adj))[1], y=torch.tensor(label))
            data_list.append(data_example)

        torch.save(self.collate(data_list), self.processed_paths[0])


class MutagenicityDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(MutagenicityDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.slices['x']) - 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return [self.name+'_A', self.name+'_edge_gt.txt', self.name+'_graph_labels', self.name+'_graph_indicator', self.name+'_node_labels']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']
    

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        node_labels = np.loadtxt(os.path.join(self.raw_dir, self.name+'_node_labels.txt'),delimiter=',').astype(np.int32)
        edges = np.loadtxt(os.path.join(self.raw_dir, self.name+'_A.txt'),delimiter=',').astype(np.int32)
        edge_labels= np.loadtxt(os.path.join(self.raw_dir, self.name+'_edge_labels.txt'),delimiter=',').astype(np.int32)
        edge_labels_gt = np.loadtxt(os.path.join(self.raw_dir, self.name+'_edge_gt.txt'),delimiter=',').astype(np.int32)

        graph_indicator = np.loadtxt(os.path.join(self.raw_dir, self.name+'_graph_indicator.txt'), delimiter=',').astype(np.int32)
        graph_labels = np.loadtxt(os.path.join(self.raw_dir, self.name+'_graph_labels.txt'), delimiter=',').astype(np.int32)

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

        data_list = []
        for gid in range(len(graph_labels)):
            label = int(graph_labels[gid]==1)
            idx = np.where(graph_indicator == gid+1)
            graph_len = len(idx[0])
            #feature = node_label_lists[gid]
            feature = node_labels[idx[0][0]:idx[0][0] + graph_len]
            if len(feature) != max(np.array(edge_lists[gid]).reshape(1,-1)[0])+1:
                print(gid, len(feature), max(np.array(edge_lists[gid]).reshape(1,-1)[0]))
            nb_clss = max(node_labels) + 1
            targets = np.array(feature).reshape(-1)
            one_hot_feature = np.eye(nb_clss)[targets]
            data_example = Data(x=torch.from_numpy(one_hot_feature).float(),
                                edge_index=torch.tensor(edge_lists[gid], dtype=torch.long).T, y=torch.tensor(graph_labels[gid], dtype=torch.long), gid = gid, edge_label= torch.from_numpy(np.array(edge_label_lists[gid])), edge_label_gt= torch.from_numpy(np.array(edge_label_gt_lists[gid])))
            data_list.append(data_example)

        torch.save(self.collate(data_list), self.processed_paths[0])


class NCI1Dataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(NCI1Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.slices['x']) - 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return [self.name+'_A', self.name+'_graph_labels', self.name+'_graph_indicator', self.name+'_node_labels']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']
    

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        node_labels = np.loadtxt(os.path.join(self.raw_dir, self.name+'_node_labels.txt'),delimiter=',').astype(np.int32)
        edges = np.loadtxt(os.path.join(self.raw_dir, self.name+'_A.txt'),delimiter=',').astype(np.int32)
        graph_indicator = np.loadtxt(os.path.join(self.raw_dir, self.name+'_graph_indicator.txt'), delimiter=',').astype(np.int32)
        graph_labels = np.loadtxt(os.path.join(self.raw_dir, self.name+'_graph_labels.txt'), delimiter=',').astype(np.int32)

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
        edge_list = []
        for s,t in edges:
            sgid = node2graph[s]
            tgid = node2graph[t]
            if sgid!=tgid:
                print('edges connecting different graphs, error here, please check.')
                print(s,t,'graph id',sgid,tgid)
                exit(1)
            gid = sgid
            if gid !=  graphid:
                edge_lists.append(edge_list)
                edge_list = []
                graphid = gid
            start = starts[gid]
            edge_list.append((s-start,t-start))
        edge_lists.append(edge_list)

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

        data_list = []
        for gid in range(len(graph_labels)):
            label = int(graph_labels[gid]==1)
            idx = np.where(graph_indicator == gid+1)
            graph_len = len(idx[0])
            #feature = node_label_lists[gid]
            feature = node_labels[idx[0][0]:idx[0][0] + graph_len]
            if len(feature) != max(np.array(edge_lists[gid]).reshape(1,-1)[0])+1:
                print(gid, len(feature), max(np.array(edge_lists[gid]).reshape(1,-1)[0]))
            nb_clss = max(node_labels) + 1
            targets = np.array(feature).reshape(-1)
            one_hot_feature = np.eye(nb_clss)[targets]
            data_example = Data(x=torch.from_numpy(one_hot_feature).float(),
                                edge_index=torch.tensor(edge_lists[gid], dtype=torch.long).T, y=torch.tensor(graph_labels[gid], dtype=torch.long), gid = gid)
            data_list.append(data_example)

        torch.save(self.collate(data_list), self.processed_paths[0])


class SentiGraphDataset(InMemoryDataset):
    r"""
    The SentiGraph datasets from `Explainability in Graph Neural Networks: A Taxonomic Survey
    <https://arxiv.org/abs/2012.15445>`_.
    The datasets take pretrained BERT as node feature extractor
    and dependency tree as edges to transfer the text sentiment datasets into
    graph classification datasets.

    The dataset `Graph-SST2 <https://drive.google.com/file/d/1-PiLsjepzT8AboGMYLdVHmmXPpgR8eK1/view?usp=sharing>`_
    should be downloaded to the proper directory before running. All the three datasets Graph-SST2, Graph-SST5, and
    Graph-Twitter can be download in this
    `link <https://drive.google.com/drive/folders/1dt0aGMBvCEUYzaG00TYu1D03GPO7305z?usp=sharing>`_.

    Args:
        root (:obj:`str`): Root directory where the datasets are saved
        name (:obj:`str`): The name of the datasets.
        transform (:obj:`Callable`, :obj:`None`): A function/transform that takes in an
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (:obj:`Callable`, :obj:`None`):  A function/transform that takes in
            an :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    .. note:: The default parameter of pre_transform is :func:`~undirected_graph`
        which transfers the directed graph in original data into undirected graph before
        being saved to disk.
    """
    def __init__(self, root, name, transform=None, pre_transform=undirected_graph):
        self.root = root
        self.name = name
        super(SentiGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.supplement = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['node_features', 'node_indicator', 'sentence_tokens', 'edge_index',
                'graph_labels', 'split_indices']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        try:
            self.data, self.slices, self.supplement = read_sentigraph_data(self.raw_dir, self.name)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            if type(e) is FileNotFoundError:
                print("Please download the required datasets file to the root directory.")
                print("The google drive link is "
                      "https://drive.google.com/drive/folders/1dt0aGMBvCEUYzaG00TYu1D03GPO7305z?usp=sharing")
            raise SystemExit()
        
        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices, self.supplement), self.processed_paths[0])


class SynGraphDataset(InMemoryDataset):
    r"""
    The Synthetic datasets used in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>`_.
    It takes Barabási–Albert(BA) graph or balance tree as base graph
    and randomly attachs specific motifs to the base graph.

    Args:
        root (:obj:`str`): Root data directory to save datasets
        name (:obj:`str`): The name of the dataset. Including :obj:`BA_shapes`, BA_grid,
        transform (:obj:`Callable`, :obj:`None`): A function/transform that takes in an
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (:obj:`Callable`, :obj:`None`):  A function/transform that takes in
            an :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(SynGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data = read_syn_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


class BA2MotifDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(BA2MotifDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = read_ba2motif_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save(self.collate(data_list), self.processed_paths[0])
'''

class BA_LRP(InMemoryDataset):
    r"""
    The synthetic graph classification dataset used in
    `Higher-Order Explanations of Graph Neural Networks via Relevant Walks <https://arxiv.org/abs/2006.03589>`_.
    The first class in :class:`~BA_LRP` is Barabási–Albert(BA) graph which connects a new node :math:`\mathcal{V}` from
    current graph :math:`\mathcal{G}`.

    .. math:: p(\mathcal{V}) = \frac{Degree(\mathcal{V})}{\sum_{\mathcal{V}' \in \mathcal{G}} Degree(\mathcal{V}')}

    The second class in :class:`~BA_LRP` has a slightly higher growth model and nodes are selected
    without replacement with the inverse preferential attachment model.

    .. math:: p(\mathcal{V}) = \frac{Degree(\mathcal{V})^{-1}}{\sum_{\mathcal{V}' \in \mathcal{G}} Degree(\mathcal{V}')^{-1}}

    Args:
        root (:obj:`str`): Root data directory to save datasets
        num_per_class (:obj:`int`): The number of the graphs for each class.
        transform (:obj:`Callable`, :obj:`None`): A function/transform that takes in an
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (:obj:`Callable`, :obj:`None`):  A function/transform that takes in
            an :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    .. note:: :class:`~BA_LRP` will automatically generate the dataset
      if the dataset file is not existed in the root directory.

    Example:
         dataset = BA_LRP(root='./datasets')
         loader = Dataloader(dataset, batch_size=32)
         data = next(iter(loader))
        # Batch(batch=[640], edge_index=[2, 1344], x=[640, 1], y=[32, 1])

    Where the attributes of data indices:

    - :obj:`batch`: The assignment vector mapping each node to its graph index
    - :obj:`x`: The node features
    - :obj:`edge_index`: The edge matrix
    - :obj:`y`: The graph label

    """
    url = ('https://github.com/divelab/DIG_storage/raw/main/xgraph/datasets/ba_lrp.pt')

    def __init__(self, root, num_per_class=10000, transform=None, pre_transform=None):
        self.root = root
        self.name = 'ba_lrp'
        self.num_per_class = num_per_class
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"raw.pt"]

    @property
    def processed_file_names(self):
        return [f'data.pt']

    def download(self):
        url = self.url
        path = download_url(url, self.raw_dir)
        shutil.move(path, path.replace('ba_lrp.pt', 'raw.pt'))

    def gen_class1(self):
        x = torch.tensor([[1], [1]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([[0]], dtype=torch.float))

        for i in range(2, 20):
            data.x = torch.cat([data.x, torch.tensor([[1]], dtype=torch.float)], dim=0)
            deg = torch.stack([(data.edge_index[0] == node_idx).float().sum() for node_idx in range(i)], dim=0)
            sum_deg = deg.sum(dim=0, keepdim=True)
            probs = (deg / sum_deg).unsqueeze(0)
            prob_dist = torch.distributions.Categorical(probs)
            node_pick = prob_dist.sample().squeeze()
            data.edge_index = torch.cat([data.edge_index,
                                         torch.tensor([[node_pick, i], [i, node_pick]], dtype=torch.long)], dim=1)
            data.y = torch.cat([data.y, torch.tensor([[0]], dtype=torch.float)], dim=0)

        return data

    def gen_class2(self):
        x = torch.tensor([[1], [1]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([[1]], dtype=torch.float))
        epsilon = 1e-30

        for i in range(2, 20):
            data.x = torch.cat([data.x, torch.tensor([[1]], dtype=torch.float)], dim=0)
            deg_reciprocal = torch.stack([1 / ((data.edge_index[0] == node_idx).float().sum() + epsilon) for node_idx in range(i)], dim=0)
            sum_deg_reciprocal = deg_reciprocal.sum(dim=0, keepdim=True)
            probs = (deg_reciprocal / sum_deg_reciprocal).unsqueeze(0)
            prob_dist = torch.distributions.Categorical(probs)
            node_pick = -1
            for _ in range(1 if i % 5 != 4 else 2):
                new_node_pick = prob_dist.sample().squeeze()
                while new_node_pick == node_pick:
                    new_node_pick = prob_dist.sample().squeeze()
                node_pick = new_node_pick
                data.edge_index = torch.cat([data.edge_index,
                                             torch.tensor([[node_pick, i], [i, node_pick]], dtype=torch.long)], dim=1)
                data.y = torch.cat([data.y, torch.tensor([[1]], dtype=torch.float)], dim=0)
        return data

    def process(self):
        if files_exist(self.raw_paths):
            shutil.copyfile(self.raw_paths[0], self.processed_paths[0])
            return

        data_list = []
        for i in range(self.num_per_class):
            data_list.append(self.gen_class1())
            data_list.append(self.gen_class2())

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
'''

class CitationDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(CitationDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.slices['x']) - 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return [self.name+'.cites', self.name+'.content']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def read_file_citation(self, file_name):
        file_read=[]
        f = open(file_name , 'r')
        for line in f:
            file_read.append(line)

        clean_list = []
        for line in file_read:
            clean_list.append(line.split())
        return np.array(clean_list, dtype = 'str')

    # 将特征进行one-hot编码表示
    def encode_onehot(self, labels):
        classes = sorted(list(set(labels)))
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                dtype=np.int32)
        return labels_onehot

    # 将向量进行归一化表示
    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    # 将稀疏矩阵变成pytorch的稀疏矩阵表示
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def extract_pubmed_text(self, data_prefix):
        a = []
        myFile = open(data_prefix + 'Pubmed-Diabetes.DIRECTED.cites.tab' , 'r' )
        for Row in myFile:
            a.append(Row.split('\t'))
        myFile.close()

        a = a[2:]    # first two lines are descriptions
        paper_pair = []

        for idx in a:
            p1 = idx[1][6:]    # form: 'paper:' + id
            p2 = idx[3][6:-1]  # form: 'paper:' + id + '\n'
            paper_pair.append([p1, p2])

        b = []
        myFile= open(data_prefix + 'Pubmed-Diabetes.NODE.paper.tab', 'r' )
        for Row in myFile:
            b.append(Row.split('\t'))
        myFile.close()

        description = b[1]
        b = b[2:]       # first two lines are descriptions
        feature_list = []
        description = description[1:-1]
        for k in description:
            k = k[8:]
            stop = k.index(':')
            k = k[:stop]
            feature_list.append(k)

        nodes = []
        labels = np.zeros(len(b))
        features = np.zeros([len(b),len(description)])
        for idx, content in enumerate(b):
            nodes.append(content[0])
            labels[idx] = content[1][-1]
            feature_description = content[2:-1]
            for k in feature_description:
                start = k.index('=')
                feature_name = k[:start]
                feature_value = float( k[start+1 : ] )
                feature_location = feature_list.index(feature_name)
                features[idx][feature_location] = feature_value

        return np.array(paper_pair), features, self.encode_onehot(labels), np.array(nodes)

    def remove_nodes(self, nodes, citation):
        new_citation = []
        for k in citation:
            p1 = k[0]
            p2 = k[1]
            if p1 in nodes and p2 in nodes:
                new_citation.append(k)
        return np.array(new_citation)

    # 根据边缘和节点分类来获得节点的邻接矩阵表示
    def adj_matrix(self, edges, labels_onehot):
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels_onehot.shape[0], labels_onehot.shape[0]),
                            dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        return adj

    # 加载数据集，该数据集是引文网络cora，citeseer，pumbed三种，
    # 获得数据集的节点图表示，节点特征，邻接矩阵并将其表示为拉普拉斯矩阵，
    # 并且根据5：3：2的比例来划分训练集，测试集，验证集
    def load_data_citation(self):
        nodes = []
        features = sp.csr_matrix([])
        citation = []
        labels_onehot = []
        if self.name.lower() == 'cora' or self.name.lower() == 'citeseer':
            citation = self.read_file_citation(os.path.join(self.raw_dir, self.name + '.cites'))
            contents = self.read_file_citation(os.path.join(self.raw_dir, self.name + '.content'))
            features = sp.csr_matrix(contents[:, 1:-1], dtype=np.float32)
            labels = contents[:,-1]
            labels_onehot = self.encode_onehot(labels)
            #labels_onehot = encode_onehot(contents[:,-1])
            nodes = contents[:,0]
        elif self.name == 'pubmed':
            citation, contents, labels_onehot, nodes = self.extract_pubmed_text(self.raw_dir)
            features = sp.csr_matrix(contents, dtype=np.float32)
        # remove if nodes in edges without features
        citation = self.remove_nodes(nodes, citation)
        # map node/citation index into int-space
        idx_map = {j: i for i, j in enumerate(nodes)}
        graph = [[str(i[0]), str(i[1])] for i in citation]
        edges = np.array([[idx_map[i[0]], idx_map[i[1]]] for i in citation])
        #adj = sp.csr_matrix(adj_matrix(edges, labels_onehot))
        adj = self.adj_matrix(edges, labels_onehot)
        normalized_adj = self.normalize(adj + sp.eye(adj.shape[0]))
        normalized_adj = sp.csr_matrix(normalized_adj)

        features = self.normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
        labels_onehot = torch.LongTensor(np.where(labels_onehot)[1])
        normalized_adj = self.sparse_mx_to_torch_sparse_tensor(normalized_adj)
        #adj = sparse_mx_to_torch_sparse_tensor(adj)

        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 1 - train_ratio - val_ratio
        idx_train = torch.LongTensor(range( int(train_ratio * len(nodes)) ) )
        idx_val = torch.LongTensor(range( int(train_ratio * len(nodes) + 1), int((train_ratio+val_ratio)*len(nodes)) ) )
        idx_test = torch.LongTensor(range( int( (1-test_ratio) * len(nodes) + 1), len(nodes) ) )

        return graph, adj,normalized_adj, features, labels_onehot, nodes, idx_map, idx_train, idx_val, idx_test

    def process(self):
        graph, adj,normalized_adj, features, labels_onehot, nodes, idx_map, idx_train, idx_val, idx_test = self.load_data_citation()
        data = Data(x=features, y=labels_onehot, edge_index=normalized_adj._indices())
        data.train_mask = idx_train
        data.val_mask = idx_val
        data.test_mask = idx_test
        data.idx_map = idx_map
        torch.save(self.collate([data]), self.processed_paths[0])

def load_CitationDataset(data_args):
    dataset = CitationDataset(root=data_args.dataset_dir, name=data_args.dataset_name)
    return dataset

def load_MUTAG(data_args):
    """ 188 molecules where label = 1 denotes mutagenic effect """
    dataset = MUTAGDataset(root=data_args.dataset_dir, name=data_args.dataset_name)
    return dataset

def load_MutagenicityDataset(data_args):
    """ Mutagenicity Dataset select data of two types: 1, mutagenic effect(graphlabel = 0)  and have groundth(N02 or NH2) 2, no mutagenic effect (graphlabel = 1)  and no groundth(N02 or NH2) """
    dataset = MutagenicityDataset(root=data_args.dataset_dir, name=data_args.dataset_name)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    return dataset

def load_NCI1Dataset(data_args):
    dataset = NCI1Dataset(root=data_args.dataset_dir, name=data_args.dataset_name)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    return dataset

def load_syn_data(data_args):
    """ The synthetic dataset """
    if data_args.dataset_name.lower() == 'BA_2Motifs'.lower():
        dataset = BA2MotifDataset(root=data_args.dataset_dir, name=data_args.dataset_name)
    else:
        dataset = SynGraphDataset(root=data_args.dataset_dir, name=data_args.dataset_name)
    #dataset.node_type_dict = {k: v for k, v in enumerate(range(dataset.num_classes))}
    #dataset.node_color = None
    return dataset


def load_MolecueNet(data_args):
    """ Attention the multi-task problems not solved yet """
    molecule_net_dataset_names = {name.lower(): name for name in MoleculeNet.names.keys()}
    dataset = MoleculeNet(root=data_args.dataset_dir, name=molecule_net_dataset_names[data_args.dataset_name.lower()])
    # Chem.PeriodicTable.GetElementSymbol()
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    #dataset.node_type_dict = None
    #dataset.node_color = None
    return dataset


def load_SeniGraph(data_args):
    dataset = SentiGraphDataset(root=data_args.dataset_dir, name=data_args.dataset_name)
    return dataset


def get_dataloader(dataset, data_args, train_args):
    """ data_args.data_split_ratio : list [float, float, float]
        return a dict with three data loaders
    """
    if not data_args.random_split and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        '''
        num_train = int(data_args.data_split_ratio[0] * len(dataset))
        num_eval = int(data_args.data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval
        '''
        num_train = int(data_args.data_split_ratio[0] * len(dataset.data.y))
        num_eval = int(data_args.data_split_ratio[1] * len(dataset.data.y))
        num_test = len(dataset.data.y) - num_train - num_eval

        train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                         generator=torch.Generator().manual_seed(data_args.seed))

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=train_args.batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(eval, batch_size=train_args.batch_size, shuffle=False)
    dataloader['test'] = DataLoader(test, batch_size=train_args.batch_size, shuffle=False)
    return dataloader


if __name__ == '__main__':
    dataset = load_syn_data(data_args)
    data = dataset[0]
    pass
