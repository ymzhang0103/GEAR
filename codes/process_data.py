import torch
from dataset.syn_dataset import SynGraphDataset
from torch_geometric.data import DataLoader
import os
import gengraph as gengraph
import numpy as np
import networkx as nx
import pickle as pkl
import fornode.config as config

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def split_dataset(dataset):
    indices = []
    num_classes = dataset.num_classes
    train_percent = 0.8
    for i in range(num_classes):
        index = (dataset.data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:int(len(i) * train_percent)] for i in indices], dim=0)
    rest_index = torch.cat([i[int(len(i) * train_percent):] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    dataset.data.train_mask = index_to_mask(train_index, size=dataset.data.num_nodes)
    dataset.data.val_mask = index_to_mask(rest_index[:len(rest_index) // 2], size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(rest_index[len(rest_index) // 2:], size=dataset.data.num_nodes)

    dataset.data, dataset.slices = dataset.collate([dataset.data])
    return dataset


def proc_data(dataset_name):
    dataset = SynGraphDataset(os.getcwd()+'/dataset', dataset_name)
    dataset.data.x = dataset.data.x.to(torch.float32)
    # dataset.data.x = dataset.data.x[:, :1]
    # dataset.data.y = dataset.data.y[:, 2]
    dim_node = dataset.num_node_features
    dim_edge = dataset.num_edge_features
    # num_targets = dataset.num_classes
    num_classes = dataset.num_classes

    splitted_dataset = split_dataset(dataset)
    splitted_dataset.data.mask = splitted_dataset.data.test_mask
    splitted_dataset.slices['mask'] = splitted_dataset.slices['test_mask']
    dataloader = DataLoader(splitted_dataset, batch_size=1, shuffle=False)
    return dataset[0]


def add_random_edge(args):
    print("load data from ", 'dataset/'+args.dataset_name+'/raw/' + args.dataset_name + '.pkl')
    with open('dataset/'+args.dataset_name+'/raw/' + args.dataset_name + '.pkl', 'rb') as fin:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pkl.load(fin)
    print("before add, total number of edges:", np.nonzero(adj)[0].shape)

    nodes_num = adj.shape[0]
    added_random_edges = args.add_random_edges
    # add random edges between nodes:
    print("add "+ str(args.add_random_edges) +" edges")
    while added_random_edges > 0:
        row, col = np.random.choice(nodes_num, 2, replace=False)
        print(row, col)
        if adj[row, col]==0:
            adj[row, col] = 1
            added_random_edges = added_random_edges - 1
    #G, role_id, name = gengraph.gen_syn1_new(args.add_random_edges)
    #adj = np.array(nx.adjacency_matrix(G).todense())
    print("after add edges, total number of edges:", np.nonzero(adj)[0].shape)
    out_filename = 'dataset/'+args.dataset_name+'/raw/' + args.dataset_name + '-' + str(args.add_random_edges) + '.pkl'
    with open(out_filename, 'wb') as fin:
        pkl.dump((adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix), fin)
    print("success save to ", out_filename)


if __name__ =="__main__":
    device = "cpu"
    args = config.get_params()
    config.set_seed(2021)
    #args.dataset = 'syn1'
    #args.dataset_name = "BA_shapes"
    args.dataset_name = "BA_community"
    args.add_random_edges=30
    add_random_edge(args)