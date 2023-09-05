import os
import torch
#from tap import Tap
from typing import List
import argparse
import torch.cuda
'''
class DataParser(Tap):
    #dataset_name: str = 'bbbp'
    #dataset_name: str = 'BA_shapes'
    #dataset_name: str = 'BA_community'
    #dataset_name: str = 'Graph-Twitter' 
    #dataset_name: str = 'Graph-SST5' 
    dataset_name: str = 'MUTAG'
    dataset_dir: str = '../datasets'
    random_split: bool = True
    data_split_ratio: List = [0.8, 0.1, 0.1]   # the ratio of training, validation and testing set for random split
    seed: int = 1


class GATParser(Tap):           # hyper-parameter for gat model
    gat_dropout: float = 0.6    # dropout in gat layer
    gat_heads: int = 10         # multi-head
    gat_hidden: int = 10        # the hidden units for each head
    gat_concate: bool = True    # the concatenation of the multi-head feature
    num_gat_layer: int = 3      # the gat layers


class ModelParser(GATParser):
    #device_id: int = 1
    model_name: str = 'gcn'
    checkpoint: str = './checkpoint'
    concate: bool = False                     # whether to concate the gnn features before mlp
    latent_dim: List[int] = [20, 20, 20]          # the hidden units for each gnn layer[20, 20, 20]      [128, 128, 128]   
    readout: 'str' = 'max'                    # the graph pooling method
    mlp_hidden: List[int] = []                # the hidden units for mlp classifier
    gnn_dropout: float = 0.0                  # the dropout after gnn layers
    dropout: float = 0.6                      # the dropout after mlp layers
    adj_normlize: bool = True                 # the edge_weight normalization for gcn conv
    #emb_normlize: bool = False                # the l2 normalization after gnn layer
    emb_normlize: bool = True
    model_path: str = ""                      # default path to save the model

    def process_args(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if not self.model_path:
            self.model_path = os.path.join(self.checkpoint,
                                           DataParser().parse_args(known_only=True).dataset_name,
                                           f"{self.model_name}_best.pth")


class TrainParser(Tap):
    learning_rate: float = 0.005
    batch_size: int = 64
    #batch_size: int = 1
    weight_decay: float = 0.0
    max_epochs: int = 800    #800
    save_epoch: int = 10
    early_stopping: int = 800 #800


class ExplainerParser(Tap):
    t0: float = 5.0                   # temperature denominator
    t1: float = 1.0                   # temperature numerator
    coff_size: float = 0.01           # constrains on mask size
    coff_ent: float = 5e-4            # constrains on smooth and continuous mask


data_args = DataParser().parse_args(known_only=True)
model_args = ModelParser().parse_args(known_only=True)
train_args = TrainParser().parse_args(known_only=True)
explainer_args = ExplainerParser().parse_args(known_only=True)

'''
def get_data_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='NCI1', help="dataset name string")   #citeseer, BA_shapes  BA_community   Mutagenicity  Graph_Twitter
    parser.add_argument("--dataset_dir", type=str, default='../datasets/', help="dataset dir string")  
    parser.add_argument('--random_split', type=bool, default=True)
    parser.add_argument('--data_split_ratio', type= List, default=[0.8, 0.1, 0.1], help="the ratio of training, validation and testing set for random split")
    parser.add_argument('--seed', type=int, default=1)
    args, _ = parser.parse_known_args()
    return args

def get_gnnModel_params():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument("--model_name", type=str, default='gcn', help="model name string")  
    parser.add_argument("--checkpoint", type=str, default='./checkpoint', help="checkpoint path string")  
    parser.add_argument("--readout", type=str, default='max', help=" the graph pooling method")  
    parser.add_argument("--model_path", type=str, default='', help="default path to save the model")  
    parser.add_argument('--concate', type=bool, default=False, help="whether to concate the gnn features before mlp")
    parser.add_argument('--adj_normlize', type=bool, default=False, help="the edge_weight normalization for gcn conv")
    parser.add_argument('--emb_normlize', type=bool, default=True, help="the l2 normalization after gnn layer")
    parser.add_argument('--latent_dim', type= List, default=[20, 20, 20], help="the hidden units for each gnn layer[20, 20, 20]      [128, 128, 128]")
    parser.add_argument('--mlp_hidden', type= List, default=[], help="the hidden units for mlp classifier")
    parser.add_argument('--gnn_dropout', type=float, default=0.0, help="the dropout after gnn layers")
    parser.add_argument('--dropout', type=float, default=0.6, help="the dropout after mlp layers")
    args, _ = parser.parse_known_args()
    return args

def get_train_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=64)  
    parser.add_argument("--weight_decay", type=float, default=0.0)  
    parser.add_argument('--max_epochs', type=int, default=800)
    parser.add_argument('--save_epoch', type= int, default=10)
    parser.add_argument('--early_stopping', type=int, default=800)
    args, _ = parser.parse_known_args()
    return args


data_args = get_data_params()
model_args = get_gnnModel_params()
model_args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#model_args.device ="cpu"
if not model_args.model_path:
    model_args.model_path = os.path.join(model_args.checkpoint, data_args.dataset_name, f"{model_args.model_name}_best.pth")
train_args = get_train_params()



import random
import numpy as np
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
