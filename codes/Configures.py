import os
import torch
#from tap import Tap
from typing import List
import argparse
import torch.cuda

def get_data_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='Computers', help="dataset name string")   #citeseer, BA_shapes  BA_community   Mutagenicity  PubMed
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
    parser.add_argument('--latent_dim', type= List, default=[20,20,20], help="the hidden units for each gnn layer[20, 20, 20]      [128, 128, 128]")
    parser.add_argument('--mlp_hidden', type= List, default=[], help="the hidden units for mlp classifier")
    parser.add_argument('--gnn_dropout', type=float, default=0.0, help="the dropout after gnn layers")
    parser.add_argument('--dropout', type=float, default=0.6, help="the dropout after mlp layers")
    args, _ = parser.parse_known_args()
    return args

def get_train_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.005)     #0.005
    parser.add_argument("--batch_size", type=int, default=64)   
    parser.add_argument("--weight_decay", type=float, default=0.0)  
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--save_epoch', type= int, default=10)
    parser.add_argument('--early_stopping', type=int, default=1000)
    args, _ = parser.parse_known_args()
    return args


data_args = get_data_params()
model_args = get_gnnModel_params()
model_args.device = "cuda" if torch.cuda.is_available() else "cpu"
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
