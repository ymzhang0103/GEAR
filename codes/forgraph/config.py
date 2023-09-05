import argparse
import numpy as np
import os
import random
import torch


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument("--dataset_root", type=str, default='datasets', help="Dataset root dir string")  
    parser.add_argument("--dataset", type=str, default='', help="Dataset string")  # 'Mutagenicity', Graph-Twitter, Graph-SST5
    parser.add_argument('--id', type=str, default='default_id', help='id to store in database')  #
    parser.add_argument('--early_stop', type=int, default= 100, help='early_stop')
    parser.add_argument('--dtype', type=str, default='float32')  #
    parser.add_argument('--seed',type=int, default=1234, help='seed')
    parser.add_argument('--setting',type=int, default=2, help='setting')

    parser.add_argument('--order', type=str, default='AW')  #
    parser.add_argument('--embnormlize', type=bool, default=True)  #
    parser.add_argument('--bias', type=bool, default=True)  #
    parser.add_argument('--random_edges_motif', type=int, default=0, help='Node to be explained')

    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
    parser.add_argument('--dropout',type=float, default=0.0, help='dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay',type=float, default=0.0, help='l2 norm')
    parser.add_argument('--hiddens', type=str, default='20-20-20')    #20-20-20,  128-128-128
    parser.add_argument("--lr", type=float, default=0.001,help='initial learning rate.')
    parser.add_argument('--act', type=str, default='relu', help='activation funciton')  #
    parser.add_argument('--initializer', default='glorot')

    parser.add_argument('--normadj', type=bool, default=False)  #
    parser.add_argument('--bn', type=bool, default=False)
    parser.add_argument('--concat', type=bool, default=False)
    parser.add_argument('--valid', type=bool, default=False)
    parser.add_argument('--batch', type=bool, default=True)

    parser.add_argument('--save_model',type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='./model_weights/GCN')

    # ---------------------paramerters for explainers----------------------
    parser.add_argument("--elr", type=float, default=0.003,help='initial learning rate.')   #0.05
    parser.add_argument('--eepochs', type=int, default=30, help='Number of epochs to train explainer.')   #20
  
    parser.add_argument('--coff_size', type=float, default=0.05, help='coefficient for size constriant')   #0.01
    parser.add_argument('--coff_ent', type=float, default=1.0, help='coefficient for entropy loss')   #0.01
    parser.add_argument('--miGroudTruth',type=bool, default=True, help='Mutual Information between hat y and GroundTruth Label')

    parser.add_argument('--coff_t0', type=float, default=5.0, help='initial temperature')
    parser.add_argument('--coff_te', type=float, default=2.0, help='final temperature')  #1.0
    parser.add_argument('--sample_bias',type=float, default=0.0, help='bias for sampling from 0-1')

    parser.add_argument('--train_ratio',type=float, default=0.6, help='ratio of instance for training')
    parser.add_argument('--train_instance',type=int, default=5, help='instance for training')

    args, _ = parser.parse_known_args()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return args
'''
def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
'''
args = get_params()
params = vars(args)

#set_seed(args.seed)

dtype = torch.float32
if args.dtype=='float64':
    dtype = torch.float64

eps = 1e-7