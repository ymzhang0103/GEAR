from codes.Configures import model_args
import torch
from codes.GNNmodelsHidden import GnnNets, GnnNets_NC
from codes.fornode.models import GCN2 as GCN
from collections import OrderedDict

def load_gnnNets_NC(ckpt_path, input_dim, output_dim, device):
    model_args.device = device
    gnnNets_NC = GnnNets_NC(input_dim=input_dim, output_dim=output_dim, model_args=model_args)
    gnnNets_NC.to_device()
    #ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    ckpt = torch.load(ckpt_path)
    gnnNets_NC.load_state_dict(ckpt['net'])
    return gnnNets_NC


def load_gnnNets_NC_cuda92GNN(ckpt_path, input_dim, output_dim, device):
    model_args.device = device
    gnnNets_NC = GnnNets_NC(input_dim=input_dim, output_dim=output_dim, model_args=model_args)
    gnnNets_NC.to_device()
    #ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    ckpt = torch.load(ckpt_path)
    new_state_dic = OrderedDict()
    for key, value in gnnNets_NC.state_dict().items():
        if "lin." in key:
            old_key = key.replace("lin.", "")
        else:
            old_key = key
        if "gnn_layers" in old_key:
            new_state_dic[key] = ckpt['net'][old_key].T
        else:
            new_state_dic[key] = ckpt['net'][old_key]
    gnnNets_NC.load_state_dict(new_state_dic)
    #gnnNets_NC.load_state_dict(ckpt['net'])
    return gnnNets_NC


def load_gnnNets_NC_bak(ckpt_path, input_dim, output_dim, device):
    model_args.device = device
    gnnNets_NC = GnnNets_NC(input_dim=input_dim, output_dim=output_dim, model_args=model_args)
    gnnNets_NC.to_device()
    #ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    ckpt = torch.load(ckpt_path)
    gnnNets_NC.load_state_dict(ckpt['net'])
    return gnnNets_NC

def load_GCN_PG(ckpt_path, input_dim, output_dim, device):
    model = GCN(input_dim=input_dim, output_dim=output_dim, device=device)
    model.to(device)
    model.load_state_dict(torch.load(ckpt_path))
    return model

def load_gnnNets_GC(ckpt_path, input_dim, output_dim, device):
    model_args.device = device
    gnnNets_GC = GnnNets(input_dim=input_dim,  output_dim=output_dim, model_args=model_args)
    gnnNets_GC.to_device()
    ckpt = torch.load(ckpt_path)
    gnnNets_GC.load_state_dict(ckpt['net'])
    return gnnNets_GC

