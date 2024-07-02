#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#CUDA_LAUNCH_BLOCKING=1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append('./codes/fornode/')
import time

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from codes.fornode.config import args
from codes.fornode.utils import *
from codes.fornode.metricsHidden_batch import *
import numpy as np
from codes.fornode.Extractor_batch import Extractor
from codes.fornode.ExplainerMO_batch import ExplainerMO
from scipy.sparse import coo_matrix,csr_matrix
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch.optim
from torch.optim import Adam, SGD

from codes.load_GNNNets_hidden import load_gnnNets_NC
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_networkx
import os.path as osp

from codes.mograd import MOGrad
import math
import operator
#from collections import OrderedDict
from torch_geometric.datasets import Amazon

def acc(masked_adj, sub_edge_index, sub_edge_label):
    mask = masked_adj.cpu().detach().numpy()
    real = []
    pred = []
    sub_edge_label = sub_edge_label.todense()
    for r,c in list(zip(sub_edge_index[0], sub_edge_index[1])):
        d = sub_edge_label[r,c] + sub_edge_label[c,r]
        if d == 0:
            real.append(0)
        else:
            real.append(1)
        pred.append(mask[r][c]+mask[c][r])

    if len(np.unique(real))==1 or len(np.unique(pred))==1:
        auc = -1
    else:
        auc = roc_auc_score(real, pred)
    return auc, real, pred


def explain_test(masked_adj, masked_pred, sub_edge_index, sub_edge_label, origin_pred, validation=False):
    #label = np.argmax(sub_label, -1)
    if args.dataset == "Computers":
        auc=0
        real = []
        pred = []
    else:
        auc, real, pred = acc(masked_adj, sub_edge_index, sub_edge_label)
    #print("origin_pred", origin_pred)
    #print("masked_pred", masked_pred)
    ndcg = 0
    if not validation:
        ndcg, r_mask, _, _ = rho_ndcg(origin_pred, masked_pred, len(masked_pred))
    return auc, ndcg, real, pred


#@profile
def train_MO(iter, args, model, explainer, extractor, save_map, trainnodes, valnodes, graphdata):
    tik = time.time()
    f_train = open(save_map + str(iter) + "/" + "train_LOG_" + args.model_filename + "_BEST.txt", "w")
    f_train_grad = open(save_map + str(iter) + "/" + "train_gradient_LOG_" + args.model_filename + "_BEST.txt", "w")
    t0 = args.coff_t0
    t1 = args.coff_te
    epochs = args.eepochs
    model.eval()
    explainer.train()
    best_decline = 0.0
    best_F_fidelity = 0.0
    best_del_F_fidelity = 0.0

    optimizer = Adam(explainer.elayers.parameters(), lr=args.elr)
    optimizer = MOGrad(optimizer)
    sim_obj = round( math.cos(math.radians(angle)), 3)
    dominant_index = dominant_loss[1]
    modify_index_arr = None
    coslist = None
    start = 0
    traversal_flag = False
    for epoch in range(epochs):
        loss = torch.tensor(0.0)
        pred_loss = 0
        lap_loss = 0
        con_loss = 0
        value_loss = 0
        sort_loss = 0
        hidden_loss = 0
        cf_loss = 0
        l, pl, ll, cl, hl, cfl, vl, sl = 0,0,0,0,0,0,0,0
        losses = []
        tmp = float(t0 * np.power(t1 / t0, epoch /epochs))

        for node_index in range(start, start + args.batch_size):
            if node_index >= len(trainnodes):
                traversal_flag = True
                break
            node = trainnodes[node_index]
            if node_index%100== 0:
                print("node_index: " + str(node_index) + ", node: " + str(node))

            with torch.no_grad():
                sub_node, sub_adj, sub_edge_label = extractor.subgraph(node)
                if len(sub_node)<=0:
                    continue
                sub_edge_index = torch.nonzero(sub_adj).T
                data = Data(x=graphdata.x[sub_node], edge_index=sub_edge_index).to(args.device)
                _, sub_output, sub_embed, sub_hidden_emb = model(data)
                nodeid = 0
                sub_output = sub_output[nodeid]
                old_pred_label = torch.argmax(sub_output)

            new_pred, cf_pred, _, new_hidden_emb = explainer((nodeid, data, sub_adj, sub_embed, tmp), training=True)
            
            if "ce" in args.loss_flag:
                l,pl,ll,cl, hl = explainer.loss_ce_hidden(new_hidden_emb, sub_hidden_emb, new_pred, old_pred_label, sub_label, nodeid)
                vl = 0
                sl= 0
                cfl = 0
            elif "kl" in args.loss_flag:
                l, pl, ll, cl, hl, cfl = explainer.loss_kl_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output)
                vl = 0
                sl = 0
            elif "pl" in args.loss_flag:
                l, pl, ll, cl, vl, sl, hl, cfl = explainer.loss_pl_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output)
            elif "pdiff" in args.loss_flag:
                l, pl, ll, cl, hl, cfl = explainer.loss_diff_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output)
                vl = 0
                sl = 0  
            elif "CF" == args.loss_flag:
                l, pl, ll, cl, hl, cfl = explainer.loss_cf_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output)
                vl = 0
                sl = 0 
            loss = loss + l
            pred_loss = pred_loss + pl
            lap_loss = lap_loss + ll
            con_loss = con_loss + cl
            value_loss = value_loss + vl
            sort_loss = sort_loss + sl
            hidden_loss = hidden_loss + hl
            cf_loss = cf_loss + cfl

            torch.cuda.empty_cache()

        if args.loss_flag == "ce":
            losses = [pred_loss, lap_loss, con_loss]
        elif args.loss_flag == "hidden":
            losses = [hidden_loss, lap_loss, con_loss]   
        elif  args.loss_flag == "ce_hidden":
            losses = [pred_loss, hidden_loss, lap_loss, con_loss]
        elif  args.loss_flag == "kl":
            losses = [pred_loss, lap_loss, con_loss]
        elif  args.loss_flag == "kl_hidden":
            losses = [pred_loss, hidden_loss, lap_loss, con_loss]
        elif  args.loss_flag == "kl_hidden_CF_LM":
            losses = [pred_loss, hidden_loss, cf_loss, lap_loss, con_loss]
        elif args.loss_flag == "pl_value":
            losses = [sort_loss, value_loss, lap_loss, con_loss]
        elif args.loss_flag == "pl_value_hidden":
            losses = [sort_loss, value_loss, hidden_loss, lap_loss, con_loss]
        elif args.loss_flag == "pl_value_hidden_CF":
            losses = [sort_loss, value_loss, hidden_loss, lap_loss, con_loss, cf_loss]
        elif args.loss_flag == "pdiff_hidden_CF_LM_conn":
            losses = [pred_loss, hidden_loss, cf_loss, lap_loss, con_loss]
        elif args.loss_flag == "pdiff_CF_LM_conn":
            losses = [pred_loss, cf_loss, lap_loss, con_loss]
        elif args.loss_flag == "pdiff_hidden_CF":
            losses = [pred_loss, hidden_loss, cf_loss]
        elif args.loss_flag == "pdiff_CF_only":
            losses = [pred_loss, cf_loss]
        elif args.loss_flag=="pdiff_LM":
            losses = [pred_loss, lap_loss]
        #optimizer.pc_backward(losses)
        if "PCGrad" in optimal_method:
            if dominant_loss[1] is not None:
                dominant_index, modify_index_arr, coslist, _, _ = optimizer.pc_backward_dominant(losses, dominant_index)
            else:
                modify_index_arr, coslist, _, _ = optimizer.pc_backward(losses)
        elif "GEAR" in optimal_method:
            if dominant_loss[1] is not None: 
                modify_index_arr, coslist, _, _, _ = optimizer.backward_adjust_grad_dominant(losses, dominant_index, sim_obj)
                #sim_obj = cur_sim_obj
            else:
                modify_index_arr, coslist, _, _, _ = optimizer.backward_adjust_grad(losses, sim_obj)
                #sim_obj = cur_sim_obj
        elif "getGrad" in optimal_method:
            coslist, _, _ = optimizer.get_grads(losses)
            loss.backward()   #original
        elif "CAGrad" in optimal_method:
            optimizer.cagrad_backward(losses, dominant_index)
        
        if traversal_flag:
            optimizer.step()
            optimizer.zero_grad()

            start = 0
            traversal_flag = False
        
            reals = []
            preds = []
            ndcgs = []
            metric = MaskoutMetric(model, args)
            classify_acc = 0
            fidelity_complete = []
            fidelityminus_complete = []
            del_fidelity_complete = []
            del_fidelityminus_complete = []
            for node in valnodes:
                sub_node, sub_adj, sub_edge_label = extractor.subgraph(node)
                if len(sub_node)<=0:
                    continue
                sub_feature = graphdata.x[sub_node]
                sub_edge_index = torch.nonzero(sub_adj).T
                sub_label = graphdata.y[sub_node]
                nodeid = 0
                with torch.no_grad():
                    data = Data(x=sub_feature, edge_index=sub_edge_index).to(args.device)
                    _, sub_output, sub_embed, _ = model(data)
                    origin_pred = sub_output[nodeid]

                explainer.eval()
                masked_pred, cf_pred, _, _ = explainer((nodeid, data, sub_adj, sub_embed, 1.0))
                
                auc_onenode, ndcg_onenode, real, pred  = explain_test(explainer.masked_adj, masked_pred, sub_edge_index, None, origin_pred, validation=True)
                reals.extend(real)
                preds.extend(pred)
                ndcgs.append(ndcg_onenode)

                _, related_preds_dict = metric.metric_pg_del_edges(nodeid, explainer.masked_adj, masked_pred, data, sub_label, origin_pred, sub_node, testing=False)
                
                maskimp_probs = related_preds_dict[10][0]["maskimp"]
                fidelity_complete_onenode = sum(abs(origin_pred - maskimp_probs)).item()
                fidelity_complete.append(fidelity_complete_onenode)
                masknotimp_probs = related_preds_dict[10][0]["masknotimp"]
                fidelityminus_complete_onenode = sum(abs(origin_pred - masknotimp_probs)).item()
                fidelityminus_complete.append(fidelityminus_complete_onenode)
                delimp_probs = related_preds_dict[10][0]["delimp"]
                del_fidelity_complete_onenode = sum(abs(origin_pred - delimp_probs)).item()
                del_fidelity_complete.append(del_fidelity_complete_onenode)
                retainimp_probs = related_preds_dict[10][0]["retainimp"]
                del_fidelityminus_complete_onenode = sum(abs(origin_pred - retainimp_probs)).item()
                del_fidelityminus_complete.append(del_fidelityminus_complete_onenode)

                if related_preds_dict[10][0]["origin_label"] == torch.argmax(related_preds_dict[10][0]["masked"]):
                    classify_acc = classify_acc + 1
                del data
                del sub_adj
                del sub_embed
                del related_preds_dict
                torch.cuda.empty_cache()

            classify_acc = classify_acc/len(valnodes)
        
            if len(np.unique(reals))<=1 or len(np.unique(preds))<=1:
                auc = -1
            else:
                auc = roc_auc_score(reals, preds)
            ndcg = np.mean(ndcgs)
            eps = 1e-7
            fidelityplus = np.mean(fidelity_complete)
            fidelityminus = np.mean(fidelityminus_complete)
            decline = torch.sub(fidelityplus, fidelityminus)
            F_fidelity = 2/(1/fidelityplus +1/(1/fidelityminus))

            del_fidelityplus = np.mean(del_fidelity_complete)
            del_fidelityminus = np.mean(del_fidelityminus_complete)
            del_F_fidelity = 2/(1/del_fidelityplus +1/(1/del_fidelityminus))

            del fidelity_complete
            del fidelityminus_complete
            del del_fidelity_complete
            del del_fidelityminus_complete
            torch.cuda.empty_cache()

            if epoch == 0:
                best_loss = loss
                best_decline = decline
                best_F_fidelity = F_fidelity
                best_del_F_fidelity = del_F_fidelity
            if decline >= best_decline:
                print("epoch", epoch, "saving best decline model...")
                f_train.write("saving best decline model...\n")
                best_decline = decline
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +"_BEST_decline.pt")
                torch.save(explainer.state_dict(), save_map + str(iter) + "/"  + args.model_filename +"_BEST_decline.pt")
            if F_fidelity >= best_F_fidelity:
                print("epoch", epoch, "saving best F_fidelity model...")
                f_train.write("saving best F_fidelity model...\n")
                best_F_fidelity = F_fidelity
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +"_BEST.pt")
                torch.save(explainer.state_dict(), save_map + str(iter) + "/"  + args.model_filename +"_BEST.pt")
            if del_F_fidelity >= best_del_F_fidelity:
                print("epoch", epoch, "saving best del_F_fidelity model...")
                f_train.write("saving best del_F_fidelity model...\n")
                best_del_F_fidelity = del_F_fidelity
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +"_BEST_delF.pt")
                torch.save(explainer.state_dict(), save_map + str(iter) + "/"  + args.model_filename +"_BEST_delF.pt")

            print("epoch", epoch, "loss", loss.item(), "hidden_loss",hidden_loss, "pred_loss",pred_loss, "lap_loss",lap_loss, "con_loss",con_loss, "sort_loss",sort_loss, "value_loss",value_loss, "cf_loss", cf_loss, "ndcg", ndcg, "auc", auc, "decline",decline, "F_fidelity",F_fidelity, "classify_acc",classify_acc, "del_F_fidelity",del_F_fidelity)
            f_train.write("epoch,{}".format(epoch) + ",loss,{}".format(loss) + ",ndcg,{}".format(ndcg) + ",auc,{}".format(auc)+ ",2,{}".format(fidelityplus) + ",1,{}".format(fidelityminus)  + ",decline,{}".format(decline)  + ",hidden_loss,{}".format(hidden_loss) + ",pred_loss,{}".format(pred_loss) + ",size_loss,{}".format(lap_loss)+ ",con_loss,{}".format(con_loss)+ ",sort_loss,{}".format(sort_loss) + ",value_loss,{}".format(value_loss) + ",cf_loss,{}".format(cf_loss)+ ",F_fidelity,{}".format(F_fidelity)+",classify_acc,{}".format(classify_acc)+",del_F_fidelity,{}".format(del_F_fidelity))
        else:
            start = start + args.batch_size

            print("epoch", epoch, "loss", loss.item(), "hidden_loss",hidden_loss, "pred_loss",pred_loss, "lap_loss",lap_loss, "con_loss",con_loss, "sort_loss",sort_loss, "value_loss",value_loss, "cf_loss", cf_loss)
            f_train.write("epoch,{}".format(epoch) + ",loss,{}".format(loss) + ",hidden_loss,{}".format(hidden_loss) + ",pred_loss,{}".format(pred_loss) + ",size_loss,{}".format(lap_loss)+ ",con_loss,{}".format(con_loss)+ ",sort_loss,{}".format(sort_loss) + ",value_loss,{}".format(value_loss) + ",cf_loss,{}".format(cf_loss))
        
        if dominant_index is not None:
            dominant_index_str = ""
            if isinstance(dominant_index, list):
                for i in dominant_index:
                    if dominant_index_str == "":
                        dominant_index_str = str(i)
                    else:
                        dominant_index_str = dominant_index_str+"+"+str(i)
            #print("dominant_index:", dominant_index)
            f_train.write(",dominant_index,{}".format(dominant_index_str))
        if modify_index_arr is not None:
            #print("modify_index_arr:", modify_index_arr)
            f_train.write(",modify_index,{}".format("_".join([str(midx) for midx in modify_index_arr])) )
        if coslist is not None:
            #print("cosvalue:", coslist)
            f_train.write(",cosvalue,{}".format("_".join(coslist)) +  "\n")
        if dominant_index is None or modify_index_arr is None or coslist is None:
            f_train.write("\n")
        tok = time.time()
        f_train.write("train time,{}".format(tok - tik) + "\n\n")
    f_train.close()
    torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename + "_LAST.pt")
    torch.save(explainer.state_dict(), save_map + str(iter) + "/"  + args.model_filename + "_LAST.pt")


def train(iter, args, model, explainer, extractor, save_map, trainnodes, valnodes, graphdata):
    tik = time.time()
    f_train = open(save_map + str(iter) + "/" + "train_LOG_" + args.model_filename + "_BEST.txt", "w")
    f_train_grad = open(save_map + str(iter) + "/" + "train_gradient_LOG_" + args.model_filename + "_BEST.txt", "w")
    #f_train_parameter = open(save_map + str(iter) + "/" + "train_parameter_LOG_" + args.model_filename + "_BEST.txt", "w")
    #f_train_embed = open(save_map + str(iter) + "/" + "train_embedding_LOG.txt", "w")
    t0 = args.coff_t0
    t1 = args.coff_te
    #clip_value_min = -2.0
    #clip_value_max = 2.0
    epochs = args.eepochs
    model.eval()
    explainer.train()
    #best_auc = 0.0
    #best_ndcg = 0.0
    #best_loss = 0.0
    best_decline = 0.0
    best_F_fidelity = 0.0
    best_del_F_fidelity = 0.0

    optimizer = Adam(explainer.elayers.parameters(), lr=args.elr)
    #optimizer = SGD(explainer.elayers.parameters(), lr=args.elr)
    optimizer.zero_grad()
    start = 0
    traversal_flag = False
    for epoch in range(epochs):
        loss = torch.tensor(0.0)
        pred_loss = 0
        lap_loss = 0
        con_loss = 0
        value_loss = 0
        sort_loss = 0
        hidden_loss = 0
        cf_loss = 0
        l, pl, ll, cl, hl, cfl, vl, sl = 0,0,0,0,0,0,0,0
        coslist = None
        #tmp = float(1.0*np.power(0.05,epoch/epochs))
        tmp = float(t0 * np.power(t1 / t0, epoch /epochs))
        #sub_adj, sub_feature, sub_embed, sub_label, sub_edge_index, sub_edge_label, sub_node  = None, None, None, None, None, None, None
        #data, sub_output, sub_hidden_emb = None, None, None
        #new_pred, cf_pred, new_hidden_emb, masked_adj,mask = None, None, None, None, None
        #for node in trainnodes:
        #batch_train = random.sample(trainnodes, args.batch_size)

        #print("000111",    torch.cuda.memory_allocated())
        #temp_memoy = torch.cuda.memory_allocated()
        #for node_index in range(len(trainnodes)):
        for node_index in range(start, start + args.batch_size):
            if node_index >= len(trainnodes):
                traversal_flag = True
                break
            node = trainnodes[node_index]
            if node_index%10== 0:
                print("node_index: " + str(node_index) + ", node: " + str(node))

            with torch.no_grad():
                sub_node, sub_adj, sub_edge_label = extractor.subgraph(node)
                if len(sub_node)<=0:
                    #print("node_index: " + str(node_index) + ", node: " + str(node) + ", len_sub_node: 0")
                    continue
                sub_edge_index = torch.nonzero(sub_adj).T
                data = Data(x=graphdata.x[sub_node], edge_index=sub_edge_index).to(args.device)
                #print("000222",    torch.cuda.memory_allocated()-temp_memoy)
                #temp_memoy = torch.cuda.memory_allocated()
                _, sub_output, sub_embed, sub_hidden_emb = model(data)
                #print("000333",    torch.cuda.memory_allocated()-temp_memoy)
                #temp_memoy = torch.cuda.memory_allocated()
                nodeid = 0
                sub_output = sub_output[nodeid]
                old_pred_label = torch.argmax(sub_output)

            #sub_feature = torch.Tensor(sub_feature).type(torch.float32)
            #sub_edge_index = torch.tensor([sub_adj.row, sub_adj.col], dtype=torch.int64)
            #data = Data(x=data.x[sub_node].to(args.device), edge_index=torch.nonzero(sub_adj).to(args.device))
            #_, sub_output, sub_embed, sub_hidden_emb = model(data)
            #sub_embed = torch.Tensor(sub_embed).to(args.device)
            #sub_label_tensor = torch.Tensor(sub_label)
            new_pred, cf_pred, _, new_hidden_emb = explainer((nodeid, data, sub_adj, sub_embed, tmp), training=True)
            #print("000444",    torch.cuda.memory_allocated()-temp_memoy)
            #temp_memoy = torch.cuda.memory_allocated()
            
            if "ce" in args.loss_flag:
                l,pl,ll,cl, hl = explainer.loss_ce_hidden(new_hidden_emb, sub_hidden_emb, new_pred, old_pred_label, sub_label, nodeid)
                vl = 0
                sl= 0
                cfl = 0
            elif "kl" in args.loss_flag:
                l, pl, ll, cl, hl, cfl = explainer.loss_kl_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output)
                vl = 0
                sl = 0
            elif "pl" in args.loss_flag:
                l, pl, ll, cl, vl, sl, hl, cfl = explainer.loss_pl_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output)
            elif "pdiff" in args.loss_flag:
                #l, pl, ll, cl, hl, cfl = explainer.loss_diff_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output[nodeid], random_cf_node_pred, mask)
                l, pl, ll, cl, hl, cfl = explainer.loss_diff_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output)
                vl = 0
                sl = 0  
            elif "CF" == args.loss_flag:
                l, pl, ll, cl, hl, cfl = explainer.loss_cf_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output)
                vl = 0
                sl = 0 
            loss = loss + l
            pred_loss = pred_loss + pl
            lap_loss = lap_loss + ll
            con_loss = con_loss + cl
            value_loss = value_loss + vl
            sort_loss = sort_loss + sl
            hidden_loss = hidden_loss + hl
            cf_loss = cf_loss + cfl
            #print("node",node, ", cfl",cfl.item(), ", origin_pred", sub_output[nodeid].cpu().detach().numpy())
            #print("cf_pred", cf_pred.cpu().detach().numpy(), ", factual_pred",new_pred.cpu().detach().numpy())
            #print("000666",    torch.cuda.memory_allocated()-temp_memoy)
            #temp_memoy = torch.cuda.memory_allocated()

            '''data = data.detach()
            new_hidden_emb = [emb.detach() for emb in new_hidden_emb]
            sub_hidden_emb =  [emb.detach() for emb in sub_hidden_emb]
            del data
            del sub_adj
            del sub_embed
            del new_hidden_emb
            del sub_hidden_emb'''
            torch.cuda.empty_cache()
        
        #f_train_embed.write("total_trainnodes={}\n".format(len(trainnodes)) )
        #print("total train nodes", len(trainnodes))
        if "getGrad" in optimal_method:
            losses = []
            if args.loss_flag == "pdiff_hidden_CF_LM_conn":
                losses = [pred_loss, hidden_loss, cf_loss, lap_loss, con_loss]
            elif args.loss_flag == "pdiff_CF_LM_conn":
                losses = [pred_loss, cf_loss, lap_loss, con_loss]
            dominant_index = dominant_loss[1]
            optimizer1 = MOGrad(optimizer)
            coslist, grads, grads_new = optimizer1.get_grads(losses, dominant_index)
        loss.backward()
        #f_train_grad.write("epoch={}\n".format(epoch))
        #f_train_grad.write("grads={}\n".format([list(g.cpu().numpy()) for g in grads]))
        #f_train_grad.write("grads_new={}\n".format([list(g.cpu().numpy()) for g in grads_new]))

        #torch.nn.utils.clip_grad_value_(explainer.elayers.parameters(), clip_value_max)
        #torch.nn.utils.clip_grad_value_(explainer.elayers.parameters(), clip_value_min)

        if traversal_flag:
            optimizer.step()
            optimizer.zero_grad()

            start = 0
            traversal_flag = False

            '''f_train_parameter.write("epoch={}\n".format(epoch))
            for k,v in dict(explainer.elayers.state_dict()).items():
                f_train_parameter.write("{}={}\n".format(k, v.cpu().numpy().tolist()))'''
        
            #global reals
            #global preds
            reals = []
            preds = []
            ndcgs = []
            #x_collector = XCollector()
            metric = MaskoutMetric(model, args)
            classify_acc = 0
            fidelity_complete = []
            fidelityminus_complete = []
            del_fidelity_complete = []
            del_fidelityminus_complete = []
            #batch_val = random.sample(valnodes, args.test_batch_size)
            #for node in batch_val:
            for node in valnodes:
                #use fullgraph prediction
                #origin_pred = outputs[node].to(args.device)  #unuse
                #use subgraph prediction
                sub_node, sub_adj, sub_edge_label = extractor.subgraph(node)
                if len(sub_node)<=0:
                    continue
                sub_feature = graphdata.x[sub_node]
                sub_edge_index = torch.nonzero(sub_adj).T
                sub_label = graphdata.y[sub_node]
                nodeid = 0
                with torch.no_grad():
                    data = Data(x=sub_feature, edge_index=sub_edge_index).to(args.device)
                    _, sub_output, sub_embed, _ = model(data)
                    origin_pred = sub_output[nodeid]
                    #sub_embed = torch.Tensor(sub_embed).to(args.device)

                explainer.eval()
                masked_pred, cf_pred, _, _ = explainer((nodeid, data, sub_adj, sub_embed, 1.0))
                
                auc_onenode, ndcg_onenode, real, pred  = explain_test(explainer.masked_adj, masked_pred, sub_edge_index, None, origin_pred, validation=True)
                reals.extend(real)
                preds.extend(pred)
                ndcgs.append(ndcg_onenode)

                _, related_preds_dict = metric.metric_pg_del_edges(nodeid, explainer.masked_adj, masked_pred, data, sub_label, origin_pred, sub_node, testing=False)
                #x_collector.collect_data(pred_mask, related_preds_dict[10], label=0)

                maskimp_probs = related_preds_dict[10][0]["maskimp"]
                fidelity_complete_onenode = sum(abs(origin_pred - maskimp_probs)).item()
                fidelity_complete.append(fidelity_complete_onenode)
                masknotimp_probs = related_preds_dict[10][0]["masknotimp"]
                fidelityminus_complete_onenode = sum(abs(origin_pred - masknotimp_probs)).item()
                fidelityminus_complete.append(fidelityminus_complete_onenode)
                delimp_probs = related_preds_dict[10][0]["delimp"]
                del_fidelity_complete_onenode = sum(abs(origin_pred - delimp_probs)).item()
                del_fidelity_complete.append(del_fidelity_complete_onenode)
                retainimp_probs = related_preds_dict[10][0]["retainimp"]
                del_fidelityminus_complete_onenode = sum(abs(origin_pred - retainimp_probs)).item()
                del_fidelityminus_complete.append(del_fidelityminus_complete_onenode)

                if related_preds_dict[10][0]["origin_label"] == torch.argmax(related_preds_dict[10][0]["masked"]):
                    classify_acc = classify_acc + 1
                del data
                del sub_adj
                del sub_embed
                del related_preds_dict
                torch.cuda.empty_cache()

            classify_acc = classify_acc/len(valnodes)
        
            if len(np.unique(reals))<=1 or len(np.unique(preds))<=1:
                auc = -1
            else:
                auc = roc_auc_score(reals, preds)
            ndcg = np.mean(ndcgs)
            eps = 1e-7
            #fidelityplus = x_collector.fidelity_complete
            #fidelityminus = x_collector.fidelityminus_complete
            fidelityplus = np.mean(fidelity_complete)
            fidelityminus = np.mean(fidelityminus_complete)
            #decline = fidelityplus - fidelityminus
            #F_fidelity = 2/(1/fidelityplus +1/(1-fidelityminus))
            decline = torch.sub(fidelityplus, fidelityminus)
            #F_fidelity = 2/ (torch.true_divide(1, (fidelityplus+eps)) + torch.true_divide(1, (torch.sub(1, fidelityminus)+eps)))
            F_fidelity = 2/(1/fidelityplus +1/(1/fidelityminus))

            #del_fidelityplus = x_collector.del_fidelity_complete
            #del_fidelityminus = x_collector.del_fidelityminus_complete
            del_fidelityplus = np.mean(del_fidelity_complete)
            del_fidelityminus = np.mean(del_fidelityminus_complete)
            #del_F_fidelity = 2/(1/del_fidelity +1/(1-del_fidelityminus))
            #del_F_fidelity = 2/ (torch.true_divide(1, del_fidelityplus) + torch.true_divide(1, torch.sub(1, del_fidelityminus)))
            del_F_fidelity = 2/(1/del_fidelityplus +1/(1/del_fidelityminus))

            del fidelity_complete
            del fidelityminus_complete
            del del_fidelity_complete
            del del_fidelityminus_complete
            torch.cuda.empty_cache()

            if epoch == 0:
                best_loss = loss
                #best_auc = auc
                best_decline = decline
                best_F_fidelity = F_fidelity
                best_del_F_fidelity = del_F_fidelity
            '''if auc >= best_auc:
                print("epoch", epoch, "saving best auc model...")
                f_train.write("saving best auc model...\n")
                best_auc = auc
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename + "_BEST_auc.pt")
                torch.save(explainer.state_dict(), save_map + str(iter) + "/"  + args.model_filename + "_BEST_auc.pt")'''
            #if ndcg > best_ndcg:
            #if loss <= best_loss:
            if decline >= best_decline:
                print("epoch", epoch, "saving best decline model...")
                f_train.write("saving best decline model...\n")
                #best_ndcg = ndcg
                #best_loss = loss
                best_decline = decline
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +"_BEST_decline.pt")
                torch.save(explainer.state_dict(), save_map + str(iter) + "/"  + args.model_filename +"_BEST_decline.pt")
                #best_state_dict = explainer.state_dict().clone()
            if F_fidelity >= best_F_fidelity:
                print("epoch", epoch, "saving best F_fidelity model...")
                f_train.write("saving best F_fidelity model...\n")
                best_F_fidelity = F_fidelity
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +"_BEST.pt")
                torch.save(explainer.state_dict(), save_map + str(iter) + "/"  + args.model_filename +"_BEST.pt")
            if del_F_fidelity >= best_del_F_fidelity:
                print("epoch", epoch, "saving best del_F_fidelity model...")
                f_train.write("saving best del_F_fidelity model...\n")
                best_del_F_fidelity = del_F_fidelity
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +"_BEST_delF.pt")
                torch.save(explainer.state_dict(), save_map + str(iter) + "/"  + args.model_filename +"_BEST_delF.pt")

            print("epoch", epoch, "loss", loss.item(), "hidden_loss",hidden_loss, "pred_loss",pred_loss, "lap_loss",lap_loss, "con_loss",con_loss, "sort_loss",sort_loss, "value_loss",value_loss, "cf_loss", cf_loss, "ndcg", ndcg, "auc", auc, "decline",decline, "F_fidelity",F_fidelity, "classify_acc",classify_acc, "del_F_fidelity",del_F_fidelity)
            f_train.write("epoch,{}".format(epoch) + ",loss,{}".format(loss) + ",ndcg,{}".format(ndcg) + ",auc,{}".format(auc)+ ",2,{}".format(fidelityplus) + ",1,{}".format(fidelityminus)  + ",decline,{}".format(decline)  + ",hidden_loss,{}".format(hidden_loss) + ",pred_loss,{}".format(pred_loss) + ",size_loss,{}".format(lap_loss)+ ",con_loss,{}".format(con_loss)+ ",sort_loss,{}".format(sort_loss) + ",value_loss,{}".format(value_loss) + ",cf_loss,{}".format(cf_loss)+ ",F_fidelity,{}".format(F_fidelity)+",classify_acc,{}".format(classify_acc)+",del_F_fidelity,{}".format(del_F_fidelity))
        else:
            start = start + args.batch_size

            print("epoch", epoch, "loss", loss.item(), "hidden_loss",hidden_loss, "pred_loss",pred_loss, "lap_loss",lap_loss, "con_loss",con_loss, "sort_loss",sort_loss, "value_loss",value_loss, "cf_loss", cf_loss)
            f_train.write("epoch,{}".format(epoch) + ",loss,{}".format(loss) + ",hidden_loss,{}".format(hidden_loss) + ",pred_loss,{}".format(pred_loss) + ",size_loss,{}".format(lap_loss)+ ",con_loss,{}".format(con_loss)+ ",sort_loss,{}".format(sort_loss) + ",value_loss,{}".format(value_loss) + ",cf_loss,{}".format(cf_loss))
        
        if coslist is not None:
            #print("cosvalue:", coslist)
            f_train.write(",cosvalue,{}".format("_".join(coslist)) +  "\n")
        else:
            f_train.write("\n")
        tok = time.time()
        f_train.write("train time,{}".format(tok - tik) + "\n\n")
    f_train.close()
    #f_train_grad.close()
    #f_train_parameter.close()
    #f_train_embed.close()
    torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename + "_LAST.pt")
    torch.save(explainer.state_dict(), save_map + str(iter) + "/"  + args.model_filename + "_LAST.pt")


def calculate_fidelity(ori_prob: torch.Tensor, unimportant_prob: torch.Tensor) -> float:
        drop_probability = abs(ori_prob - unimportant_prob)
        return drop_probability.mean().item()

def calculate_fidelity_complete(ori_probs, important_probs):
    drop_prob_complete = [abs(ori_probs[i]-important_probs[i]) for i in range(len(ori_probs))]
    result = sum(drop_prob_complete).item()
    return result


def test_main_onetopk(topk, save_map, dataset_filename, testmodel_filename, plot_flag):
    log_filename = save_map + "log_test_topk" + str(topk) + ".txt"
    f_mean = open(log_filename, "w")
    auc_all = []
    ndcg_all = []
    PN_all = []
    PS_all = []
    FNS_all = []
    size_all = []
    acc_all = []
    pre_all = []
    rec_all = []
    f1_all = []
    simula_arr = []
    simula_origin_arr = []
    simula_complete_arr = []
    fidelity_arr = []
    fidelity_origin_arr = []
    fidelity_complete_arr = []
    fidelityminus_arr = []
    fidelityminus_origin_arr = []
    fidelityminus_complete_arr = []
    finalfidelity_complete_arr = []
    fvaluefidelity_complete_arr = []
    del_fidelity_arr = []
    del_fidelity_origin_arr = []
    del_fidelity_complete_arr = []
    del_fidelityminus_arr = []
    del_fidelityminus_origin_arr = []
    del_fidelityminus_complete_arr = []
    del_finalfidelity_complete_arr = []
    del_fvaluefidelity_complete_arr = []
    sparsity_edges_arr = []
    fidelity_nodes_arr = []
    fidelity_origin_nodes_arr = []
    fidelity_complete_nodes_arr = []
    fidelityminus_nodes_arr = []
    fidelityminus_origin_nodes_arr = []
    fidelityminus_complete_nodes_arr = []
    finalfidelity_complete_nodes_arr = []
    sparsity_nodes_arr = []
    for iter in range(1):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        #args.device = "cpu"
        print("Starting iteration: {}".format(iter))
        if not os.path.exists(save_map+str(iter)):
            os.makedirs(save_map+str(iter))
        #load data
        if args.dataset == "Computers":
            dataset = Amazon(root=args.dataset_root, name=args.dataset)
        #data = dataset.data
        num_class = dataset.num_classes
        label = dataset.data.y
        adj =  torch.sparse_coo_tensor(indices=dataset.data.edge_index, values= torch.ones(dataset.data.edge_index.shape[1])).to_dense()
        #adj = csr_matrix(adj.numpy())
        edge_label_matrix = None
        num_nodes = dataset.data.x.shape[0]
        node_index = list(range(num_nodes))
        random.seed(2023)
        random.shuffle(node_index)
        trainnodes = node_index[ : math.floor(num_nodes*0.8)]
        valnodes = node_index[math.floor(num_nodes*0.8) : math.floor(num_nodes*0.9)]
        testnodes_0 = node_index[math.floor(num_nodes*0.9) : ]

        #load trained GCN (survey)
        GNNmodel_ckpt_path = osp.join('GNN_checkpoint', args.dataset+"_"+str(iter), 'gcn_best.pth') 
        model = load_gnnNets_NC(GNNmodel_ckpt_path, input_dim=dataset.data.x.shape[1], output_dim=num_class, device = args.device)
        model.eval()

        hops = len(args.hiddens.split('-'))
        #extractor = Extractor(adj, edge_index, features, edge_label_matrix, embeds, label, hops)
        extractor = Extractor(adj, edge_label_matrix, hops)
        
        explainer = ExplainerMO(model=model, args=args)
        explainer.to(args.device)

        testnodes = []
        for node in testnodes_0:
            sub_node, _, _ = extractor.subgraph(node)
            if len(sub_node) != 0:
                testnodes.append(node)
        print("filted test nodes count:", len(testnodes))

        f = open(save_map + str(iter) + "/" + "LOG_" + testmodel_filename + "_test_topk" + str(topk) + ".txt", "w")
        savedDict = torch.load(save_map + str(iter) + "/" + testmodel_filename + ".pt") 
        explainer.load_state_dict(savedDict)

        f.write("test result.")
        # metrics
        tik = time.time()
        reals = []
        preds = []
        ndcgs = []
        #exp_dict={}
        #pred_label_dict={}
        #feature_dict = {}
        #adj_dict = {}
        #e_labels_dict = {}
        PN, PS, ave_size = 0, 0, 0
        acc_1, pre, rec, f1 =0, 0, 0, 0
        #allnode_related_preds_dict = dict()
        #allnode_mask_dict = dict()
        metric = MaskoutMetric(model, args)
        plotutils = PlotUtils(dataset_name=args.dataset)
        
        one_simula_arr = []
        one_simula_origin_arr = []
        one_simula_complete_arr = []
        one_fidelity_arr = []
        one_fidelity_origin_arr = []
        one_fidelity_complete_arr = []
        one_fidelityminus_arr = []
        one_fidelityminus_origin_arr = []
        one_fidelityminus_complete_arr = []
        one_finalfidelity_complete_arr = []
        one_fvaluefidelity_complete_arr = []
        one_del_fidelity_arr = []
        one_del_fidelity_origin_arr = []
        one_del_fidelity_complete_arr = []
        one_del_fidelityminus_arr = []
        one_del_fidelityminus_origin_arr = []
        one_del_fidelityminus_complete_arr = []
        one_del_finalfidelity_complete_arr = []
        one_del_fvaluefidelity_complete_arr = []
        one_sparsity_edges_arr = []
        one_fidelity_nodes_arr = []
        one_fidelity_origin_nodes_arr = []
        one_fidelity_complete_nodes_arr = []
        one_fidelityminus_nodes_arr = []
        one_fidelityminus_origin_nodes_arr = []
        one_fidelityminus_complete_nodes_arr = []
        one_finalfidelity_complete_nodes_arr = []
        one_sparsity_nodes_arr = []
        for top_k in [topk]:
            print("top_k",top_k)
            fidelity_complete = []
            fidelity = []
            fidelity_origin = []
            fidelityminus_complete = []
            fidelityminus = []
            fidelityminus_origin = []
            del_fidelity_complete = []
            del_fidelity = []
            del_fidelity_origin = []
            del_fidelityminus_complete = []
            del_fidelityminus = []
            del_fidelityminus_origin = []
            simula_complete = []
            simula = []
            simula_origin = []
            sparsity_edges = []
            fidelity_nodes = []
            fidelity_origin_nodes = []
            fidelity_complete_nodes = []
            fidelityminus_nodes = []
            fidelityminus_origin_nodes = []
            fidelityminus_complete_nodes = []
            sparsity_nodes = []

            #x_collector = XCollector()
            #for node in testnodes:
            for j in range(len(testnodes)):
                node = testnodes[j]
                print("node index", j, "test node", node)
                sub_node, sub_adj,sub_edge_label = extractor.subgraph(node)
                if len(sub_node) == 0:
                    continue
                sub_edge_index = torch.nonzero(sub_adj).T
                sub_feature = dataset.data.x[sub_node]
                data = Data(x=sub_feature, edge_index=sub_edge_index).to(args.device)
                with torch.no_grad():
                    _, sub_output, sub_embed, _ = model(data)
                
                #use fullgraph prediction
                #origin_pred = outputs[node].to(args.device)
                #use subgraph prediction
                nodeid = 0
                origin_pred = sub_output[nodeid]
                origin_label = origin_pred.argmax(-1)
                explainer.eval()
                with torch.no_grad():
                    masked_pred, _, _, _ = explainer((nodeid, data, sub_adj, sub_embed, 1.0))
                sub_label = dataset.data.y[sub_node]
                masked_adj = explainer.masked_adj

                pred_mask, related_preds_dict = metric.metric_pg_del_edges(nodeid, masked_adj, masked_pred, data, sub_label, origin_pred, sub_node, topk_arr = [top_k])
                related_preds = related_preds_dict[top_k]

                #mask the selected edges
                fidelity_complete.append(calculate_fidelity_complete(related_preds[0]["origin"], related_preds[0]["maskimp"]))
                fidelity.append(calculate_fidelity(related_preds[0]["origin_l"], related_preds[0]["maskimp_l"]))
                fidelity_origin.append(calculate_fidelity(related_preds[0]["origin_ol"], related_preds[0]["maskimp_ol"]))

                #mask other edges except the selected edges
                fidelityminus_complete.append(calculate_fidelity_complete(related_preds[0]["origin"], related_preds[0]["masknotimp"]))
                fidelityminus.append(calculate_fidelity(related_preds[0]["origin_l"], related_preds[0]["masknotimp_l"]))
                fidelityminus_origin.append(calculate_fidelity(related_preds[0]["origin_ol"], related_preds[0]["masknotimp_ol"]))

                #delete the important edges
                del_fidelity_complete.append(calculate_fidelity_complete(related_preds[0]["origin"], related_preds[0]["delimp"]))
                del_fidelity.append(calculate_fidelity(related_preds[0]["origin_l"], related_preds[0]["delimp_l"]))
                del_fidelity_origin.append(calculate_fidelity(related_preds[0]["origin_ol"], related_preds[0]["delimp_ol"]))

                #retain the important edges 
                del_fidelityminus_complete.append(calculate_fidelity_complete(related_preds[0]["origin"], related_preds[0]["retainimp"]))
                del_fidelityminus.append(calculate_fidelity(related_preds[0]["origin_l"], related_preds[0]["retainimp_l"]))
                del_fidelityminus_origin.append(calculate_fidelity(related_preds[0]["origin_ol"], related_preds[0]["retainimp_ol"]))

                simula_complete.append( calculate_fidelity_complete(related_preds[0]["origin"], related_preds[0]["masked"]))
                simula.append(calculate_fidelity(related_preds[0]["origin_l"], related_preds[0]["masked_l"]))
                simula_origin.append(calculate_fidelity(related_preds[0]["origin_ol"], related_preds[0]["masked_ol"]))
                sparsity_edges.append(related_preds[0]["sparsity_edges"])

                #mask the selected nodes
                fidelity_complete_nodes.append(calculate_fidelity_complete(related_preds[0]["origin"], related_preds[0]["maskimp_nodes"]))
                fidelity_nodes.append(calculate_fidelity(related_preds[0]["origin_l"], related_preds[0]["maskimp_nodes_l"]))
                fidelity_origin_nodes.append(calculate_fidelity(related_preds[0]["origin_ol"], related_preds[0]["maskimp_nodes_ol"]))

                #mask other nodes except the selected nodes
                fidelityminus_complete_nodes.append(calculate_fidelity_complete(related_preds[0]["origin"], related_preds[0]["retainimp_nodes"]))
                fidelityminus_nodes.append(calculate_fidelity(related_preds[0]["origin_l"], related_preds[0]["retainimp_nodes_l"]))
                fidelityminus_origin_nodes.append(calculate_fidelity(related_preds[0]["origin_ol"], related_preds[0]["retainimp_nodes_ol"]))
                sparsity_nodes.append(related_preds[0]["sparsity_nodes"])

                #allnode_related_preds_dict[node] = related_preds_dict
                #allnode_mask_dict[node] = pred_mask
                #related_preds = allnode_related_preds_dict[node][top_k]
                #mask = allnode_mask_dict[node]
                
                #x_collector.collect_data(pred_mask, related_preds)
                f.write("node,{}\n".format(node))
                f.write("mask,{}\n".format(pred_mask))
                f.write("related_preds,{}\n".format(related_preds))

                auc_onenode, ndcg_onenode, real, pred  = explain_test(masked_adj, masked_pred, sub_edge_index, None, origin_pred)
                ndcgs.append(ndcg_onenode)
                reals.extend(real)
                preds.extend(pred)

                ps, pn, size= compute_pnps_NC_onenode(masked_adj, origin_label, model, sub_feature, sub_adj, nodeid, args.fix_exp, args.mask_thresh)
                PN += pn
                PS += ps
                ave_size += size
                if args.dataset == "BA_shapes" or args.dataset == "BA_community":
                    acc_onenode_1, pre_onenode, rec_onenode, f1_onenode = compute_precision_recall_NC_onenode(masked_adj, sub_edge_label, sub_edge_index, args.fix_exp, args.mask_thresh)
                else:
                    acc_onenode_1, pre_onenode, rec_onenode, f1_onenode = 0, 0, 0, 0
                acc_1 += acc_onenode_1
                pre += pre_onenode
                rec += rec_onenode
                f1 += f1_onenode

                if plot_flag:
                    #plot(node, label, iter)
                    # visualization
                    edge_mask = masked_adj[sub_adj.row, sub_adj.col].detach()
                    edges_idx_desc = edge_mask.reshape(-1).sort(descending=True).indices
                    important_nodelist = []
                    important_edgelist = []
                    for idx in edges_idx_desc:
                        if len(important_edgelist)<12:
                            if (sub_edge_index[0][idx].item(), sub_edge_index[1][idx].item()) not in important_edgelist:
                                important_nodelist.append(sub_edge_index[0][idx].item())
                                important_nodelist.append(sub_edge_index[1][idx].item())
                                important_edgelist.append((sub_edge_index[0][idx].item(), sub_edge_index[1][idx].item()))
                                important_edgelist.append((sub_edge_index[1][idx].item(), sub_edge_index[0][idx].item()))
                    important_nodelist = list(set(important_nodelist))
                    data = Data(x=sub_feature, edge_index=sub_edge_index, y =sub_label.argmax(-1) )
                    ori_graph = to_networkx(data, to_undirected=True)
                    plotutils.plot_new(ori_graph, nodelist=important_nodelist, edgelist=important_edgelist, y=sub_label.argmax(-1), node_idx=nodeid,
                                figname=os.path.join(save_map + str(iter), f"node_{node}_new.png"))

                tok = time.time()
                f.write("node,{}".format(node) + ",auc_onenode,{}".format(auc_onenode) + ",ndcg_onenode,{}".format(ndcg_onenode)  + ",time,{}".format(tok - tik) + "\n")
            
            one_simula_arr.append(round(np.mean(simula), 4))
            one_simula_origin_arr.append(round(np.mean(simula_origin), 4))
            one_simula_complete_arr.append(round(np.mean(simula_complete), 4))
            one_fidelity_arr.append(round(np.mean(fidelity), 4))
            one_fidelity_origin_arr.append(round(np.mean(fidelity_origin), 4))
            one_fidelity_complete_arr.append(round(np.mean(fidelity_complete), 4))
            one_fidelityminus_arr.append(round(np.mean(fidelityminus), 4))
            one_fidelityminus_origin_arr.append(round(np.mean(fidelityminus_origin), 4))
            one_fidelityminus_complete_arr.append(round(np.mean(fidelityminus_complete), 4))
            one_finalfidelity_complete_arr.append(round(np.mean(fidelity_complete) - np.mean(fidelityminus_complete), 4))
            F_fidelity = 2/(1/np.mean(fidelity_complete) +1/(1/np.mean(fidelityminus_complete)))
            one_fvaluefidelity_complete_arr.append(round(F_fidelity, 4))
            one_del_fidelity_arr.append(round(np.mean(del_fidelity), 4))
            one_del_fidelity_origin_arr.append(round(np.mean(del_fidelity_origin), 4))
            one_del_fidelity_complete_arr.append(round(np.mean(del_fidelity_complete), 4))
            one_del_fidelityminus_arr.append(round(np.mean(del_fidelityminus), 4))
            one_del_fidelityminus_origin_arr.append(round(np.mean(del_fidelityminus_origin), 4))
            one_del_fidelityminus_complete_arr.append(round(np.mean(del_fidelityminus_complete), 4))
            one_del_finalfidelity_complete_arr.append(round(np.mean(del_fidelity_complete) - np.mean(del_fidelityminus_complete), 4))
            del_F_fidelity = 2/(1/np.mean(del_fidelity_complete) +1/(1/np.mean(del_fidelityminus_complete)))
            one_del_fvaluefidelity_complete_arr.append(round(del_F_fidelity, 4))
            one_sparsity_edges_arr.append(round(np.mean(sparsity_edges), 4))
            one_fidelity_nodes_arr.append(round(np.mean(fidelity_nodes), 4))
            one_fidelity_origin_nodes_arr.append(round(np.mean(fidelity_origin_nodes), 4))
            one_fidelity_complete_nodes_arr.append(round(np.mean(fidelity_complete_nodes), 4))
            one_fidelityminus_nodes_arr.append(round(np.mean(fidelityminus_nodes), 4))
            one_fidelityminus_origin_nodes_arr.append(round(np.mean(fidelityminus_origin_nodes), 4))
            one_fidelityminus_complete_nodes_arr.append(round(np.mean(fidelityminus_complete_nodes), 4))
            one_finalfidelity_complete_nodes_arr.append(round(np.mean(fidelity_complete_nodes) - np.mean(fidelityminus_complete_nodes), 4))
            one_sparsity_nodes_arr.append(round(np.mean(sparsity_nodes), 4))

        if args.dataset == "Computers":
            auc=0
        else:
            if len(np.unique(reals))==1 or len(np.unique(preds))==1:
                auc = -1
            else:
                auc = roc_auc_score(reals, preds)
        ndcg = np.mean(ndcgs)
        auc_all.append(auc)
        ndcg_all.append(ndcg)
        tok = time.time()
        f.write("iter,{}".format(iter) + ",auc,{}".format(auc) + ",ndcg,{}".format(ndcg) + ",time,{}".format(tok - tik) + "\n")
        
        PN = PN/len(testnodes)
        PS = PS/len(testnodes)
        ave_size = ave_size/len(testnodes)

        if PN + PS==0:
            FNS=0
        else:
            FNS = 2 * PN * PS / (PN + PS)
        acc_1 = acc_1/len(testnodes)
        pre = pre/len(testnodes)
        rec = rec/len(testnodes)
        f1 = f1/len(testnodes)

        PN_all.append(PN)
        PS_all.append(PS)
        FNS_all.append(FNS)
        size_all.append(ave_size)
        acc_all.append(acc_1)
        pre_all.append(pre)
        rec_all.append(rec)
        f1_all.append(f1)
        
        print("one_auc=", auc)
        print("one_ndcg=", ndcg)
        print("one_simula_arr =", one_simula_arr)
        print("one_simula_origin_arr =", one_simula_origin_arr)
        print("one_simula_complete_arr =", one_simula_complete_arr)
        print("one_fidelity_arr =", one_fidelity_arr)
        print("one_fidelity_origin_arr =", one_fidelity_origin_arr)
        print("one_fidelity_complete_arr =", one_fidelity_complete_arr)
        print("one_fidelityminus_arr=", one_fidelityminus_arr)
        print("one_fidelityminus_origin_arr=", one_fidelityminus_origin_arr)
        print("one_fidelityminus_complete_arr=", one_fidelityminus_complete_arr)
        print("one_finalfidelity_complete_arr=", one_finalfidelity_complete_arr)
        print("one_fvaluefidelity_complete_arr=", one_fvaluefidelity_complete_arr)
        print("one_del_fidelity_arr =", one_del_fidelity_arr)
        print("one_del_fidelity_origin_arr =", one_del_fidelity_origin_arr)
        print("one_del_fidelity_complete_arr =", one_del_fidelity_complete_arr)
        print("one_del_fidelityminus_arr=", one_del_fidelityminus_arr)
        print("one_del_fidelityminus_origin_arr=", one_del_fidelityminus_origin_arr)
        print("one_del_fidelityminus_complete_arr=", one_del_fidelityminus_complete_arr)
        print("one_del_finalfidelity_complete_arr=", one_del_finalfidelity_complete_arr)
        print("one_del_fvaluefidelity_complete_arr=", one_del_fvaluefidelity_complete_arr)
        print("one_sparsity_edges_arr =", one_sparsity_edges_arr)
        print("one_fidelity_nodes_arr =", one_fidelity_nodes_arr)
        print("one_fidelity_origin_nodes_arr =", one_fidelity_origin_nodes_arr)
        print("one_fidelity_complete_nodes_arr =", one_fidelity_complete_nodes_arr)
        print("one_fidelityminus_nodes_arr=", one_fidelityminus_nodes_arr)
        print("one_fidelityminus_origin_nodes_arr=", one_fidelityminus_origin_nodes_arr)
        print("one_fidelityminus_complete_nodes_arr=", one_fidelityminus_complete_nodes_arr)
        print("one_finalfidelity_complete_nodes_arr=", one_finalfidelity_complete_nodes_arr)
        print("one_sparsity_nodes_arr =", one_sparsity_nodes_arr)

        tok = time.time()
        f.write("one_auc={}".format(auc) + "\n")
        f.write("one_ndcg={}".format(ndcg) + "\n")
        f.write("one_simula={}".format(one_simula_arr) + "\n")
        f.write("one_simula_orign={}".format(one_simula_origin_arr) + "\n")
        f.write("one_simula_complete={}".format(one_simula_complete_arr) + "\n")
        f.write("one_fidelity={}".format(one_fidelity_arr) + "\n")
        f.write("one_fidelity_orign={}".format(one_fidelity_origin_arr) + "\n")
        f.write("one_fidelity_complete={}".format(one_fidelity_complete_arr) + "\n")
        f.write("one_fidelityminus={}".format(one_fidelityminus_arr)+"\n")
        f.write("one_fidelityminus_origin={}".format(one_fidelityminus_origin_arr)+"\n")
        f.write("one_fidelityminus_complete={}".format(one_fidelityminus_complete_arr)+"\n")
        f.write("one_finalfidelity_complete={}".format(one_finalfidelity_complete_arr)+"\n")
        f.write("one_fvaluefidelity_complete={}".format(one_fvaluefidelity_complete_arr)+"\n")
        f.write("one_del_fidelity={}".format(one_del_fidelity_arr) + "\n")
        f.write("one_del_fidelity_orign={}".format(one_del_fidelity_origin_arr) + "\n")
        f.write("one_del_fidelity_complete={}".format(one_del_fidelity_complete_arr) + "\n")
        f.write("one_del_fidelityminus={}".format(one_del_fidelityminus_arr)+"\n")
        f.write("one_del_fidelityminus_origin={}".format(one_del_fidelityminus_origin_arr)+"\n")
        f.write("one_del_fidelityminus_complete={}".format(one_del_fidelityminus_complete_arr)+"\n")
        f.write("one_del_finalfidelity_complete={}".format(one_del_finalfidelity_complete_arr)+"\n")
        f.write("one_del_fvaluefidelity_complete={}".format(one_del_fvaluefidelity_complete_arr)+"\n")
        f.write("one_sparsity_edges={}".format(one_sparsity_edges_arr) + "\n")
        f.write("one_fidelity_nodes={}".format(one_fidelity_nodes_arr) + "\n")
        f.write("one_fidelity_origin_nodes={}".format(one_fidelity_origin_nodes_arr) + "\n")
        f.write("one_fidelity_complete_nodes={}".format(one_fidelity_complete_nodes_arr) + "\n")
        f.write("one_fidelityminus_nodes={}".format(one_fidelityminus_nodes_arr)+"\n")
        f.write("one_fidelityminus_origin_nodes={}".format(one_fidelityminus_origin_nodes_arr)+"\n")
        f.write("one_fidelityminus_complete_nodes={}".format(one_fidelityminus_complete_nodes_arr)+"\n")
        f.write("one_finalfidelity_complete_nodes={}".format(one_finalfidelity_complete_nodes_arr)+"\n")
        f.write("one_sparsity_nodes={}".format(one_sparsity_nodes_arr) + "\n")
        f.write("test time,{}".format(tok-tik))
        f.close()

        simula_arr.append(one_simula_arr)
        simula_origin_arr.append(one_simula_origin_arr)
        simula_complete_arr.append(one_simula_complete_arr)
        fidelity_arr.append(one_fidelity_arr)
        fidelity_origin_arr.append(one_fidelity_origin_arr)
        fidelity_complete_arr.append(one_fidelity_complete_arr)
        fidelityminus_arr.append(one_fidelityminus_arr)
        fidelityminus_origin_arr.append(one_fidelityminus_origin_arr)
        fidelityminus_complete_arr.append(one_fidelityminus_complete_arr)
        finalfidelity_complete_arr.append(one_finalfidelity_complete_arr)
        fvaluefidelity_complete_arr.append(one_fvaluefidelity_complete_arr)
        del_fidelity_arr.append(one_del_fidelity_arr)
        del_fidelity_origin_arr.append(one_del_fidelity_origin_arr)
        del_fidelity_complete_arr.append(one_del_fidelity_complete_arr)
        del_fidelityminus_arr.append(one_del_fidelityminus_arr)
        del_fidelityminus_origin_arr.append(one_del_fidelityminus_origin_arr)
        del_fidelityminus_complete_arr.append(one_del_fidelityminus_complete_arr)
        del_finalfidelity_complete_arr.append(one_del_finalfidelity_complete_arr)
        del_fvaluefidelity_complete_arr.append(one_del_fvaluefidelity_complete_arr)
        sparsity_edges_arr.append(one_sparsity_edges_arr)
        fidelity_nodes_arr.append(one_fidelity_nodes_arr)
        fidelity_origin_nodes_arr.append(one_fidelity_origin_nodes_arr)
        fidelity_complete_nodes_arr.append(one_fidelity_complete_nodes_arr)
        fidelityminus_nodes_arr.append(one_fidelityminus_nodes_arr)
        fidelityminus_origin_nodes_arr.append(one_fidelityminus_origin_nodes_arr)
        fidelityminus_complete_nodes_arr.append(one_fidelityminus_complete_nodes_arr)
        finalfidelity_complete_nodes_arr.append(one_finalfidelity_complete_nodes_arr)
        sparsity_nodes_arr.append(one_sparsity_nodes_arr)

    print("args.dataset", args.dataset)
    print("MO_auc_all = ", auc_all)
    print("MO_ndcg_all = ", ndcg_all)
    print("MO_simula_arr =", simula_arr)
    print("MO_simula_origin_arr =", simula_origin_arr)
    print("MO_simula_complete_arr =", simula_complete_arr)
    print("MO_fidelity_arr =", fidelity_arr)
    print("MO_fidelity_origin_arr =", fidelity_origin_arr)
    print("MO_fidelity_complete_arr =", fidelity_complete_arr)
    print("MO_fidelityminus_arr=", fidelityminus_arr)
    print("MO_fidelityminus_origin_arr=", fidelityminus_origin_arr)
    print("MO_fidelityminus_complete_arr=", fidelityminus_complete_arr)
    print("MO_finalfidelity_complete_arr", finalfidelity_complete_arr)
    print("MO_fvaluefidelity_complete_arr", fvaluefidelity_complete_arr)
    print("MO_del_fidelity_arr =", del_fidelity_arr)
    print("MO_del_fidelity_origin_arr =", del_fidelity_origin_arr)
    print("MO_del_fidelity_complete_arr =", del_fidelity_complete_arr)
    print("MO_del_fidelityminus_arr=", del_fidelityminus_arr)
    print("MO_del_fidelityminus_origin_arr=", del_fidelityminus_origin_arr)
    print("MO_del_fidelityminus_complete_arr=", del_fidelityminus_complete_arr)
    print("MO_del_finalfidelity_complete_arr", del_finalfidelity_complete_arr)
    print("MO_del_fvaluefidelity_complete_arr", del_fvaluefidelity_complete_arr)
    print("MO_sparsity_edges_arr =", sparsity_edges_arr)
    print("MO_fidelity_nodes_arr =", fidelity_nodes_arr)
    print("MO_fidelity_origin_nodes_arr =", fidelity_origin_nodes_arr)
    print("MO_fidelity_complete_nodes_arr =", fidelity_complete_nodes_arr)
    print("MO_fidelityminus_nodes_arr=", fidelityminus_nodes_arr)
    print("MO_fidelityminus_origin_nodes_arr=", fidelityminus_origin_nodes_arr)
    print("MO_fidelityminus_complete_nodes_arr=", fidelityminus_complete_nodes_arr)
    print("MO_finalfidelity_complete_nodes_arr", finalfidelity_complete_nodes_arr)
    print("MO_sparsity_nodes_arr =", sparsity_nodes_arr)

    f_mean.write("MO_auc_all={}".format(auc_all) + "\n")
    f_mean.write("MO_ndcg_all={}".format(ndcg_all) + "\n")
    f_mean.write("MO_PN_all={}".format(PN_all) + "\n")
    f_mean.write("MO_PS_all={}".format(PS_all) + "\n")
    f_mean.write("MO_FNS_all={}".format(FNS_all) + "\n")
    f_mean.write("MO_size_all={}".format(size_all) + "\n")
    f_mean.write("MO_acc_all={}".format(acc_all) + "\n")
    f_mean.write("MO_pre_all={}".format(pre_all) + "\n")
    f_mean.write("MO_rec_all={}".format(rec_all) + "\n")
    f_mean.write("MO_f1_all={}".format(f1_all) + "\n")
    f_mean.write("MO_simula_arr={}".format(simula_arr) + "\n")
    f_mean.write("MO_simula_origin_arr={}".format(simula_origin_arr) + "\n")
    f_mean.write("MO_simula_complete_arr={}".format(simula_complete_arr) + "\n")
    f_mean.write("MO_fidelity_arr={}".format(fidelity_arr) + "\n")
    f_mean.write("MO_fidelity_origin_arr={}".format(fidelity_origin_arr) + "\n")
    f_mean.write("MO_fidelity_complete_arr={}".format(fidelity_complete_arr) + "\n")
    f_mean.write("MO_fidelityminus_arr = {}".format(fidelityminus_arr)+"\n")
    f_mean.write("MO_fidelityminus_origin_arr = {}".format(fidelityminus_origin_arr)+"\n")
    f_mean.write("MO_fidelityminus_complete_arr = {}".format(fidelityminus_complete_arr)+"\n")
    f_mean.write("MO_finalfidelity_complete_arr = {}".format(finalfidelity_complete_arr)+"\n")
    f_mean.write("MO_fvaluefidelity_complete_arr = {}".format(fvaluefidelity_complete_arr)+"\n")
    f_mean.write("MO_del_fidelity_arr={}".format(del_fidelity_arr) + "\n")
    f_mean.write("MO_del_fidelity_origin_arr={}".format(del_fidelity_origin_arr) + "\n")
    f_mean.write("MO_del_fidelity_complete_arr={}".format(del_fidelity_complete_arr) + "\n")
    f_mean.write("MO_del_fidelityminus_arr = {}".format(del_fidelityminus_arr)+"\n")
    f_mean.write("MO_del_fidelityminus_origin_arr = {}".format(del_fidelityminus_origin_arr)+"\n")
    f_mean.write("MO_del_fidelityminus_complete_arr = {}".format(del_fidelityminus_complete_arr)+"\n")
    f_mean.write("MO_del_finalfidelity_complete_arr = {}".format(del_finalfidelity_complete_arr)+"\n")
    f_mean.write("MO_del_fvaluefidelity_complete_arr = {}".format(del_fvaluefidelity_complete_arr)+"\n")
    f_mean.write("MO_sparsity_edges_arr={}".format(sparsity_edges_arr) + "\n")
    f_mean.write("MO_fidelity_nodes_arr={}".format(fidelity_nodes_arr) + "\n")
    f_mean.write("MO_fidelity_origin_nodes_arr={}".format(fidelity_origin_nodes_arr) + "\n")
    f_mean.write("MO_fidelity_complete_nodes_arr={}".format(fidelity_complete_nodes_arr) + "\n")
    f_mean.write("MO_fidelityminus_nodes_arr = {}".format(fidelityminus_nodes_arr)+"\n")
    f_mean.write("MO_fidelityminus_origin_nodes_arr = {}".format(fidelityminus_origin_nodes_arr)+"\n")
    f_mean.write("MO_fidelityminus_complete_nodes_arr = {}".format(fidelityminus_complete_nodes_arr)+"\n")
    f_mean.write("MO_finalfidelity_complete_nodes_arr = {}".format(finalfidelity_complete_nodes_arr)+"\n")
    f_mean.write("MO_sparsity_nodes_arr={}".format(sparsity_nodes_arr) + "\n")

    simula_mean = np.average(np.array(simula_arr), axis=0)
    simula_origin_mean = np.average(np.array(simula_origin_arr), axis=0)
    simula_complete_mean = np.average(np.array(simula_complete_arr),axis=0)
    fidelity_mean = np.average(np.array(fidelity_arr),axis=0)
    fidelity_origin_mean = np.average(np.array(fidelity_origin_arr),axis=0)
    fidelity_complete_mean = np.average(np.array(fidelity_complete_arr),axis=0)
    fidelityminus_mean = np.average(np.array(fidelityminus_arr),axis=0)
    fidelityminus_origin_mean = np.average(np.array(fidelityminus_origin_arr),axis=0)
    fidelityminus_complete_mean = np.average(np.array(fidelityminus_complete_arr),axis=0)
    finalfidelity_complete_mean = np.average(np.array(finalfidelity_complete_arr), axis=0)
    fvaluefidelity_complete_mean = np.average(np.array(fvaluefidelity_complete_arr), axis=0)
    del_fidelity_mean = np.average(np.array(del_fidelity_arr),axis=0)
    del_fidelity_origin_mean = np.average(np.array(del_fidelity_origin_arr),axis=0)
    del_fidelity_complete_mean = np.average(np.array(del_fidelity_complete_arr),axis=0)
    del_fidelityminus_mean = np.average(np.array(del_fidelityminus_arr),axis=0)
    del_fidelityminus_origin_mean = np.average(np.array(del_fidelityminus_origin_arr),axis=0)
    del_fidelityminus_complete_mean = np.average(np.array(del_fidelityminus_complete_arr),axis=0)
    del_finalfidelity_complete_mean = np.average(np.array(del_finalfidelity_complete_arr), axis=0)
    del_fvaluefidelity_complete_mean = np.average(np.array(del_fvaluefidelity_complete_arr), axis=0)
    sparsity_edges_mean = np.average(np.array(sparsity_edges_arr),axis=0)
    fidelity_nodes_mean = np.average(np.array(fidelity_nodes_arr),axis=0)
    fidelity_origin_nodes_mean = np.average(np.array(fidelity_origin_nodes_arr),axis=0)
    fidelity_complete_nodes_mean = np.average(np.array(fidelity_complete_nodes_arr),axis=0)
    fidelityminus_nodes_mean = np.average(np.array(fidelityminus_nodes_arr),axis=0)
    fidelityminus_origin_nodes_mean = np.average(np.array(fidelityminus_origin_nodes_arr),axis=0)
    fidelityminus_complete_nodes_mean = np.average(np.array(fidelityminus_complete_nodes_arr),axis=0)
    finalfidelity_complete_nodes_mean = np.average(np.array(finalfidelity_complete_nodes_arr), axis=0)
    sparsity_nodes_mean = np.average(np.array(sparsity_nodes_arr),axis=0)

    print("MO_auc_mean =", np.mean(auc_all))
    print("MO_ndcg_mean =", np.mean(ndcg_all))
    print("MO_simula_mean =", list(simula_mean))
    print("MO_simula_origin_mean =", list(simula_origin_mean))
    print("MO_simula_complete_mean =", list(simula_complete_mean))
    print("MO_fidelity_mean = ", list(fidelity_mean))
    print("MO_fidelity_origin_mean = ", list(fidelity_origin_mean))
    print("MO_fidelity_complete_mean = ", list(fidelity_complete_mean))
    print("MO_fidelityminus_mean = ", list(fidelityminus_mean))
    print("MO_fidelityminus_origin_mean = ", list(fidelityminus_origin_mean))
    print("MO_fidelityminus_complete_mean = ", list(fidelityminus_complete_mean))
    print("MO_finalfidelity_complete_mean = ", list(finalfidelity_complete_mean))
    print("MO_fvaluefidelity_complete_mean = ", list(fvaluefidelity_complete_mean))
    print("MO_del_fidelity_mean = ", list(del_fidelity_mean))
    print("MO_del_fidelity_origin_mean = ", list(del_fidelity_origin_mean))
    print("MO_del_fidelity_complete_mean = ", list(del_fidelity_complete_mean))
    print("MO_del_fidelityminus_mean = ", list(del_fidelityminus_mean))
    print("MO_del_fidelityminus_origin_mean = ", list(del_fidelityminus_origin_mean))
    print("MO_del_fidelityminus_complete_mean = ", list(del_fidelityminus_complete_mean))
    print("MO_del_finalfidelity_complete_mean = ", list(del_finalfidelity_complete_mean))
    print("MO_del_fvaluefidelity_complete_mean = ", list(del_fvaluefidelity_complete_mean))
    print("MO_sparsity_edges_mean =", list(sparsity_edges_mean))
    print("MO_fidelity_nodes_mean =", list(fidelity_nodes_mean))
    print("MO_fidelity_origin_nodes_mean =", list(fidelity_origin_nodes_mean))
    print("MO_fidelity_complete_nodes_mean =", list(fidelity_complete_nodes_mean))
    print("MO_fidelityminus_nodes_mean =", list(fidelityminus_nodes_mean))
    print("MO_fidelityminus_origin_nodes_mean =", list(fidelityminus_origin_nodes_mean))
    print("MO_fidelityminus_complete_nodes_mean =", list(fidelityminus_complete_nodes_mean))
    print("MO_finalfidelity_complete_nodes_mean =", list(finalfidelity_complete_nodes_mean))
    print("MO_sparsity_nodes_mean =", list(sparsity_nodes_mean))

    f_mean.write("MO_auc_mean = {}".format(np.mean(auc_all))+ "\n")
    f_mean.write("MO_ndcg_mean = {}".format(np.mean(ndcg_all))+ "\n")
    f_mean.write("MO_PN_mean = {}".format(np.mean(PN_all))+ "\n")
    f_mean.write("MO_PS_mean = {}".format(np.mean(PS_all))+ "\n")
    f_mean.write("MO_FNS_mean = {}".format(np.mean(FNS_all))+ "\n")
    f_mean.write("MO_size_mean = {}".format(np.mean(size_all))+ "\n")
    f_mean.write("MO_acc_mean = {}".format(np.mean(acc_all))+ "\n")
    f_mean.write("MO_pre_mean = {}".format(np.mean(pre_all))+ "\n")
    f_mean.write("MO_rec_mean = {}".format(np.mean(rec_all))+ "\n")
    f_mean.write("MO_f1_mean = {}".format(np.mean(f1_all))+ "\n")
    f_mean.write("MO_simula_mean = {}".format(list(simula_mean))+ "\n")
    f_mean.write("MO_simula_origin_mean = {}".format(list(simula_origin_mean))+ "\n")
    f_mean.write("MO_simula_complete_mean = {}".format(list(simula_complete_mean))+ "\n")
    f_mean.write("MO_fidelity_mean = {}".format(list(fidelity_mean))+ "\n")
    f_mean.write("MO_fidelity_origin_mean = {}".format(list(fidelity_origin_mean))+ "\n")
    f_mean.write("MO_fidelity_complete_mean = {}".format(list(fidelity_complete_mean))+ "\n")
    f_mean.write("MO_fidelityminus_mean = {}".format(list(fidelityminus_mean))+"\n")
    f_mean.write("MO_fidelityminus_origin_mean = {}".format(list(fidelityminus_origin_mean))+"\n")
    f_mean.write("MO_fidelityminus_complete_mean = {}".format(list(fidelityminus_complete_mean))+"\n")
    f_mean.write("MO_finalfidelity_complete_mean = {}".format(list(finalfidelity_complete_mean))+"\n")
    f_mean.write("MO_fvaluefidelity_complete_mean = {}".format(list(fvaluefidelity_complete_mean)) + "\n")
    f_mean.write("MO_del_fidelity_mean = {}".format(list(del_fidelity_mean))+ "\n")
    f_mean.write("MO_del_fidelity_origin_mean = {}".format(list(del_fidelity_origin_mean))+ "\n")
    f_mean.write("MO_del_fidelity_complete_mean = {}".format(list(del_fidelity_complete_mean))+ "\n")
    f_mean.write("MO_del_fidelityminus_mean = {}".format(list(del_fidelityminus_mean))+"\n")
    f_mean.write("MO_del_fidelityminus_origin_mean = {}".format(list(del_fidelityminus_origin_mean))+"\n")
    f_mean.write("MO_del_fidelityminus_complete_mean = {}".format(list(del_fidelityminus_complete_mean))+"\n")
    f_mean.write("MO_del_finalfidelity_complete_mean = {}".format(list(del_finalfidelity_complete_mean))+"\n")
    f_mean.write("MO_del_fvaluefidelity_complete_mean = {}".format(list(del_fvaluefidelity_complete_mean)) + "\n")
    f_mean.write("MO_sparsity_edges_mean = {}".format(list(sparsity_edges_mean))+ "\n")
    f_mean.write("MO_fidelity_nodes_mean = {}".format(list(fidelity_nodes_mean))+ "\n")
    f_mean.write("MO_fidelity_origin_nodes_mean = {}".format(list(fidelity_origin_nodes_mean))+ "\n")
    f_mean.write("MO_fidelity_complete_nodes_mean = {}".format(list(fidelity_complete_nodes_mean))+ "\n")
    f_mean.write("MO_fidelityminus_nodes_mean = {}".format(list(fidelityminus_nodes_mean))+"\n")
    f_mean.write("MO_fidelityminus_origin_nodes_mean = {}".format(list(fidelityminus_origin_nodes_mean))+"\n")
    f_mean.write("MO_fidelityminus_complete_nodes_mean = {}".format(list(fidelityminus_complete_nodes_mean))+"\n")
    f_mean.write("MO_finalfidelity_complete_nodes_mean = {}".format(list(finalfidelity_complete_nodes_mean))+"\n")
    f_mean.write("MO_sparsity_nodes_mean = {}".format(list(sparsity_nodes_mean))+ "\n")
    f_mean.close()



def test_main(log_filename, save_map, dataset_filename, testmodel_filename, plot_flag):
    f_mean = open(log_filename, "w")
    auc_all = []
    ndcg_all = []
    PN_all = []
    PS_all = []
    FNS_all = []
    size_all = []
    acc_all = []
    pre_all = []
    rec_all = []
    f1_all = []
    simula_arr = []
    simula_origin_arr = []
    simula_complete_arr = []
    fidelity_arr = []
    fidelity_origin_arr = []
    fidelity_complete_arr = []
    fidelityminus_arr = []
    fidelityminus_origin_arr = []
    fidelityminus_complete_arr = []
    finalfidelity_complete_arr = []
    fvaluefidelity_complete_arr = []
    del_fidelity_arr = []
    del_fidelity_origin_arr = []
    del_fidelity_complete_arr = []
    del_fidelityminus_arr = []
    del_fidelityminus_origin_arr = []
    del_fidelityminus_complete_arr = []
    del_finalfidelity_complete_arr = []
    del_fvaluefidelity_complete_arr = []
    sparsity_edges_arr = []
    fidelity_nodes_arr = []
    fidelity_origin_nodes_arr = []
    fidelity_complete_nodes_arr = []
    fidelityminus_nodes_arr = []
    fidelityminus_origin_nodes_arr = []
    fidelityminus_complete_nodes_arr = []
    finalfidelity_complete_nodes_arr = []
    sparsity_nodes_arr = []
    for iter in range(1):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        #args.device = "cpu"
        print("Starting iteration: {}".format(iter))
        if not os.path.exists(save_map+str(iter)):
            os.makedirs(save_map+str(iter))
        #load data
        if args.dataset == "Computers":
            dataset = Amazon(root=args.dataset_root, name=args.dataset)
        #data = dataset.data
        num_class = dataset.num_classes
        label = dataset.data.y
        adj =  torch.sparse_coo_tensor(indices=dataset.data.edge_index, values= torch.ones(dataset.data.edge_index.shape[1])).to_dense()
        #adj = csr_matrix(adj.numpy())
        edge_label_matrix = None
        num_nodes = dataset.data.x.shape[0]
        node_index = list(range(num_nodes))
        random.seed(2023)
        random.shuffle(node_index)
        trainnodes = node_index[ : math.floor(num_nodes*0.8)]
        valnodes = node_index[math.floor(num_nodes*0.8) : math.floor(num_nodes*0.9)]
        testnodes = node_index[math.floor(num_nodes*0.9) : ]

        #load trained GCN (survey)
        GNNmodel_ckpt_path = osp.join('GNN_checkpoint', args.dataset+"_"+str(iter), 'gcn_best.pth') 
        model = load_gnnNets_NC(GNNmodel_ckpt_path, input_dim=dataset.data.x.shape[1], output_dim=num_class, device = args.device)
        model.eval()

        hops = len(args.hiddens.split('-'))
        #extractor = Extractor(adj, edge_index, features, edge_label_matrix, embeds, label, hops)
        extractor = Extractor(adj, edge_label_matrix, hops)
        
        explainer = ExplainerMO(model=model, args=args)
        explainer.to(args.device)

        f = open(save_map + str(iter) + "/" + "LOG_" + testmodel_filename + "_test.txt", "w")
        savedDict = torch.load(save_map + str(iter) + "/" + testmodel_filename + ".pt") 
        explainer.load_state_dict(savedDict)

        f.write("test result.")
        # metrics
        tik = time.time()
        reals = []
        preds = []
        ndcgs = []
        #exp_dict={}
        #pred_label_dict={}
        #feature_dict = {}
        #adj_dict = {}
        #e_labels_dict = {}
        PN, PS, ave_size = 0, 0, 0
        acc_1, pre, rec, f1 =0, 0, 0, 0
        #allnode_related_preds_dict = dict()
        #allnode_mask_dict = dict()
        metric = MaskoutMetric(model, args)
        plotutils = PlotUtils(dataset_name=args.dataset)
        #f_hidden = open(save_map + str(iter) + "/" +"hidden_emb_test.txt", "w")
        
        if not os.path.exists(save_map+str(iter)+"/explanation"):
            os.makedirs(save_map+str(iter)+"/explanation")
        
        print("save explanation json")
        for j in range(len(testnodes)):
            node = testnodes[j]
            if os.path.exists(save_map+str(iter)+"/explanation/exp_"+str(node)+".json"):
                continue
            sub_node, sub_adj,sub_edge_label = extractor.subgraph(node)
            if len(sub_node) == 0:
                continue
            sub_edge_index = torch.nonzero(sub_adj).T
            sub_feature = dataset.data.x[sub_node]
            data = Data(x=sub_feature, edge_index=sub_edge_index).to(args.device)
            with torch.no_grad():
                _, sub_output, sub_embed, _ = model(data)
            
            #use fullgraph prediction
            #origin_pred = outputs[node].to(args.device)
            #use subgraph prediction
            nodeid = 0
            origin_pred = sub_output[nodeid]
            origin_label = origin_pred.argmax(-1)
            explainer.eval()
            with torch.no_grad():
                masked_pred, _, _, _ = explainer((nodeid, data, sub_adj, sub_embed, 1.0))

            sub_label = dataset.data.y[sub_node]
            masked_adj = explainer.masked_adj
            explain_result_json = {"masked_adj":masked_adj, "masked_pred":masked_pred, "new_nodeid":nodeid, "sub_feature":sub_feature, "sub_edge_index":sub_edge_index,
                                "pred_label":origin_label, "sub_adj":sub_adj, "sub_embed":sub_embed,"origin_pred":origin_pred, "sub_label": sub_label, "sub_node":sub_node}
            torch.save(explain_result_json,  save_map+str(iter)+"/explanation/exp_"+str(node)+".json")
        
            
        print("metric AUC,NDCG")
        for j in range(len(testnodes)):
            node = testnodes[j]
            if not os.path.exists(save_map+str(iter)+"/explanation/exp_"+str(node)+".json"):
                continue
            explain_result_json = torch.load(save_map+str(iter)+"/explanation/exp_"+str(node)+".json")
            sub_adj = explain_result_json["sub_adj"]
            masked_adj = explain_result_json["masked_adj"]
            masked_pred = explain_result_json["masked_pred"]
            nodeid = explain_result_json["new_nodeid"]
            sub_feature = explain_result_json["sub_feature"]
            sub_edge_index = explain_result_json["sub_edge_index"]
            origin_pred = explain_result_json["origin_pred"]
            origin_label = explain_result_json["pred_label"]
            data = Data(x=sub_feature, edge_index=sub_edge_index).to(args.device)
            auc_onenode, ndcg_onenode, real, pred  = explain_test(masked_adj, masked_pred, sub_edge_index, None, origin_pred)
            ndcgs.append(ndcg_onenode)
            reals.extend(real)
            preds.extend(pred)

            ps, pn, size= compute_pnps_NC_onenode(masked_adj, origin_label, model, sub_feature, sub_adj, nodeid, args.fix_exp, args.mask_thresh)
            PN += pn
            PS += ps
            ave_size += size
            if args.dataset == "BA_shapes" or args.dataset == "BA_community":
                acc_onenode_1, pre_onenode, rec_onenode, f1_onenode = compute_precision_recall_NC_onenode(masked_adj, sub_edge_label, sub_edge_index, args.fix_exp, args.mask_thresh)
            else:
                acc_onenode_1, pre_onenode, rec_onenode, f1_onenode = 0, 0, 0, 0
            acc_1 += acc_onenode_1
            pre += pre_onenode
            rec += rec_onenode
            f1 += f1_onenode

            if plot_flag:
                #plot(node, label, iter)
                # visualization
                edge_mask = masked_adj[sub_adj.row, sub_adj.col].detach()
                edges_idx_desc = edge_mask.reshape(-1).sort(descending=True).indices
                important_nodelist = []
                important_edgelist = []
                for idx in edges_idx_desc:
                    if len(important_edgelist)<12:
                        if (sub_edge_index[0][idx].item(), sub_edge_index[1][idx].item()) not in important_edgelist:
                            important_nodelist.append(sub_edge_index[0][idx].item())
                            important_nodelist.append(sub_edge_index[1][idx].item())
                            important_edgelist.append((sub_edge_index[0][idx].item(), sub_edge_index[1][idx].item()))
                            important_edgelist.append((sub_edge_index[1][idx].item(), sub_edge_index[0][idx].item()))
                important_nodelist = list(set(important_nodelist))
                data = Data(x=sub_feature, edge_index=sub_edge_index, y =sub_label.argmax(-1) )
                ori_graph = to_networkx(data, to_undirected=True)
                plotutils.plot_new(ori_graph, nodelist=important_nodelist, edgelist=important_edgelist, y=sub_label.argmax(-1), node_idx=nodeid,
                            figname=os.path.join(save_map + str(iter), f"node_{node}_new.png"))

            tok = time.time()
            f.write("node,{}".format(node) + ",auc_onenode,{}".format(auc_onenode) + ",ndcg_onenode,{}".format(ndcg_onenode)  + ",time,{}".format(tok - tik) + "\n")
        
        #f_hidden.close()

        if args.dataset == "Computers":
            auc=0
        else:
            if len(np.unique(reals))==1 or len(np.unique(preds))==1:
                auc = -1
            else:
                auc = roc_auc_score(reals, preds)
        ndcg = np.mean(ndcgs)
        auc_all.append(auc)
        ndcg_all.append(ndcg)
        tok = time.time()
        f.write("iter,{}".format(iter) + ",auc,{}".format(auc) + ",ndcg,{}".format(ndcg) + ",time,{}".format(tok - tik) + "\n")
        
        PN = PN/len(testnodes)
        PS = PS/len(testnodes)
        ave_size = ave_size/len(testnodes)

        if PN + PS==0:
            FNS=0
        else:
            FNS = 2 * PN * PS / (PN + PS)
        acc_1 = acc_1/len(testnodes)
        pre = pre/len(testnodes)
        rec = rec/len(testnodes)
        f1 = f1/len(testnodes)

        PN_all.append(PN)
        PS_all.append(PS)
        FNS_all.append(FNS)
        size_all.append(ave_size)
        acc_all.append(acc_1)
        pre_all.append(pre)
        rec_all.append(rec)
        f1_all.append(f1)

        one_simula_arr = []
        one_simula_origin_arr = []
        one_simula_complete_arr = []
        one_fidelity_arr = []
        one_fidelity_origin_arr = []
        one_fidelity_complete_arr = []
        one_fidelityminus_arr = []
        one_fidelityminus_origin_arr = []
        one_fidelityminus_complete_arr = []
        one_finalfidelity_complete_arr = []
        one_fvaluefidelity_complete_arr = []
        one_del_fidelity_arr = []
        one_del_fidelity_origin_arr = []
        one_del_fidelity_complete_arr = []
        one_del_fidelityminus_arr = []
        one_del_fidelityminus_origin_arr = []
        one_del_fidelityminus_complete_arr = []
        one_del_finalfidelity_complete_arr = []
        one_del_fvaluefidelity_complete_arr = []
        one_sparsity_edges_arr = []
        one_fidelity_nodes_arr = []
        one_fidelity_origin_nodes_arr = []
        one_fidelity_complete_nodes_arr = []
        one_fidelityminus_nodes_arr = []
        one_fidelityminus_origin_nodes_arr = []
        one_fidelityminus_complete_nodes_arr = []
        one_finalfidelity_complete_nodes_arr = []
        one_sparsity_nodes_arr = []
        for top_k in args.topk_arr:
            print("top_k",top_k)
            fidelity_complete = []
            fidelity = []
            fidelity_origin = []
            fidelityminus_complete = []
            fidelityminus = []
            fidelityminus_origin = []
            del_fidelity_complete = []
            del_fidelity = []
            del_fidelity_origin = []
            del_fidelityminus_complete = []
            del_fidelityminus = []
            del_fidelityminus_origin = []
            simula_complete = []
            simula = []
            simula_origin = []
            sparsity_edges = []
            fidelity_nodes = []
            fidelity_origin_nodes = []
            fidelity_complete_nodes = []
            fidelityminus_nodes = []
            fidelityminus_origin_nodes = []
            fidelityminus_complete_nodes = []
            sparsity_nodes = []

            #x_collector = XCollector()
            #for node in testnodes:
            for j in range(len(testnodes)):
                node = testnodes[j]
                print("index", j, "node", node)
                if not os.path.exists(save_map+str(iter)+"/explanation/exp_"+str(node)+".json"):
                    continue
                explain_result_json = torch.load(save_map+str(iter)+"/explanation/exp_"+str(node)+".json")
                masked_adj = explain_result_json["masked_adj"]
                masked_pred = explain_result_json["masked_pred"]
                nodeid = explain_result_json["new_nodeid"]
                sub_feature = explain_result_json["sub_feature"]
                sub_edge_index = explain_result_json["sub_edge_index"]
                origin_pred = explain_result_json["origin_pred"]
                sub_label = explain_result_json["sub_label"]
                sub_node = explain_result_json["sub_node"]
                data = Data(x=sub_feature, edge_index=sub_edge_index).to(args.device)

                pred_mask, related_preds_dict = metric.metric_pg_del_edges(nodeid, masked_adj, masked_pred, data, sub_label, origin_pred, sub_node, topk_arr = [top_k])
                related_preds = related_preds_dict[top_k]

                #mask the selected edges
                fidelity_complete.append(calculate_fidelity_complete(related_preds[0]["origin"], related_preds[0]["maskimp"]))
                fidelity.append(calculate_fidelity(related_preds[0]["origin_l"], related_preds[0]["maskimp_l"]))
                fidelity_origin.append(calculate_fidelity(related_preds[0]["origin_ol"], related_preds[0]["maskimp_ol"]))

                #mask other edges except the selected edges
                fidelityminus_complete.append(calculate_fidelity_complete(related_preds[0]["origin"], related_preds[0]["masknotimp"]))
                fidelityminus.append(calculate_fidelity(related_preds[0]["origin_l"], related_preds[0]["masknotimp_l"]))
                fidelityminus_origin.append(calculate_fidelity(related_preds[0]["origin_ol"], related_preds[0]["masknotimp_ol"]))

                #delete the important edges
                del_fidelity_complete.append(calculate_fidelity_complete(related_preds[0]["origin"], related_preds[0]["delimp"]))
                del_fidelity.append(calculate_fidelity(related_preds[0]["origin_l"], related_preds[0]["delimp_l"]))
                del_fidelity_origin.append(calculate_fidelity(related_preds[0]["origin_ol"], related_preds[0]["delimp_ol"]))

                #retain the important edges 
                del_fidelityminus_complete.append(calculate_fidelity_complete(related_preds[0]["origin"], related_preds[0]["retainimp"]))
                del_fidelityminus.append(calculate_fidelity(related_preds[0]["origin_l"], related_preds[0]["retainimp_l"]))
                del_fidelityminus_origin.append(calculate_fidelity(related_preds[0]["origin_ol"], related_preds[0]["retainimp_ol"]))

                simula_complete.append( calculate_fidelity_complete(related_preds[0]["origin"], related_preds[0]["masked"]))
                simula.append(calculate_fidelity(related_preds[0]["origin_l"], related_preds[0]["masked_l"]))
                simula_origin.append(calculate_fidelity(related_preds[0]["origin_ol"], related_preds[0]["masked_ol"]))
                sparsity_edges.append(related_preds[0]["sparsity_edges"])

                #mask the selected nodes
                fidelity_complete_nodes.append(calculate_fidelity_complete(related_preds[0]["origin"], related_preds[0]["maskimp_nodes"]))
                fidelity_nodes.append(calculate_fidelity(related_preds[0]["origin_l"], related_preds[0]["maskimp_nodes_l"]))
                fidelity_origin_nodes.append(calculate_fidelity(related_preds[0]["origin_ol"], related_preds[0]["maskimp_nodes_ol"]))

                #mask other nodes except the selected nodes
                fidelityminus_complete_nodes.append(calculate_fidelity_complete(related_preds[0]["origin"], related_preds[0]["retainimp_nodes"]))
                fidelityminus_nodes.append(calculate_fidelity(related_preds[0]["origin_l"], related_preds[0]["retainimp_nodes_l"]))
                fidelityminus_origin_nodes.append(calculate_fidelity(related_preds[0]["origin_ol"], related_preds[0]["retainimp_nodes_ol"]))
                sparsity_nodes.append(related_preds[0]["sparsity_nodes"])

                #allnode_related_preds_dict[node] = related_preds_dict
                #allnode_mask_dict[node] = pred_mask
                #related_preds = allnode_related_preds_dict[node][top_k]
                #mask = allnode_mask_dict[node]
                
                #x_collector.collect_data(pred_mask, related_preds)
                f.write("node,{}\n".format(node))
                f.write("mask,{}\n".format(pred_mask))
                f.write("related_preds,{}\n".format(related_preds))

            one_simula_arr.append(round(np.mean(simula), 4))
            one_simula_origin_arr.append(round(np.mean(simula_origin), 4))
            one_simula_complete_arr.append(round(np.mean(simula_complete), 4))
            one_fidelity_arr.append(round(np.mean(fidelity), 4))
            one_fidelity_origin_arr.append(round(np.mean(fidelity_origin), 4))
            one_fidelity_complete_arr.append(round(np.mean(fidelity_complete), 4))
            one_fidelityminus_arr.append(round(np.mean(fidelityminus), 4))
            one_fidelityminus_origin_arr.append(round(np.mean(fidelityminus_origin), 4))
            one_fidelityminus_complete_arr.append(round(np.mean(fidelityminus_complete), 4))
            one_finalfidelity_complete_arr.append(round(np.mean(fidelity_complete) - np.mean(fidelityminus_complete), 4))
            F_fidelity = 2/(1/np.mean(fidelity_complete) +1/(1/np.mean(fidelityminus_complete)))
            one_fvaluefidelity_complete_arr.append(round(F_fidelity, 4))
            one_del_fidelity_arr.append(round(np.mean(del_fidelity), 4))
            one_del_fidelity_origin_arr.append(round(np.mean(del_fidelity_origin), 4))
            one_del_fidelity_complete_arr.append(round(np.mean(del_fidelity_complete), 4))
            one_del_fidelityminus_arr.append(round(np.mean(del_fidelityminus), 4))
            one_del_fidelityminus_origin_arr.append(round(np.mean(del_fidelityminus_origin), 4))
            one_del_fidelityminus_complete_arr.append(round(np.mean(del_fidelityminus_complete), 4))
            one_del_finalfidelity_complete_arr.append(round(np.mean(del_fidelity_complete) - np.mean(del_fidelityminus_complete), 4))
            del_F_fidelity = 2/(1/np.mean(del_fidelity_complete) +1/(1/np.mean(del_fidelityminus_complete)))
            one_del_fvaluefidelity_complete_arr.append(round(del_F_fidelity, 4))
            one_sparsity_edges_arr.append(round(np.mean(sparsity_edges), 4))
            one_fidelity_nodes_arr.append(round(np.mean(fidelity_nodes), 4))
            one_fidelity_origin_nodes_arr.append(round(np.mean(fidelity_origin_nodes), 4))
            one_fidelity_complete_nodes_arr.append(round(np.mean(fidelity_complete_nodes), 4))
            one_fidelityminus_nodes_arr.append(round(np.mean(fidelityminus_nodes), 4))
            one_fidelityminus_origin_nodes_arr.append(round(np.mean(fidelityminus_origin_nodes), 4))
            one_fidelityminus_complete_nodes_arr.append(round(np.mean(fidelityminus_complete_nodes), 4))
            one_finalfidelity_complete_nodes_arr.append(round(np.mean(fidelity_complete_nodes) - np.mean(fidelityminus_complete_nodes), 4))
            one_sparsity_nodes_arr.append(round(np.mean(sparsity_nodes), 4))

        print("one_auc=", auc)
        print("one_ndcg=", ndcg)
        print("one_simula_arr =", one_simula_arr)
        print("one_simula_origin_arr =", one_simula_origin_arr)
        print("one_simula_complete_arr =", one_simula_complete_arr)
        print("one_fidelity_arr =", one_fidelity_arr)
        print("one_fidelity_origin_arr =", one_fidelity_origin_arr)
        print("one_fidelity_complete_arr =", one_fidelity_complete_arr)
        print("one_fidelityminus_arr=", one_fidelityminus_arr)
        print("one_fidelityminus_origin_arr=", one_fidelityminus_origin_arr)
        print("one_fidelityminus_complete_arr=", one_fidelityminus_complete_arr)
        print("one_finalfidelity_complete_arr=", one_finalfidelity_complete_arr)
        print("one_fvaluefidelity_complete_arr=", one_fvaluefidelity_complete_arr)
        print("one_del_fidelity_arr =", one_del_fidelity_arr)
        print("one_del_fidelity_origin_arr =", one_del_fidelity_origin_arr)
        print("one_del_fidelity_complete_arr =", one_del_fidelity_complete_arr)
        print("one_del_fidelityminus_arr=", one_del_fidelityminus_arr)
        print("one_del_fidelityminus_origin_arr=", one_del_fidelityminus_origin_arr)
        print("one_del_fidelityminus_complete_arr=", one_del_fidelityminus_complete_arr)
        print("one_del_finalfidelity_complete_arr=", one_del_finalfidelity_complete_arr)
        print("one_del_fvaluefidelity_complete_arr=", one_del_fvaluefidelity_complete_arr)
        print("one_sparsity_edges_arr =", one_sparsity_edges_arr)
        print("one_fidelity_nodes_arr =", one_fidelity_nodes_arr)
        print("one_fidelity_origin_nodes_arr =", one_fidelity_origin_nodes_arr)
        print("one_fidelity_complete_nodes_arr =", one_fidelity_complete_nodes_arr)
        print("one_fidelityminus_nodes_arr=", one_fidelityminus_nodes_arr)
        print("one_fidelityminus_origin_nodes_arr=", one_fidelityminus_origin_nodes_arr)
        print("one_fidelityminus_complete_nodes_arr=", one_fidelityminus_complete_nodes_arr)
        print("one_finalfidelity_complete_nodes_arr=", one_finalfidelity_complete_nodes_arr)
        print("one_sparsity_nodes_arr =", one_sparsity_nodes_arr)

        tok = time.time()
        f.write("one_auc={}".format(auc) + "\n")
        f.write("one_ndcg={}".format(ndcg) + "\n")
        f.write("one_simula={}".format(one_simula_arr) + "\n")
        f.write("one_simula_orign={}".format(one_simula_origin_arr) + "\n")
        f.write("one_simula_complete={}".format(one_simula_complete_arr) + "\n")
        f.write("one_fidelity={}".format(one_fidelity_arr) + "\n")
        f.write("one_fidelity_orign={}".format(one_fidelity_origin_arr) + "\n")
        f.write("one_fidelity_complete={}".format(one_fidelity_complete_arr) + "\n")
        f.write("one_fidelityminus={}".format(one_fidelityminus_arr)+"\n")
        f.write("one_fidelityminus_origin={}".format(one_fidelityminus_origin_arr)+"\n")
        f.write("one_fidelityminus_complete={}".format(one_fidelityminus_complete_arr)+"\n")
        f.write("one_finalfidelity_complete={}".format(one_finalfidelity_complete_arr)+"\n")
        f.write("one_fvaluefidelity_complete={}".format(one_fvaluefidelity_complete_arr)+"\n")
        f.write("one_del_fidelity={}".format(one_del_fidelity_arr) + "\n")
        f.write("one_del_fidelity_orign={}".format(one_del_fidelity_origin_arr) + "\n")
        f.write("one_del_fidelity_complete={}".format(one_del_fidelity_complete_arr) + "\n")
        f.write("one_del_fidelityminus={}".format(one_del_fidelityminus_arr)+"\n")
        f.write("one_del_fidelityminus_origin={}".format(one_del_fidelityminus_origin_arr)+"\n")
        f.write("one_del_fidelityminus_complete={}".format(one_del_fidelityminus_complete_arr)+"\n")
        f.write("one_del_finalfidelity_complete={}".format(one_del_finalfidelity_complete_arr)+"\n")
        f.write("one_del_fvaluefidelity_complete={}".format(one_del_fvaluefidelity_complete_arr)+"\n")
        f.write("one_sparsity_edges={}".format(one_sparsity_edges_arr) + "\n")
        f.write("one_fidelity_nodes={}".format(one_fidelity_nodes_arr) + "\n")
        f.write("one_fidelity_origin_nodes={}".format(one_fidelity_origin_nodes_arr) + "\n")
        f.write("one_fidelity_complete_nodes={}".format(one_fidelity_complete_nodes_arr) + "\n")
        f.write("one_fidelityminus_nodes={}".format(one_fidelityminus_nodes_arr)+"\n")
        f.write("one_fidelityminus_origin_nodes={}".format(one_fidelityminus_origin_nodes_arr)+"\n")
        f.write("one_fidelityminus_complete_nodes={}".format(one_fidelityminus_complete_nodes_arr)+"\n")
        f.write("one_finalfidelity_complete_nodes={}".format(one_finalfidelity_complete_nodes_arr)+"\n")
        f.write("one_sparsity_nodes={}".format(one_sparsity_nodes_arr) + "\n")
        f.write("test time,{}".format(tok-tik))
        f.close()

        simula_arr.append(one_simula_arr)
        simula_origin_arr.append(one_simula_origin_arr)
        simula_complete_arr.append(one_simula_complete_arr)
        fidelity_arr.append(one_fidelity_arr)
        fidelity_origin_arr.append(one_fidelity_origin_arr)
        fidelity_complete_arr.append(one_fidelity_complete_arr)
        fidelityminus_arr.append(one_fidelityminus_arr)
        fidelityminus_origin_arr.append(one_fidelityminus_origin_arr)
        fidelityminus_complete_arr.append(one_fidelityminus_complete_arr)
        finalfidelity_complete_arr.append(one_finalfidelity_complete_arr)
        fvaluefidelity_complete_arr.append(one_fvaluefidelity_complete_arr)
        del_fidelity_arr.append(one_del_fidelity_arr)
        del_fidelity_origin_arr.append(one_del_fidelity_origin_arr)
        del_fidelity_complete_arr.append(one_del_fidelity_complete_arr)
        del_fidelityminus_arr.append(one_del_fidelityminus_arr)
        del_fidelityminus_origin_arr.append(one_del_fidelityminus_origin_arr)
        del_fidelityminus_complete_arr.append(one_del_fidelityminus_complete_arr)
        del_finalfidelity_complete_arr.append(one_del_finalfidelity_complete_arr)
        del_fvaluefidelity_complete_arr.append(one_del_fvaluefidelity_complete_arr)
        sparsity_edges_arr.append(one_sparsity_edges_arr)
        fidelity_nodes_arr.append(one_fidelity_nodes_arr)
        fidelity_origin_nodes_arr.append(one_fidelity_origin_nodes_arr)
        fidelity_complete_nodes_arr.append(one_fidelity_complete_nodes_arr)
        fidelityminus_nodes_arr.append(one_fidelityminus_nodes_arr)
        fidelityminus_origin_nodes_arr.append(one_fidelityminus_origin_nodes_arr)
        fidelityminus_complete_nodes_arr.append(one_fidelityminus_complete_nodes_arr)
        finalfidelity_complete_nodes_arr.append(one_finalfidelity_complete_nodes_arr)
        sparsity_nodes_arr.append(one_sparsity_nodes_arr)

    print("args.dataset", args.dataset)
    print("MO_auc_all = ", auc_all)
    print("MO_ndcg_all = ", ndcg_all)
    print("MO_simula_arr =", simula_arr)
    print("MO_simula_origin_arr =", simula_origin_arr)
    print("MO_simula_complete_arr =", simula_complete_arr)
    print("MO_fidelity_arr =", fidelity_arr)
    print("MO_fidelity_origin_arr =", fidelity_origin_arr)
    print("MO_fidelity_complete_arr =", fidelity_complete_arr)
    print("MO_fidelityminus_arr=", fidelityminus_arr)
    print("MO_fidelityminus_origin_arr=", fidelityminus_origin_arr)
    print("MO_fidelityminus_complete_arr=", fidelityminus_complete_arr)
    print("MO_finalfidelity_complete_arr", finalfidelity_complete_arr)
    print("MO_fvaluefidelity_complete_arr", fvaluefidelity_complete_arr)
    print("MO_del_fidelity_arr =", del_fidelity_arr)
    print("MO_del_fidelity_origin_arr =", del_fidelity_origin_arr)
    print("MO_del_fidelity_complete_arr =", del_fidelity_complete_arr)
    print("MO_del_fidelityminus_arr=", del_fidelityminus_arr)
    print("MO_del_fidelityminus_origin_arr=", del_fidelityminus_origin_arr)
    print("MO_del_fidelityminus_complete_arr=", del_fidelityminus_complete_arr)
    print("MO_del_finalfidelity_complete_arr", del_finalfidelity_complete_arr)
    print("MO_del_fvaluefidelity_complete_arr", del_fvaluefidelity_complete_arr)
    print("MO_sparsity_edges_arr =", sparsity_edges_arr)
    print("MO_fidelity_nodes_arr =", fidelity_nodes_arr)
    print("MO_fidelity_origin_nodes_arr =", fidelity_origin_nodes_arr)
    print("MO_fidelity_complete_nodes_arr =", fidelity_complete_nodes_arr)
    print("MO_fidelityminus_nodes_arr=", fidelityminus_nodes_arr)
    print("MO_fidelityminus_origin_nodes_arr=", fidelityminus_origin_nodes_arr)
    print("MO_fidelityminus_complete_nodes_arr=", fidelityminus_complete_nodes_arr)
    print("MO_finalfidelity_complete_nodes_arr", finalfidelity_complete_nodes_arr)
    print("MO_sparsity_nodes_arr =", sparsity_nodes_arr)

    f_mean.write("MO_auc_all={}".format(auc_all) + "\n")
    f_mean.write("MO_ndcg_all={}".format(ndcg_all) + "\n")
    f_mean.write("MO_PN_all={}".format(PN_all) + "\n")
    f_mean.write("MO_PS_all={}".format(PS_all) + "\n")
    f_mean.write("MO_FNS_all={}".format(FNS_all) + "\n")
    f_mean.write("MO_size_all={}".format(size_all) + "\n")
    f_mean.write("MO_acc_all={}".format(acc_all) + "\n")
    f_mean.write("MO_pre_all={}".format(pre_all) + "\n")
    f_mean.write("MO_rec_all={}".format(rec_all) + "\n")
    f_mean.write("MO_f1_all={}".format(f1_all) + "\n")
    f_mean.write("MO_simula_arr={}".format(simula_arr) + "\n")
    f_mean.write("MO_simula_origin_arr={}".format(simula_origin_arr) + "\n")
    f_mean.write("MO_simula_complete_arr={}".format(simula_complete_arr) + "\n")
    f_mean.write("MO_fidelity_arr={}".format(fidelity_arr) + "\n")
    f_mean.write("MO_fidelity_origin_arr={}".format(fidelity_origin_arr) + "\n")
    f_mean.write("MO_fidelity_complete_arr={}".format(fidelity_complete_arr) + "\n")
    f_mean.write("MO_fidelityminus_arr = {}".format(fidelityminus_arr)+"\n")
    f_mean.write("MO_fidelityminus_origin_arr = {}".format(fidelityminus_origin_arr)+"\n")
    f_mean.write("MO_fidelityminus_complete_arr = {}".format(fidelityminus_complete_arr)+"\n")
    f_mean.write("MO_finalfidelity_complete_arr = {}".format(finalfidelity_complete_arr)+"\n")
    f_mean.write("MO_fvaluefidelity_complete_arr = {}".format(fvaluefidelity_complete_arr)+"\n")
    f_mean.write("MO_del_fidelity_arr={}".format(del_fidelity_arr) + "\n")
    f_mean.write("MO_del_fidelity_origin_arr={}".format(del_fidelity_origin_arr) + "\n")
    f_mean.write("MO_del_fidelity_complete_arr={}".format(del_fidelity_complete_arr) + "\n")
    f_mean.write("MO_del_fidelityminus_arr = {}".format(del_fidelityminus_arr)+"\n")
    f_mean.write("MO_del_fidelityminus_origin_arr = {}".format(del_fidelityminus_origin_arr)+"\n")
    f_mean.write("MO_del_fidelityminus_complete_arr = {}".format(del_fidelityminus_complete_arr)+"\n")
    f_mean.write("MO_del_finalfidelity_complete_arr = {}".format(del_finalfidelity_complete_arr)+"\n")
    f_mean.write("MO_del_fvaluefidelity_complete_arr = {}".format(del_fvaluefidelity_complete_arr)+"\n")
    f_mean.write("MO_sparsity_edges_arr={}".format(sparsity_edges_arr) + "\n")
    f_mean.write("MO_fidelity_nodes_arr={}".format(fidelity_nodes_arr) + "\n")
    f_mean.write("MO_fidelity_origin_nodes_arr={}".format(fidelity_origin_nodes_arr) + "\n")
    f_mean.write("MO_fidelity_complete_nodes_arr={}".format(fidelity_complete_nodes_arr) + "\n")
    f_mean.write("MO_fidelityminus_nodes_arr = {}".format(fidelityminus_nodes_arr)+"\n")
    f_mean.write("MO_fidelityminus_origin_nodes_arr = {}".format(fidelityminus_origin_nodes_arr)+"\n")
    f_mean.write("MO_fidelityminus_complete_nodes_arr = {}".format(fidelityminus_complete_nodes_arr)+"\n")
    f_mean.write("MO_finalfidelity_complete_nodes_arr = {}".format(finalfidelity_complete_nodes_arr)+"\n")
    f_mean.write("MO_sparsity_nodes_arr={}".format(sparsity_nodes_arr) + "\n")

    simula_mean = np.average(np.array(simula_arr), axis=0)
    simula_origin_mean = np.average(np.array(simula_origin_arr), axis=0)
    simula_complete_mean = np.average(np.array(simula_complete_arr),axis=0)
    fidelity_mean = np.average(np.array(fidelity_arr),axis=0)
    fidelity_origin_mean = np.average(np.array(fidelity_origin_arr),axis=0)
    fidelity_complete_mean = np.average(np.array(fidelity_complete_arr),axis=0)
    fidelityminus_mean = np.average(np.array(fidelityminus_arr),axis=0)
    fidelityminus_origin_mean = np.average(np.array(fidelityminus_origin_arr),axis=0)
    fidelityminus_complete_mean = np.average(np.array(fidelityminus_complete_arr),axis=0)
    finalfidelity_complete_mean = np.average(np.array(finalfidelity_complete_arr), axis=0)
    fvaluefidelity_complete_mean = np.average(np.array(fvaluefidelity_complete_arr), axis=0)
    del_fidelity_mean = np.average(np.array(del_fidelity_arr),axis=0)
    del_fidelity_origin_mean = np.average(np.array(del_fidelity_origin_arr),axis=0)
    del_fidelity_complete_mean = np.average(np.array(del_fidelity_complete_arr),axis=0)
    del_fidelityminus_mean = np.average(np.array(del_fidelityminus_arr),axis=0)
    del_fidelityminus_origin_mean = np.average(np.array(del_fidelityminus_origin_arr),axis=0)
    del_fidelityminus_complete_mean = np.average(np.array(del_fidelityminus_complete_arr),axis=0)
    del_finalfidelity_complete_mean = np.average(np.array(del_finalfidelity_complete_arr), axis=0)
    del_fvaluefidelity_complete_mean = np.average(np.array(del_fvaluefidelity_complete_arr), axis=0)
    sparsity_edges_mean = np.average(np.array(sparsity_edges_arr),axis=0)
    fidelity_nodes_mean = np.average(np.array(fidelity_nodes_arr),axis=0)
    fidelity_origin_nodes_mean = np.average(np.array(fidelity_origin_nodes_arr),axis=0)
    fidelity_complete_nodes_mean = np.average(np.array(fidelity_complete_nodes_arr),axis=0)
    fidelityminus_nodes_mean = np.average(np.array(fidelityminus_nodes_arr),axis=0)
    fidelityminus_origin_nodes_mean = np.average(np.array(fidelityminus_origin_nodes_arr),axis=0)
    fidelityminus_complete_nodes_mean = np.average(np.array(fidelityminus_complete_nodes_arr),axis=0)
    finalfidelity_complete_nodes_mean = np.average(np.array(finalfidelity_complete_nodes_arr), axis=0)
    sparsity_nodes_mean = np.average(np.array(sparsity_nodes_arr),axis=0)

    print("MO_auc_mean =", np.mean(auc_all))
    print("MO_ndcg_mean =", np.mean(ndcg_all))
    print("MO_simula_mean =", list(simula_mean))
    print("MO_simula_origin_mean =", list(simula_origin_mean))
    print("MO_simula_complete_mean =", list(simula_complete_mean))
    print("MO_fidelity_mean = ", list(fidelity_mean))
    print("MO_fidelity_origin_mean = ", list(fidelity_origin_mean))
    print("MO_fidelity_complete_mean = ", list(fidelity_complete_mean))
    print("MO_fidelityminus_mean = ", list(fidelityminus_mean))
    print("MO_fidelityminus_origin_mean = ", list(fidelityminus_origin_mean))
    print("MO_fidelityminus_complete_mean = ", list(fidelityminus_complete_mean))
    print("MO_finalfidelity_complete_mean = ", list(finalfidelity_complete_mean))
    print("MO_fvaluefidelity_complete_mean = ", list(fvaluefidelity_complete_mean))
    print("MO_del_fidelity_mean = ", list(del_fidelity_mean))
    print("MO_del_fidelity_origin_mean = ", list(del_fidelity_origin_mean))
    print("MO_del_fidelity_complete_mean = ", list(del_fidelity_complete_mean))
    print("MO_del_fidelityminus_mean = ", list(del_fidelityminus_mean))
    print("MO_del_fidelityminus_origin_mean = ", list(del_fidelityminus_origin_mean))
    print("MO_del_fidelityminus_complete_mean = ", list(del_fidelityminus_complete_mean))
    print("MO_del_finalfidelity_complete_mean = ", list(del_finalfidelity_complete_mean))
    print("MO_del_fvaluefidelity_complete_mean = ", list(del_fvaluefidelity_complete_mean))
    print("MO_sparsity_edges_mean =", list(sparsity_edges_mean))
    print("MO_fidelity_nodes_mean =", list(fidelity_nodes_mean))
    print("MO_fidelity_origin_nodes_mean =", list(fidelity_origin_nodes_mean))
    print("MO_fidelity_complete_nodes_mean =", list(fidelity_complete_nodes_mean))
    print("MO_fidelityminus_nodes_mean =", list(fidelityminus_nodes_mean))
    print("MO_fidelityminus_origin_nodes_mean =", list(fidelityminus_origin_nodes_mean))
    print("MO_fidelityminus_complete_nodes_mean =", list(fidelityminus_complete_nodes_mean))
    print("MO_finalfidelity_complete_nodes_mean =", list(finalfidelity_complete_nodes_mean))
    print("MO_sparsity_nodes_mean =", list(sparsity_nodes_mean))

    f_mean.write("MO_auc_mean = {}".format(np.mean(auc_all))+ "\n")
    f_mean.write("MO_ndcg_mean = {}".format(np.mean(ndcg_all))+ "\n")
    f_mean.write("MO_PN_mean = {}".format(np.mean(PN_all))+ "\n")
    f_mean.write("MO_PS_mean = {}".format(np.mean(PS_all))+ "\n")
    f_mean.write("MO_FNS_mean = {}".format(np.mean(FNS_all))+ "\n")
    f_mean.write("MO_size_mean = {}".format(np.mean(size_all))+ "\n")
    f_mean.write("MO_acc_mean = {}".format(np.mean(acc_all))+ "\n")
    f_mean.write("MO_pre_mean = {}".format(np.mean(pre_all))+ "\n")
    f_mean.write("MO_rec_mean = {}".format(np.mean(rec_all))+ "\n")
    f_mean.write("MO_f1_mean = {}".format(np.mean(f1_all))+ "\n")
    f_mean.write("MO_simula_mean = {}".format(list(simula_mean))+ "\n")
    f_mean.write("MO_simula_origin_mean = {}".format(list(simula_origin_mean))+ "\n")
    f_mean.write("MO_simula_complete_mean = {}".format(list(simula_complete_mean))+ "\n")
    f_mean.write("MO_fidelity_mean = {}".format(list(fidelity_mean))+ "\n")
    f_mean.write("MO_fidelity_origin_mean = {}".format(list(fidelity_origin_mean))+ "\n")
    f_mean.write("MO_fidelity_complete_mean = {}".format(list(fidelity_complete_mean))+ "\n")
    f_mean.write("MO_fidelityminus_mean = {}".format(list(fidelityminus_mean))+"\n")
    f_mean.write("MO_fidelityminus_origin_mean = {}".format(list(fidelityminus_origin_mean))+"\n")
    f_mean.write("MO_fidelityminus_complete_mean = {}".format(list(fidelityminus_complete_mean))+"\n")
    f_mean.write("MO_finalfidelity_complete_mean = {}".format(list(finalfidelity_complete_mean))+"\n")
    f_mean.write("MO_fvaluefidelity_complete_mean = {}".format(list(fvaluefidelity_complete_mean)) + "\n")
    f_mean.write("MO_del_fidelity_mean = {}".format(list(del_fidelity_mean))+ "\n")
    f_mean.write("MO_del_fidelity_origin_mean = {}".format(list(del_fidelity_origin_mean))+ "\n")
    f_mean.write("MO_del_fidelity_complete_mean = {}".format(list(del_fidelity_complete_mean))+ "\n")
    f_mean.write("MO_del_fidelityminus_mean = {}".format(list(del_fidelityminus_mean))+"\n")
    f_mean.write("MO_del_fidelityminus_origin_mean = {}".format(list(del_fidelityminus_origin_mean))+"\n")
    f_mean.write("MO_del_fidelityminus_complete_mean = {}".format(list(del_fidelityminus_complete_mean))+"\n")
    f_mean.write("MO_del_finalfidelity_complete_mean = {}".format(list(del_finalfidelity_complete_mean))+"\n")
    f_mean.write("MO_del_fvaluefidelity_complete_mean = {}".format(list(del_fvaluefidelity_complete_mean)) + "\n")
    f_mean.write("MO_sparsity_edges_mean = {}".format(list(sparsity_edges_mean))+ "\n")
    f_mean.write("MO_fidelity_nodes_mean = {}".format(list(fidelity_nodes_mean))+ "\n")
    f_mean.write("MO_fidelity_origin_nodes_mean = {}".format(list(fidelity_origin_nodes_mean))+ "\n")
    f_mean.write("MO_fidelity_complete_nodes_mean = {}".format(list(fidelity_complete_nodes_mean))+ "\n")
    f_mean.write("MO_fidelityminus_nodes_mean = {}".format(list(fidelityminus_nodes_mean))+"\n")
    f_mean.write("MO_fidelityminus_origin_nodes_mean = {}".format(list(fidelityminus_origin_nodes_mean))+"\n")
    f_mean.write("MO_fidelityminus_complete_nodes_mean = {}".format(list(fidelityminus_complete_nodes_mean))+"\n")
    f_mean.write("MO_finalfidelity_complete_nodes_mean = {}".format(list(finalfidelity_complete_nodes_mean))+"\n")
    f_mean.write("MO_sparsity_nodes_mean = {}".format(list(sparsity_nodes_mean))+ "\n")
    f_mean.close()



def main(iteration, optimal_method, loss_type, hidden_layer, dominant_loss, angle, coff=None):
    args.elr = 0.03
    args.eepochs = 10000
    args.coff_t0=1.0   #5.0
    args.coff_te=0.05  #2.0
    args.coff_size = 1.0
    args.budget = -1
    args.coff_ent = 1.0
    args.concat = False
    args.topk_arr = list(range(10))+list(range(10,101,5))
    #args.topk_arr = [10]
    args.loss_flag = loss_type
    args.hidden_layer = hidden_layer
    args.ema_beta = 0.01
    #args.batch_size = None

    args.batch_size = 32
    #args.test_batch_size = 32

    args.dataset="Computers"

    save_map = "MO_LISA_TEST_LOGS_NEW/"+args.dataset.upper() +"_loss"
    save_map = save_map + "_" + loss_type
    if hidden_layer:
       save_map = save_map +"_hidden_" + hidden_layer
    save_map = save_map + "_iter"+str(iteration)+"_elr"+str(args.elr).replace(".","")+"_epoch"+str(args.eepochs)
    if optimal_method == "weightsum":
        args.coff_diff = coff
        args.coff_cf = coff
        save_map = save_map + "_coffdiff"+str(args.coff_diff).replace(".","")+ "_coffcf"+str(args.coff_cf).replace(".","")
    if "MO" in optimal_method or optimal_method=="getGrad":
        args.coff_diff = 1.0
        args.coff_cf =1.0
    save_map = save_map + "_"+optimal_method
    if angle:
        save_map = save_map + "_angle"+str(angle)
    if dominant_loss[0]:
         save_map = save_map + "_dominant_"+dominant_loss[0]
    
    if args.batch_size is not None:
        save_map = save_map + "_fiveGCN_batch" + str(args.batch_size) + "/"
    else:
        save_map = save_map + "_fiveGCN/"

    print("save_map: ", save_map)
    if not os.path.exists(save_map):
        os.makedirs(save_map)
    
    args.add_edge_num = 0
    if args.add_edge_num==0:
        dataset_filename = 'datasets/'+args.dataset+'/raw/' + args.dataset + '.pkl'
    else:
        dataset_filename = 'datasets/'+args.dataset+'/raw/' + args.dataset + '-' + str(args.add_edge_num) + '.pkl'
    
    #train
    args.model_filename = args.dataset+'_' + str(args.add_edge_num)
    #test    
    test_flag = False
    testmodel_filename = args.dataset + '_'+ str(args.add_edge_num) + '_BEST'
    plot_flag = False

    if args.dataset=="BA_community":
        args.sample_bias = 0.5

    args.fix_exp=12
    args.mask_thresh = 0.5

    '''if test_flag:
        log_filename = save_map + "log_test.txt"
    else:
        log_filename = save_map + "log.txt"
    '''

    if test_flag:
        for topk in args.topk_arr:
            test_main_onetopk(topk, save_map, dataset_filename, testmodel_filename, plot_flag)
    else:
        log_filename = save_map + "log.txt"
        for iter in [iteration]:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
            #args.device = "cpu"
            print("Starting iteration: {}".format(iter))
            if not os.path.exists(save_map+str(iter)):
                os.makedirs(save_map+str(iter))
            #load data
            if args.dataset == "Computers":
                dataset = Amazon(root=args.dataset_root, name=args.dataset)
            #data = dataset.data
            num_class = dataset.num_classes
            label = dataset.data.y
            adj =  torch.sparse_coo_tensor(indices=dataset.data.edge_index, values= torch.ones(dataset.data.edge_index.shape[1])).to_dense()
            #adj = csr_matrix(adj.numpy())
            edge_label_matrix = None
            num_nodes = dataset.data.x.shape[0]
            node_index = list(range(num_nodes))
            random.seed(2023)
            random.shuffle(node_index)
            trainnodes_0 = node_index[ : math.floor(num_nodes*0.8)]
            valnodes_0 = node_index[math.floor(num_nodes*0.8) : math.floor(num_nodes*0.9)]
            testnodes_0 = node_index[math.floor(num_nodes*0.9) : ]
            print("train nodes count:", len(trainnodes_0))
            print("test nodes count:", len(testnodes_0))

            '''train_mask = torch.tensor([False] * num_nodes)
            val_mask = torch.tensor([False] * num_nodes)
            test_mask = torch.tensor([False] * num_nodes)
            train_mask[node_index[ : math.floor(num_nodes*0.8)]] =True
            val_mask[node_index[math.floor(num_nodes*0.8) : math.floor(num_nodes*0.9)]] = True
            test_mask[node_index[math.floor(num_nodes*0.9) : ]] = True'''

            #load trained GCN (PGE)
            ''' model = load_GCN_PG(GNNmodel_ckpt_path, input_dim=features.shape[1], output_dim=y_train.shape[1], device=device)
            model.eval()
            embeds = model.embedding((features_tensor,support_tensor)).cpu().detach().numpy()'''
            #load trained GCN (survey)
            GNNmodel_ckpt_path = osp.join('GNN_checkpoint', args.dataset+"_"+str(iter), 'gcn_best.pth')
            print("GNNmodel_ckpt_path", GNNmodel_ckpt_path)
            model = load_gnnNets_NC(GNNmodel_ckpt_path, input_dim=dataset.data.x.shape[1], output_dim=num_class, device = args.device)
            model.eval()
            #logits, outputs, embeds, hidden_embs = model(data.to(args.device))
            #embeds = embeds.cpu().detach().numpy()

            explainer = ExplainerMO(model=model, args=args)
            explainer.to(args.device)

            hops = len(args.hiddens.split('-'))
            #extractor = Extractor(adj, edge_index, features, edge_label_matrix, embeds, label, hops)
            extractor = Extractor(adj, edge_label_matrix, hops)
            print("start filter nodes")
            trainnodes = []
            for node in trainnodes_0:
                sub_node, _, _ = extractor.subgraph(node)
                if len(sub_node) != 0:
                    trainnodes.append(node)
            valnodes = []
            for node in valnodes_0:
                sub_node, _, _ = extractor.subgraph(node)
                if len(sub_node) != 0:
                    valnodes.append(node)
            testnodes = []
            for node in testnodes_0:
                sub_node, _, _ = extractor.subgraph(node)
                if len(sub_node) != 0:
                    testnodes.append(node)
            trainnodes = trainnodes[:3]
            valnodes = valnodes[:3]
            testnodes = testnodes[:3]
            print("filted train nodes count:", len(trainnodes))
            print("filted test nodes count:", len(testnodes))

            f = open(save_map + str(iter) + "/" + "LOG_" + args.model_filename + "_BEST.txt", "w")
            tik = time.time()
            if "MO" in optimal_method:
                train_MO(iter, args, model, explainer, extractor, save_map, trainnodes, valnodes, dataset.data)
            else:
                train(iter, args, model, explainer, extractor, save_map, trainnodes, valnodes, dataset.data)
            tok = time.time()
            f.write("train time,{}".format(tok - tik) + "\n\n")




if __name__ == "__main__":
    iteration = 0
    optimal_method_arr = ["MO-GEAR"]  #weightsum, MO-PCGrad, MO-GEAR, MO-CAGrad
    loss_type_arr = ["pdiff_hidden_CF_LM_conn"]    #"ce", "ce_hidden", "kl", "kl_hidden", "pl_value", "pl_value_hidden"
    for optimal_method in optimal_method_arr:
        for loss_type in loss_type_arr: 
            if "weightsum" in optimal_method:
                coff_arr = [1.0, 5.0, 10.0, 50.0, 100.0]
                for coff in coff_arr:
                    main(iteration, optimal_method, loss_type, "alllayer", (None,None), None, coff)
            elif "getGrad" in optimal_method:
                main(iteration, optimal_method, loss_type, "alllayer", ("pdiff-CF-mean",[0,2]), None)
            elif "MO" in optimal_method:
                #dominant_loss_dic = {"pdiff": 0, "hidden":1 , "CF": 2, "mask":3, "conn":4 } 
                dominant_loss_dic = {"pdiff-CF-disector": [0,2]}  
                #dominant_loss_dic = {None: None}   
                angle_arr = [45]    #[90, 75, 60, 45, 30, 15]
                for dominant_loss in dominant_loss_dic.items():
                        for angle in angle_arr:
                            if "hidden" in loss_type:
                                hidden_layer_arr = ["alllayer"]
                                for hidden_layer in hidden_layer_arr:
                                    main(iteration, optimal_method, loss_type, hidden_layer, dominant_loss, angle)
                            else:
                                main(iteration, optimal_method, loss_type, None, dominant_loss, angle)
    #testnodes = list(range(700))
    #test_explainmodel(testnodes)


