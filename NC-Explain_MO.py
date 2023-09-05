#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#CUDA_LAUNCH_BLOCKING=1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append('./codes/fornode/')
import time
from codes.fornode.config import args
#from tqdm import tqdm
# import tensorflow as tf
from codes.fornode.utils import *
from codes.fornode.metricsHidden import *
import numpy as np
from codes.fornode.ExtractorNew import ExtractorNew
from codes.fornode.ExplainerMOCopy import ExplainerMOCopy
from scipy.sparse import coo_matrix,csr_matrix
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch
import torch.optim
from torch.optim import Adam, SGD

from codes.load_GNNNets_hidden import load_gnnNets_NC, load_GCN_PG
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_networkx
import os.path as osp

from codes.mograd import MOGrad, Grad
import math
import operator
from collections import OrderedDict

def main(iteration, optimal_method, loss_type, hidden_layer, dominant_loss, angle, coff=None):
    def train_MO(iter, args):
        t0 = args.coff_t0
        t1 = args.coff_te
        #clip_value_min = -2.0
        #clip_value_max = 2.0
        epochs = args.eepochs
        model.eval()
        explainer.train()
        best_auc = 0.0
        #best_ndcg = 0.0
        #best_loss = 0.0
        best_decline = 0.0
        best_F_fidelity = 0.0
        best_del_F_fidelity = 0.0
        f_train = open(save_map + str(iter) + "/" + "train_LOG_" + args.model_filename + "_BEST.txt", "w")
        f_train_grad = open(save_map + str(iter) + "/" + "train_gradient_LOG_" + args.model_filename + "_BEST.txt", "w")
        f_train_parameter = open(save_map + str(iter) + "/" + "train_parameter_LOG_" + args.model_filename + "_BEST.txt", "w")
        #f_train_embed = open(save_map + str(iter) + "/" + "train_embedding_LOG.txt", "w")
        random_edge_mask_dict = {}
        for node in trainnodes:
            newid = remap[node]
            sub_adj = sub_adjs[newid]
            random_edge_mask = torch.Tensor()
            for ri in range(5):
                if ri==0:
                    random_edge_mask = torch.rand(len(sub_adj.data)).unsqueeze(0)
                else:
                    random_edge_mask = torch.cat((random_edge_mask, torch.rand(len(sub_adj.data)).unsqueeze(0)),dim=0)
            random_edge_mask = torch.mean(random_edge_mask, dim=0)
            random_edge_mask_dict[node] = random_edge_mask

        optimizer = Adam(explainer.elayers.parameters(), lr=args.elr)
        #optimizer = SGD(explainer.elayers.parameters(), lr=args.elr)
        optimizer = MOGrad(optimizer)
        #sim_obj = 0.0
        sim_obj = round( math.cos(math.radians(angle)), 3)
        dominant_index = dominant_loss[1]
        modify_index_arr = None
        coslist = None
        for epoch in range(epochs):
            train_accs = []
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
            grads = torch.tensor([])
            grads_new = torch.tensor([])
            #tmp = float(1.0*np.power(0.05,epoch/epochs))
            tmp = float(t0 * np.power(t1 / t0, epoch /epochs))
            for node in trainnodes:
                newid = remap[node]
                sub_output = sub_outputs[newid]
                old_pred_label = torch.argmax(sub_output, 1)

                sub_feature = sub_features[newid]
                sub_adj = sub_adjs[newid]
                nodeid = 0
                sub_embed = sub_embeds[newid]
                if not isinstance(sub_embed, torch.Tensor):
                    sub_embed = torch.Tensor(sub_embeds[newid]).to(args.device)
                sub_hidden_emb = sub_hidden_embs[newid]
                random_edge_mask = random_edge_mask_dict[node].to(args.device)
                new_pred, cf_pred, masked_embed, new_hidden_emb = explainer((sub_feature, sub_adj, nodeid, sub_embed, tmp), training=True)
                
                #random mask predict
                sub_edge_index = dense_to_sparse(torch.tensor(sub_adj.todense()))[0]
                explainer.__set_masks__(sub_feature, sub_edge_index, 1-random_edge_mask)
                data = Data(x=sub_feature, edge_index=sub_edge_index)
                random_output, random_probs, cf_embed, _ = model(data)
                random_cf_node_pred = random_probs[nodeid]
                explainer.__clear_masks__()
                
                if "ce" in args.loss_flag:
                    l,pl,ll,cl, hl = explainer.loss_ce_hidden(new_hidden_emb, sub_hidden_emb, new_pred, old_pred_label, sub_label_tensors[newid], nodeid)
                    vl = 0
                    sl= 0
                    cfl = 0
                elif "kl" in args.loss_flag:
                    l, pl, ll, cl, hl, cfl = explainer.loss_kl_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output[nodeid])
                    vl = 0
                    sl = 0
                elif "pl" in args.loss_flag:
                    l, pl, ll, cl, vl, sl, hl, cfl = explainer.loss_pl_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output[nodeid])
                elif "pdiff" in args.loss_flag:
                    l, pl, ll, cl, hl, cfl = explainer.loss_diff_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output[nodeid], random_cf_node_pred)
                    vl = 0
                    sl = 0  
                elif "CF" == args.loss_flag:
                    l, pl, ll, cl, hl, cfl = explainer.loss_cf_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output[nodeid])
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
                '''f_train_embed.write("node={}\n".format(node) )
                f_train_embed.write("cfloss={}\n".format(cfl.item()) )
                f_train_embed.write("origin_pred={}\n".format(sub_output[nodeid].cpu().detach().numpy()) )
                f_train_embed.write("factual_pred={}\n".format(new_pred.cpu().detach().numpy()) )
                f_train_embed.write("cf_pred={}\n".format(cf_pred.cpu().detach().numpy()) )
                f_train_embed.write("label={}\n".format(sub_labels[newid][nodeid].argmax(0)) )
                f_train_embed.write("origin_label={}\n".format(sub_output[nodeid].argmax(0).item()) )
                f_train_embed.write("sub_embed={}\n".format(sub_embed[nodeid].cpu().detach().numpy().tolist()) )
                f_train_embed.write("masked_embed={}\n".format(masked_embed.cpu().detach().numpy().tolist()) )
                for i in range(len(sub_hidden_emb)):
                    f_train_embed.write("sub_hidden_emb_layer{}={}\n".format(str(i), sub_hidden_emb[i][nodeid].cpu().detach().numpy().tolist()) )
                for i in range(len(new_hidden_emb)):
                    f_train_embed.write("masked_sub_hidden_emb_layer{}={}\n".format(str(i), new_hidden_emb[i][nodeid].cpu().detach().numpy().tolist()) )'''
                #print("node",node, ", cfl",cfl.item(), ", origin_pred", sub_output[nodeid].cpu().detach().numpy())
                #print("cf_pred", cf_pred.cpu().detach().numpy(), ", factual_pred",new_pred.cpu().detach().numpy())
            
            #f_train_embed.write("total_trainnodes={}\n".format(len(trainnodes)) )
            #print("total train nodes", len(trainnodes))
            optimizer.zero_grad()

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
                    dominant_index, modify_index_arr, coslist, grads, grads_new = optimizer.pc_backward_dominant(losses, dominant_index)
                else:
                    modify_index_arr, coslist, grads, grads_new = optimizer.pc_backward(losses)
            elif "GradVac" in optimal_method:
                if dominant_loss[1] is not None:
                    modify_index_arr, coslist, cur_sim_obj, grads, grads_new = optimizer.pc_backward_gradvac_dominant(losses, dominant_index, sim_obj)
                    #sim_obj = cur_sim_obj
                else:
                    modify_index_arr, coslist, cur_sim_obj, grads, grads_new = optimizer.pc_backward_gradvac(losses, sim_obj)
                    #sim_obj = cur_sim_obj
            elif "getGrad" in optimal_method:
                coslist, grads, grads_new = optimizer.get_grads(losses)
                loss.backward()   #original
            elif "CAGrad" in optimal_method:
                grads, grads_new = optimizer.cagrad_backward(losses, dominant_index)
            
            f_train_grad.write("epoch={}\n".format(epoch))
            f_train_grad.write("grads={}\n".format([list(g.cpu().numpy()) for g in grads]))
            f_train_grad.write("grads_new={}\n".format([list(g.cpu().numpy()) for g in grads_new]))

            #torch.nn.utils.clip_grad_value_(explainer.elayers.parameters(), clip_value_max)
            #torch.nn.utils.clip_grad_value_(explainer.elayers.parameters(), clip_value_min)
            optimizer.step()

            '''f_train_parameter.write("epoch={}\n".format(epoch))
            for k,v in dict(explainer.elayers.state_dict()).items():
                f_train_parameter.write("{}={}\n".format(k, v.cpu().numpy().tolist()))'''

            #global reals
            #global preds
            #reals = []
            #preds = []
            #ndcgs = []
            x_collector = XCollector()
            metric = MaskoutMetric(model, args)
            classify_acc = 0
            for node in valnodes:
                newid = remap[node]
                #use fullgraph prediction
                #origin_pred = outputs[node].to(args.device)
                #use subgraph prediction
                nodeid = 0
                origin_pred = sub_outputs[newid][nodeid]
                #auc_onenode, ndcg_onenode, r_mask, real, pred  = explain_test(explainer, node, origin_pred)
                #ndcgs.append(ndcg_onenode)
                #reals.extend(real)
                #preds.extend(pred)
                pred_mask, related_preds_dict = metric.metric_pg_del_edges(nodeid, explainer, sub_adjs[newid], sub_features[newid], sub_embeds[newid], sub_labels[newid], origin_pred, sub_nodes[newid], testing=False)
                x_collector.collect_data(pred_mask, related_preds_dict[10], label=0)
                if related_preds_dict[10][0]["origin_label"] == torch.argmax(related_preds_dict[10][0]["masked"]):
                    classify_acc = classify_acc + 1

            classify_acc = classify_acc/len(valnodes)
            '''if len(np.unique(reals))==1 or len(np.unique(preds))==1:
                auc = -1
            else:
                auc = roc_auc_score(reals, preds)
            ndcg = np.mean(ndcgs)'''
            ndcg = 0
            auc = 0
            eps = 1e-7
            fidelityplus = x_collector.fidelity_complete
            fidelityminus = x_collector.fidelityminus_complete
            #decline = fidelityplus - fidelityminus
            #F_fidelity = 2/(1/fidelityplus +1/(1-fidelityminus))
            decline = torch.sub(fidelityplus, fidelityminus)
            #F_fidelity = 2/ (torch.true_divide(1, (fidelityplus+eps)) + torch.true_divide(1, (torch.sub(1, fidelityminus)+eps)))
            F_fidelity = 2/(1/fidelityplus +1/(1/fidelityminus))

            del_fidelityplus = x_collector.del_fidelity_complete
            del_fidelityminus = x_collector.del_fidelityminus_complete
            #del_F_fidelity = 2/(1/del_fidelity +1/(1-del_fidelityminus))
            #del_F_fidelity = 2/ (torch.true_divide(1, del_fidelityplus) + torch.true_divide(1, torch.sub(1, del_fidelityminus)))
            del_F_fidelity = 2/(1/del_fidelityplus +1/(1/del_fidelityminus))

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
        f_train.close()
        f_train_grad.close()
        f_train_parameter.close()
        #f_train_embed.close()
        torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename + "_LAST.pt")
        torch.save(explainer.state_dict(), save_map + str(iter) + "/"  + args.model_filename + "_LAST.pt")


    def train(iter, args):
        t0 = args.coff_t0
        t1 = args.coff_te
        #clip_value_min = -2.0
        #clip_value_max = 2.0
        epochs = args.eepochs
        model.eval()
        explainer.train()
        best_auc = 0.0
        #best_ndcg = 0.0
        #best_loss = 0.0
        best_decline = 0.0
        best_F_fidelity = 0.0
        best_del_F_fidelity = 0.0
        f_train = open(save_map + str(iter) + "/" + "train_LOG_" + args.model_filename + "_BEST.txt", "w")
        f_train_grad = open(save_map + str(iter) + "/" + "train_gradient_LOG_" + args.model_filename + "_BEST.txt", "w")
        f_train_parameter = open(save_map + str(iter) + "/" + "train_parameter_LOG_" + args.model_filename + "_BEST.txt", "w")
        #f_train_embed = open(save_map + str(iter) + "/" + "train_embedding_LOG.txt", "w")
        random_edge_mask_dict = {}
        for node in trainnodes:
            newid = remap[node]
            sub_adj = sub_adjs[newid]
            random_edge_mask = torch.Tensor()
            for ri in range(5):
                if ri==0:
                    random_edge_mask = torch.rand(len(sub_adj.data)).unsqueeze(0)
                else:
                    random_edge_mask = torch.cat((random_edge_mask, torch.rand(len(sub_adj.data)).unsqueeze(0)),dim=0)
            random_edge_mask = torch.mean(random_edge_mask, dim=0)
            random_edge_mask_dict[node] = random_edge_mask

        optimizer = Adam(explainer.elayers.parameters(), lr=args.elr)
        #optimizer = SGD(explainer.elayers.parameters(), lr=args.elr)
        for epoch in range(epochs):
            train_accs = []
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
            for node in trainnodes:
                newid = remap[node]
                sub_output = sub_outputs[newid]
                old_pred_label = torch.argmax(sub_output, 1)

                sub_feature = sub_features[newid]
                sub_adj = sub_adjs[newid]
                nodeid = 0
                sub_embed = sub_embeds[newid]
                if not isinstance(sub_embed, torch.Tensor):
                    sub_embed = torch.Tensor(sub_embeds[newid]).to(args.device)
                sub_hidden_emb = sub_hidden_embs[newid]
                random_edge_mask = random_edge_mask_dict[node].to(args.device)
                new_pred, cf_pred, masked_embed, new_hidden_emb = explainer((sub_feature, sub_adj, nodeid, sub_embed, tmp), training=True)
                
                #conterfactual predict
                sub_edge_index = dense_to_sparse(torch.tensor(sub_adj.todense()))[0]
                explainer.__set_masks__(sub_feature, sub_edge_index, 1-random_edge_mask)
                data = Data(x=sub_feature, edge_index=sub_edge_index)
                random_output, random_probs, cf_embed, _ = model(data)
                random_cf_node_pred = random_probs[nodeid]
                explainer.__clear_masks__()
                
                if "ce" in args.loss_flag:
                    l,pl,ll,cl, hl = explainer.loss_ce_hidden(new_hidden_emb, sub_hidden_emb, new_pred, old_pred_label, sub_label_tensors[newid], nodeid)
                    vl = 0
                    sl= 0
                    cfl = 0
                elif "kl" in args.loss_flag:
                    l, pl, ll, cl, hl, cfl = explainer.loss_kl_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output[nodeid])
                    vl = 0
                    sl = 0
                elif "pl" in args.loss_flag:
                    l, pl, ll, cl, vl, sl, hl, cfl = explainer.loss_pl_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output[nodeid])
                elif "pdiff" in args.loss_flag:
                    l, pl, ll, cl, hl, cfl = explainer.loss_diff_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output[nodeid], random_cf_node_pred)
                    vl = 0
                    sl = 0  
                elif "CF" == args.loss_flag:
                    l, pl, ll, cl, hl, cfl = explainer.loss_cf_hidden(new_hidden_emb, sub_hidden_emb, new_pred, cf_pred, sub_output[nodeid])
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
            
            '''if "getGrad" in optimal_method:
                coslist, grads, grads_new = optimizer.get_grads(losses)
                f_train_grad.write("epoch={}\n".format(epoch))
                f_train_grad.write("grads={}\n".format([list(g.cpu().numpy()) for g in grads]))
                f_train_grad.write("grads_new={}\n".format([list(g.cpu().numpy()) for g in grads_new]))'''
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
            optimizer.zero_grad()
            loss.backward()

            #torch.nn.utils.clip_grad_value_(explainer.elayers.parameters(), clip_value_max)
            optimizer.step()

            #global reals
            #global preds
            reals = []
            preds = []
            ndcgs = []
            x_collector = XCollector()
            metric = MaskoutMetric(model, args)
            classify_acc = 0
            for node in valnodes:
                newid = remap[node]
                #use fullgraph prediction
                #origin_pred = outputs[node].to(args.device)
                #use subgraph prediction
                nodeid = 0
                origin_pred = sub_outputs[newid][nodeid]
                auc_onenode, ndcg_onenode, real, pred  = explain_test(explainer, node, origin_pred, validation=True)
                ndcgs.append(ndcg_onenode)
                reals.extend(real)
                preds.extend(pred)
                pred_mask, related_preds_dict = metric.metric_pg_del_edges(nodeid, explainer, sub_adjs[newid], sub_features[newid], sub_embeds[newid], sub_labels[newid], origin_pred, sub_nodes[newid], testing=False)
                x_collector.collect_data(pred_mask, related_preds_dict[10], label=0)
                if related_preds_dict[10][0]["origin_label"] == torch.argmax(related_preds_dict[10][0]["masked"]):
                    classify_acc =classify_acc + 1

            classify_acc = classify_acc/len(valnodes)
            if len(np.unique(reals))==1 or len(np.unique(preds))==1:
                auc = -1
            else:
                auc = roc_auc_score(reals, preds)
            ndcg = np.mean(ndcgs)
            fidelityplus = x_collector.fidelity_complete
            fidelityminus = x_collector.fidelityminus_complete
            #decline = fidelityplus - fidelityminus
            #F_fidelity = 2/(1/fidelityplus +1/(1-fidelityminus))
            decline = operator.sub(fidelityplus, fidelityminus)
            #F_fidelity = 2/ (operator.truediv(1, fidelityplus) + operator.truediv(1, operator.sub(1, fidelityminus)))
            F_fidelity = 2/(1/fidelityplus +1/(1/fidelityminus))

            del_fidelityplus = x_collector.del_fidelity_complete
            del_fidelityminus = x_collector.del_fidelityminus_complete
            #del_F_fidelity = 2/(1/del_fidelity +1/(1-del_fidelityminus))
            #del_F_fidelity = 2/ (operator.truediv(1, del_fidelityplus) + operator.truediv(1, operator.sub(1, del_fidelityminus)))
            del_F_fidelity = 2/(1/del_fidelityplus +1/(1/del_fidelityminus))

            if epoch == 0:
                best_loss = loss
                best_auc = auc
                best_decline = decline
                best_F_fidelity = F_fidelity
                best_del_F_fidelity = del_F_fidelity
            if auc >= best_auc:
                print("epoch", epoch, "saving best auc model...")
                f_train.write("saving best auc model...\n")
                best_auc = auc
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename + "_BEST_auc.pt")
                torch.save(explainer.state_dict(), save_map + str(iter) + "/"  + args.model_filename + "_BEST_auc.pt")
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

            print("epoch", epoch, "loss", loss, "hidden_loss",hidden_loss, "pred_loss",pred_loss, "lap_loss",lap_loss, "con_loss",con_loss, "sort_loss",sort_loss, "value_loss",value_loss, "cf_loss", cf_loss, "ndcg", ndcg, "auc", auc, "decline",decline, "F_fidelity",F_fidelity, "classify_acc",classify_acc)
            f_train.write("epoch,{}".format(epoch) + ",loss,{}".format(loss) + ",ndcg,{}".format(ndcg) + ",auc,{}".format(auc)+ ",2,{}".format(fidelityplus) + ",1,{}".format(fidelityminus)  + ",decline,{}".format(decline)  + ",hidden_loss,{}".format(hidden_loss) + ",pred_loss,{}".format(pred_loss) + ",size_loss,{}".format(lap_loss)+ ",con_loss,{}".format(con_loss)+ ",sort_loss,{}".format(sort_loss) + ",value_loss,{}".format(value_loss) + ",cf_loss,{}".format(cf_loss)+ ",F_fidelity,{}".format(F_fidelity)+",classify_acc,{}".format(classify_acc))
            if coslist is not None:
                #print("cosvalue:", coslist)
                f_train.write(",cosvalue,{}".format("_".join(coslist)) +  "\n")
            else:
                f_train.write("\n")
        f_train.close()
        f_train_grad.close()
        f_train_parameter.close()
        #f_train_embed.close()
        torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename + "_LAST.pt")
        torch.save(explainer.state_dict(), save_map + str(iter) + "/"  + args.model_filename + "_LAST.pt")


    def acc(sub_adj, sub_edge_label):
        mask = explainer.masked_adj.cpu().detach().numpy()
        real = []
        pred = []
        sub_edge_label = sub_edge_label.todense()
        for r,c in list(zip(sub_adj.row,sub_adj.col)):
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


    def explain_test(explainer, node, origin_pred, validation=False):
        newid = remap[node]
        sub_adj, sub_feature, sub_embed, sub_label, sub_edge_label = sub_adjs[newid], sub_features[newid], sub_embeds[
            newid], sub_labels[newid], sub_edge_labels[newid]

        explainer.eval()
        nodeid = 0
        masked_pred, cf_pred, masked_emb, masked_hidden_emb = explainer((sub_feature, sub_adj, nodeid, sub_embed, 1.0))

        #label = np.argmax(sub_label, -1)
        auc, real, pred = acc(sub_adj, sub_edge_label)
        #print("origin_pred", origin_pred)
        #print("masked_pred", masked_pred)
        ndcg = 0
        if not validation:
            ndcg, r_mask, _, _ = rho_ndcg(origin_pred, masked_pred, len(masked_pred))
        return auc, ndcg, real, pred


    def record_hidden_info(explainer, node, loss_type):
        newid = remap[node]
        sub_adj, sub_feature, sub_embed, sub_label = sub_adjs[newid], sub_features[newid], sub_embeds[newid], sub_labels[newid]

        explainer.eval()
        nodeid = 0
        masked_pred, cf_pred, masked_emb, masked_hidden_emb = explainer((sub_feature, sub_adj, nodeid, sub_embed, 1.0))
        sub_edge_index = [sub_adj.row.tolist(), sub_adj.col.tolist()]
        edge_mask = explainer.masked_adj[sub_adj.row, sub_adj.col].detach()
        edges_idx_desc = edge_mask.reshape(-1).sort(descending=True).indices.cpu().numpy().tolist()
        sub_y = sub_label.argmax(-1).tolist()

        if "hidden" in loss_type:
            name_prefix = "withhidden_"
        else:
            name_prefix = "nohidden_"
        f_hidden.write("node={}".format(node) +"\n")
        f_hidden.write("sub_nodes={}".format(sub_nodes[newid]) +"\n")
        f_hidden.write("sub_edge_index={}".format(sub_edge_index) +"\n")
        f_hidden.write("sub_edge_mask={}".format(edge_mask.cpu().numpy().tolist()) +"\n")
        f_hidden.write("sub_edges_idx_desc={}".format(edges_idx_desc) +"\n")
        f_hidden.write("sub_y={}".format(sub_y) +"\n")
        sub_hidden_emb = sub_hidden_embs[newid]
        for i in range(len(masked_hidden_emb)):
            cosine_sim = F.cosine_similarity(masked_hidden_emb[i], sub_hidden_emb[i], dim=1)
            f_hidden.write(name_prefix+"hidden_emb_layer{}={}".format(i, sub_hidden_emb[i].cpu().numpy().tolist() )+"\n")
            f_hidden.write(name_prefix+"masked_hidden_emb_layer{}={}".format(i, masked_hidden_emb[i].detach().cpu().numpy().tolist() )+"\n")
            f_hidden.write(name_prefix+"masked_cosine_sim_layer{}={}".format(i, cosine_sim.cpu().tolist())+"\n")
            f_hidden.write(name_prefix+"masked_sum_cosine_sim_layer{}={}".format(i, sum(cosine_sim))+"\n")

        for top_k in [15, 20, 25, 30]:
            f_hidden.write("top_k={}".format(top_k)+"\n")
            maskimp_pred, maskimp_hidden_emb, retainimp_pred, retainimp_hidden_emb = explainer.mask_topk(nodeid, sub_feature, sub_adj, top_k)
            for i in range(len(maskimp_hidden_emb)):
                maskimp_cosine_sim = F.cosine_similarity(maskimp_hidden_emb[i], sub_hidden_emb[i], dim=1)
                f_hidden.write(name_prefix+"maskimp_hidden_emb_layer{}={}".format(i, maskimp_hidden_emb[i].detach().cpu().numpy().tolist() )+"\n")
                f_hidden.write(name_prefix+"maskimp_cosine_sim_layer{}={}".format(i, maskimp_cosine_sim.cpu().tolist())+"\n")
                f_hidden.write(name_prefix+"maskimp_sum_cosine_sim_layer{}={}".format(i, sum(maskimp_cosine_sim))+"\n")
                retainimp_cosine_sim = F.cosine_similarity(retainimp_hidden_emb[i], sub_hidden_emb[i], dim=1)
                f_hidden.write(name_prefix+"retainimp_hidden_emb_layer{}={}".format(i, retainimp_hidden_emb[i].detach().cpu().numpy().tolist() )+"\n")
                f_hidden.write(name_prefix+"retainimp_cosine_sim_layer{}={}".format(i, retainimp_cosine_sim.cpu().tolist())+"\n")
                f_hidden.write(name_prefix+"retainimp_sum_cosine_sim_layer{}={}".format(i, sum(retainimp_cosine_sim))+"\n")




    args.elr = 0.01
    args.eepochs = 100
    args.coff_t0=1.0   #5.0
    args.coff_te=0.05  #2.0
    args.coff_size = 1.0
    args.budget = -1
    args.coff_ent = 1.0
    args.concat = False
    args.topk_arr = list(range(10))+list(range(10,101,5))
    args.loss_flag = loss_type
    args.hidden_layer = hidden_layer
    args.ema_beta = 0.01
        

    #args.dataset = "BA_shapes"
    args.dataset="BA_community"
    #save_map = "LISA_TEST_LOGS/BA_COMMUNITY_hidden/"
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
    #save_map = "LISA_TEST_LOGS/BA_SHAPES_final"
    save_map = save_map + "_fiveGCN/"
    #debug loss
    #save_map = save_map + "_debugloss/"

    print("save_map: ", save_map)
    if not os.path.exists(save_map):
        os.makedirs(save_map)
    
    args.add_edge_num = 0
    if args.add_edge_num==0:
        filename = '/home/liuli/zhangym/torch_projects/datasets/'+args.dataset+'/raw/' + args.dataset + '.pkl'
    else:
        filename = '/home/liuli/zhangym/torch_projects/datasets/'+args.dataset+'/raw/' + args.dataset + '-' + str(args.add_edge_num) + '.pkl'
    #GNNmodel_ckpt_path = osp.join('checkpoint', args.dataset, 'gcn_best.pth') 
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

    if test_flag:
        log_filename = save_map + "log_test.txt"
    else:
        log_filename = save_map + "log.txt"

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
    for iter in range(iteration):
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        #args.device = "cpu"
        print("Starting iteration: {}".format(iter))
        if not os.path.exists(save_map+str(iter)):
            os.makedirs(save_map+str(iter))
        #load data
        #with open('./dataset/' + args.dataset + '.pkl', 'rb') as fin:
        with open(filename, 'rb') as fin:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pkl.load(fin)

        #edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
        adj = csr_matrix(adj)
        support = preprocess_adj(adj, norm=True)

        features_tensor = torch.tensor(features).type(torch.float32)
        edge_index = torch.LongTensor([*support[0]]).t().to(args.device)
        edge_data = torch.FloatTensor([*support[1]]).to(args.device)
        # LET OP: i moet getransposed worden om sparse tensor te maken met pytorch
        support_tensor = torch.sparse_coo_tensor(edge_index, edge_data, torch.Size([*support[2]]))
        support_tensor = support_tensor.type(torch.float32)

        all_label = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
        label = np.where(all_label)[1]
        
        #load trained GCN (PGE)
        ''' model = load_GCN_PG(GNNmodel_ckpt_path, input_dim=features.shape[1], output_dim=y_train.shape[1], device=device)
        model.eval()
        embeds = model.embedding((features_tensor,support_tensor)).cpu().detach().numpy()'''
        
        y = torch.from_numpy(label).to(args.device)
        data = Data(x=features_tensor, y=y, edge_index=edge_index)
        data.train_mask = torch.from_numpy(train_mask)
        data.val_mask = torch.from_numpy(val_mask)
        data.test_mask = torch.from_numpy(test_mask)

        #load trained GCN (survey)
        GNNmodel_ckpt_path = osp.join('checkpoint', args.dataset+"_"+str(iter), 'gcn_best.pth') 
        model = load_gnnNets_NC(GNNmodel_ckpt_path, input_dim=features.shape[1], output_dim=y_train.shape[1], device = args.device)
        model.eval()
        logits, outputs, embeds, hidden_embs = model(data.to(args.device))
        embeds = embeds.cpu().detach().numpy()
        
        explainer = ExplainerMOCopy(model=model, args=args)
        explainer.to(args.device)

        hops = len(args.hiddens.split('-'))
        extractor = ExtractorNew(adj, edge_index, features, edge_label_matrix, embeds, all_label, hops)

        allnodes = []
        trainnodes, valnodes, testnodes = [], [], []
        if args.dataset == "BA_shapes":
            trainnodes = torch.where(torch.from_numpy(train_mask) * label != 0)[0].tolist()
            valnodes = torch.where(torch.from_numpy(val_mask) * label != 0)[0].tolist()
            testnodes = torch.where(torch.from_numpy(test_mask) * label != 0)[0].tolist()
        elif args.dataset == "BA_community":
            trainnodes = torch.where(((torch.from_numpy(train_mask) * label != 0) & (torch.from_numpy(train_mask) * label != 4)))[0].tolist()
            valnodes = torch.where(((torch.from_numpy(val_mask) * label != 0) & (torch.from_numpy(val_mask) * label != 4)))[0].tolist()
            testnodes = torch.where(((torch.from_numpy(test_mask) * label != 0) & (torch.from_numpy(test_mask) * label != 4)))[0].tolist()
        allnodes.extend(trainnodes)
        allnodes.extend(valnodes)
        allnodes.extend(testnodes)
        #allnodes = trainnodes = list(range(700))
        sub_support_tensors = []
        sub_label_tensors = []
        sub_features = []
        sub_embeds = []
        sub_adjs = []
        sub_edge_labels = []
        sub_labels = []
        sub_nodes = []
        sub_outputs = []
        sub_hidden_embs = []
        remap = {}
        #CELL 2
        for node in allnodes:
            sub_adj, sub_feature, sub_embed, sub_label, sub_edge_label_matrix, sub_node, sub_edge_idx = extractor.subgraph(node)
            remap[node] = len(sub_adjs)
            sub_support = preprocess_adj(sub_adj)
            i = torch.LongTensor([*sub_support[0]])
            v = torch.FloatTensor([*sub_support[1]])
            # LET OP: i moet getransposed worden om sparse tensor te maken met pytorch
            sub_support_tensor = torch.sparse_coo_tensor(i.t(), v, torch.Size([*sub_support[2]])).type(torch.float32) 
            sub_label_tensor = torch.Tensor(sub_label).type(torch.float32)
            sub_feature_tensor = torch.Tensor(sub_feature).type(torch.float32)

            sub_adjs.append(sub_adj)
            sub_features.append(sub_feature_tensor)
            sub_labels.append(sub_label)
            sub_edge_labels.append(sub_edge_label_matrix)
            sub_label_tensors.append(sub_label_tensor)
            sub_support_tensors.append(sub_support_tensor)
            sub_nodes.append(sub_node)
            with torch.no_grad():
                #load trained GCN (PGE)
                #sub_output = model((sub_feature,sub_support_tensor), training=False)
                #use trained GCN (survey)
                sub_edge_index = sub_support_tensor._indices()
                data = Data(x=sub_feature_tensor.to(args.device), edge_index=sub_edge_index.to(args.device))
                _, sub_output, sub_embed, sub_hidden_emb = model(data)
            sub_outputs.append(sub_output)
            sub_embeds.append(sub_embed)
            sub_hidden_embs.append(sub_hidden_emb)

        if test_flag:
            f = open(save_map + str(iter) + "/" + "LOG_" + testmodel_filename + "_test.txt", "w")
            savedDict = torch.load(save_map + str(iter) + "/" + testmodel_filename + ".pt") 
            explainer.load_state_dict(savedDict)
            #savedkeys = [key for key in savedDict.keys() if key !='node_feat_mask']
            #newDict = {key:savedDict[key] for key in savedkeys}
            #ckpt = torch.load(ckpt_path)
            '''new_state_dic = OrderedDict()
            for key, value in explainer.state_dict().items():
                if "lin." in key:
                    old_key = key.replace("lin.", "")
                else:
                    old_key = key
                if "gnn_layers" in old_key:
                    new_state_dic[key] = savedDict[old_key].T
                else:
                    new_state_dic[key] = savedDict[old_key]
            explainer.load_state_dict(new_state_dic)'''
        else:
            f = open(save_map + str(iter) + "/" + "LOG_" + args.model_filename + "_BEST.txt", "w")
            tik = time.time()
            if "MO" in optimal_method:
                train_MO(iter, args)
            else:
                train(iter, args)
            tok = time.time()
            f.write("train time,{}".format(tok - tik) + "\n\n")
            explainer.load_state_dict(torch.load(save_map + str(iter) + "/" + args.model_filename +"_BEST.pt"))

        f.write("test result.")
        # metrics
        tik = time.time()
        reals = []
        preds = []
        ndcgs = []
        exp_dict={}
        pred_label_dict={}
        feature_dict = {}
        adj_dict = {}
        e_labels_dict = {}
        allnode_related_preds_dict = dict()
        allnode_mask_dict = dict()
        metric = MaskoutMetric(model, args)
        plotutils = PlotUtils(dataset_name=args.dataset)
        if test_flag:
            f_hidden = open(save_map + str(iter) + "/" +"hidden_emb_test.txt", "w")
        else:
            f_hidden = open(save_map + str(iter) + "/" +"hidden_emb.txt", "w")
        for node in testnodes:
            newid = remap[node]
            sub_adj, sub_feature, sub_embed, sub_label = sub_adjs[newid], sub_features[newid], sub_embeds[newid], sub_labels[newid]
            sub_edge_index = torch.tensor([sub_adj.row, sub_adj.col], dtype=torch.int64)
            nodeid = 0
            #use fullgraph prediction
            #origin_pred = outputs[node].to(args.device)
            #use subgraph prediction
            origin_pred = sub_outputs[newid][nodeid]

            auc_onenode, ndcg_onenode, real, pred = explain_test(explainer, node, origin_pred)
            ndcgs.append(ndcg_onenode)
            reals.extend(real)
            preds.extend(pred)

            pred_mask, related_preds_dict = metric.metric_pg_del_edges(nodeid, explainer, sub_adj, sub_feature, sub_embed, sub_label, origin_pred, sub_nodes[newid])
            allnode_related_preds_dict[node] = related_preds_dict
            allnode_mask_dict[node] = pred_mask

            exp_dict[node] = explainer.masked_adj.detach()
            pred_label_dict[node]=torch.argmax(origin_pred)
            feature_dict[node] = sub_feature
            adj_dict[node] = torch.tensor(sub_adj.todense())
            e_labels_dict[node] = torch.tensor(sub_edge_labels[newid].todense())[sub_edge_index[0], sub_edge_index[1]]

            #record hidden embedding and cosine similarity
            record_hidden_info(explainer, node, loss_type)

            if plot_flag:
                #plot(node, label, iter)
                # visualization
                edge_mask = explainer.masked_adj[sub_adj.row, sub_adj.col].detach()
                edges_idx_desc = edge_mask.reshape(-1).sort(descending=True).indices
                '''important_edges = sub_edge_index[:, edges_idx_desc]
                important_nodes = list(set(important_edges[0].numpy()) | set(important_edges[1].numpy()))
                data = Data(x=sub_feature, edge_index=sub_edge_index, y =sub_label.argmax(-1) )
                ori_graph = to_networkx(data, to_undirected=True)
                plotutils.plot(ori_graph, important_nodes, y=sub_label.argmax(-1), node_idx=nodeid,
                            figname=os.path.join(save_map + str(iter), f"node_{node}.png"))'''
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
        f_hidden.close()

        if len(np.unique(reals))==1 or len(np.unique(preds))==1:
            auc = -1
        else:
            auc = roc_auc_score(reals, preds)
        ndcg = np.mean(ndcgs)
        auc_all.append(auc)
        ndcg_all.append(ndcg)
        tok = time.time()
        f.write("iter,{}".format(iter) + ",auc,{}".format(auc) + ",ndcg,{}".format(ndcg) + ",time,{}".format(tok - tik) + "\n")

        PN = compute_pn_NC(exp_dict, pred_label_dict, args, model, feature_dict, adj_dict)
        PS, ave_size = compute_ps_NC(exp_dict, pred_label_dict, args, model, feature_dict)
        if PN + PS==0:
            FNS=0
        else:
            FNS = 2 * PN * PS / (PN + PS)
        acc_1, pre, rec, f1=0,0,0,0
        if args.dataset == "BA_shapes" or args.dataset == "BA_community":
            acc_1, pre, rec, f1 = compute_precision_recall_NC(exp_dict, args, e_labels_dict, adj_dict)
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
            x_collector = XCollector()
            for node in testnodes:
                related_preds = allnode_related_preds_dict[node][top_k]
                mask = allnode_mask_dict[node]
                x_collector.collect_data(mask, related_preds)
                f.write("node,{}\n".format(node))
                f.write("mask,{}\n".format(mask))
                f.write("related_preds,{}\n".format(related_preds))

            one_simula_arr.append(round(x_collector.simula, 4))
            one_simula_origin_arr.append(round(x_collector.simula_origin, 4))
            one_simula_complete_arr.append(round(x_collector.simula_complete, 4))
            one_fidelity_arr.append(round(x_collector.fidelity, 4))
            one_fidelity_origin_arr.append(round(x_collector.fidelity_origin, 4))
            one_fidelity_complete_arr.append(round(x_collector.fidelity_complete, 4))
            one_fidelityminus_arr.append(round(x_collector.fidelityminus, 4))
            one_fidelityminus_origin_arr.append(round(x_collector.fidelityminus_origin, 4))
            one_fidelityminus_complete_arr.append(round(x_collector.fidelityminus_complete, 4))
            one_finalfidelity_complete_arr.append(round(x_collector.fidelity_complete - x_collector.fidelityminus_complete, 4))
            F_fidelity = 2/(1/x_collector.fidelity_complete +1/(1/x_collector.fidelityminus_complete))
            one_fvaluefidelity_complete_arr.append(round(F_fidelity, 4))
            one_del_fidelity_arr.append(round(x_collector.del_fidelity, 4))
            one_del_fidelity_origin_arr.append(round(x_collector.del_fidelity_origin, 4))
            one_del_fidelity_complete_arr.append(round(x_collector.del_fidelity_complete, 4))
            one_del_fidelityminus_arr.append(round(x_collector.del_fidelityminus, 4))
            one_del_fidelityminus_origin_arr.append(round(x_collector.del_fidelityminus_origin, 4))
            one_del_fidelityminus_complete_arr.append(round(x_collector.del_fidelityminus_complete, 4))
            one_del_finalfidelity_complete_arr.append(round(x_collector.del_fidelity_complete - x_collector.del_fidelityminus_complete, 4))
            del_F_fidelity = 2/(1/x_collector.del_fidelity_complete +1/(1/x_collector.del_fidelityminus_complete))
            one_del_fvaluefidelity_complete_arr.append(round(del_F_fidelity, 4))
            one_sparsity_edges_arr.append(round(x_collector.sparsity_edges, 4))
            one_fidelity_nodes_arr.append(round(x_collector.fidelity_nodes, 4))
            one_fidelity_origin_nodes_arr.append(round(x_collector.fidelity_origin_nodes, 4))
            one_fidelity_complete_nodes_arr.append(round(x_collector.fidelity_complete_nodes, 4))
            one_fidelityminus_nodes_arr.append(round(x_collector.fidelityminus_nodes, 4))
            one_fidelityminus_origin_nodes_arr.append(round(x_collector.fidelityminus_origin_nodes, 4))
            one_fidelityminus_complete_nodes_arr.append(round(x_collector.fidelityminus_complete_nodes, 4))
            one_finalfidelity_complete_nodes_arr.append(round(x_collector.fidelity_complete_nodes - x_collector.fidelityminus_complete_nodes, 4))
            one_sparsity_nodes_arr.append(round(x_collector.sparsity_nodes, 4))

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


def fullgraph_mask(node, explainer, model, features, edge_index, sub_edge_idx, edge_mask):
    full_edge_mask = torch.ones(edge_index.shape[1], dtype=torch.float32).to(args.device) 
    full_edge_mask[sub_edge_idx] = edge_mask
    explainer.eval()
    explainer.__clear_masks__()
    explainer.__set_masks__(features, edge_index, full_edge_mask)    
    data = Data(x=features, edge_index=edge_index)
    model.eval()
    _, mask_preds, mask_emb, mask_h_all= model(data)
    mask_pred = mask_preds[node]
    explainer.__clear_masks__()
    return mask_pred, mask_emb

def subgraph_mask(nodeid, explainer, model, sub_feature, sub_edge_index, edge_mask):
    explainer.eval()
    explainer.__set_masks__(sub_feature, sub_edge_index, edge_mask)    
    data = Data(x=sub_feature, edge_index=sub_edge_index)
    model.eval()
    _, sub_mask_preds, sub_mask_emb, sub_mask_h_all= model(data)
    sub_mask_pred = sub_mask_preds[nodeid]
    explainer.__clear_masks__()
    return sub_mask_pred, sub_mask_emb


def test_explainmodel(testnodes):
    args.dataset = "BA_shapes"
    filename = '/home/liuli/zhangym/torch_projects/datasets/'+args.dataset+'/raw/' + args.dataset + '.pkl'
    GNNmodel_ckpt_path = osp.join('model_weights', args.dataset, 'gcn_best.pth')
    save_map = "MO_LISA_TEST_LOGS/BA_SHAPES_loss_pdiff_CF_LM_iter1_elr0003_epoch1000_MO-GradVac_angle45_dominant_pdiff-1divide-cf/0/"
    explainmodel_ckpt_path = save_map+"BA_shapes_0_BEST.pt"
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #load data
    #with open('./dataset/' + args.dataset + '.pkl', 'rb') as fin:
    with open(filename, 'rb') as fin:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pkl.load(fin)

    #edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
    adj = csr_matrix(adj)
    support = preprocess_adj(adj)

    features = torch.tensor(features).type(torch.float32)
    edge_index = torch.LongTensor([*support[0]]).t().to(args.device)
    edge_data = torch.FloatTensor([*support[1]]).to(args.device)
    # LET OP: i moet getransposed worden om sparse tensor te maken met pytorch
    support_tensor = torch.sparse_coo_tensor(edge_index, edge_data, torch.Size([*support[2]]))
    support_tensor = support_tensor.type(torch.float32)

    all_label = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
    label = np.where(all_label)[1]
    
    #load trained GCN (PGE)
    ''' model = load_GCN_PG(GNNmodel_ckpt_path, input_dim=features.shape[1], output_dim=y_train.shape[1], device=device)
    model.eval()
    embeds = model.embedding((features,support_tensor)).cpu().detach().numpy()'''
    
    y = torch.from_numpy(label).to(args.device)
    data = Data(x=features, y=y, edge_index=edge_index)

    f_embed = open(save_map + "test_embedding_LOG_1.txt", "w")
    #load trained GCN (survey)
    model = load_gnnNets_NC(GNNmodel_ckpt_path, input_dim=features.shape[1], output_dim=y_train.shape[1], device = args.device)
    model.eval()
    logits, outputs, embeds, hidden_embs = model(data.to(args.device))
    embeds = embeds.cpu().detach().numpy()

    explainer = ExplainerMOCopy(model=model, args=args)
    explainer.to(args.device)

    hops = len(args.hiddens.split('-'))
    extractor = ExtractorNew(adj, edge_index, features, edge_label_matrix, embeds, all_label, hops)

    #load explain model
    savedDict = torch.load(explainmodel_ckpt_path) 
    savedkeys = [key for key in savedDict.keys() if key !='node_feat_mask']
    newDict = {key:savedDict[key] for key in savedkeys}
    explainer.load_state_dict(newDict)
    fminus_pred_arr = []
    fplus_pred_arr = []
    fminus_emb_arr = []
    fplus_emb_arr = []
    random_fminus_pred_arr = []
    random_fplus_pred_arr = []
    random_fminus_emb_arr = []
    random_fplus_emb_arr = []
    for node in testnodes:
        sub_adj, sub_feature, sub_embed, sub_label, sub_edge_label_matrix, sub_node, sub_edge_idx = extractor.subgraph(node)
        sub_support = preprocess_adj(sub_adj)
        i = torch.LongTensor([*sub_support[0]])
        v = torch.FloatTensor([*sub_support[1]])
        # LET OP: i moet getransposed worden om sparse tensor te maken met pytorch
        sub_support_tensor = torch.sparse_coo_tensor(i.t(), v, torch.Size([*sub_support[2]])).type(torch.float32) 
        sub_label_tensor = torch.Tensor(sub_label).type(torch.float32)
        sub_feature = torch.Tensor(sub_feature).type(torch.float32).to(args.device)
        sub_edge_index = sub_support_tensor._indices().to(args.device)
        with torch.no_grad():
            #load trained GCN (PGE)
            #sub_output = model((sub_feature,sub_support_tensor), training=False)
            #use trained GCN (survey)
            data = Data(x=sub_feature, edge_index=sub_edge_index)
            _, sub_output, sub_embed, sub_hidden_emb = model(data)

        origin_label = outputs[node].argmax(0)

        explainer.eval()
        nodeid = 0
        masked_pred, cf_pred, masked_emb, masked_hidden_emb = explainer((sub_feature, sub_adj, nodeid, sub_embed, 1.0))
        
        edge_mask = explainer.masked_adj[sub_adj.row, sub_adj.col].detach()
        fminus_pred, fminus_emb = fullgraph_mask(node, explainer, model, features, edge_index, sub_edge_idx, edge_mask)
        fplus_pred, fplus_emb = fullgraph_mask(node, explainer, model, features, edge_index, sub_edge_idx, 1-edge_mask)

        random_edge_mask = torch.rand(edge_mask.shape[0]).to(args.device)
        random_fminus_pred, random_fminus_emb = fullgraph_mask(node, explainer, model, features, edge_index, sub_edge_idx, random_edge_mask)
        random_fplus_pred, random_fplus_emb = fullgraph_mask(node, explainer, model, features, edge_index, sub_edge_idx, 1-random_edge_mask)

        top_k = 20
        select_k = round(top_k/100 * len(sub_edge_index[0]))

        selected_impedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[:select_k]        #按比例选择top_k%的重要边
        other_notimpedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[select_k:]        
        sparsity_edges = 1- len(selected_impedges_idx) / sub_edge_index.shape[1]

        #mask edges
        maskimp_edge_mask = torch.ones(len(edge_mask), dtype=torch.float32).to(args.device) 
        maskimp_edge_mask[selected_impedges_idx] = 1-edge_mask[selected_impedges_idx]      #重要的top_k%边置为1-mask
        maskimp_pred, maskimp_emb = fullgraph_mask(node, explainer, model, features, edge_index, sub_edge_idx, maskimp_edge_mask)
        sub_maskimp_pred, sub_maskimp_emb = subgraph_mask(nodeid, explainer, model, sub_feature, sub_edge_index, maskimp_edge_mask)
        
        masknotimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(args.device) 
        masknotimp_edge_mask[other_notimpedges_idx] = edge_mask[other_notimpedges_idx]      #除了重要的top_k%之外的其他边置为mask
        masknotimp_pred, masknotimp_emb = fullgraph_mask(node, explainer, model, features, edge_index, sub_edge_idx, masknotimp_edge_mask)
        sub_masknotimp_pred, sub_masknotimp_emb = subgraph_mask(nodeid, explainer, model, sub_feature, sub_edge_index, masknotimp_edge_mask)

        #traditional fidelity (delete edges)
        delimp_edge_mask = torch.ones(len(edge_mask), dtype=torch.float32).to(args.device) 
        delimp_edge_mask[selected_impedges_idx] = 0.0   #remove important edges
        delimp_pred, delimp_emb = fullgraph_mask(node, explainer, model, features, edge_index, sub_edge_idx, delimp_edge_mask)

        delnotimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(args.device) 
        delnotimp_edge_mask[other_notimpedges_idx] =  0.0   #remove not important edges
        delnotimp_pred, delnotimp_emb = fullgraph_mask(node, explainer, model, features, edge_index, sub_edge_idx, delnotimp_edge_mask)

        print("node",node)
        print("origin_pred=", outputs[node].cpu().detach().numpy())
        print("sub_origin_pred=", sub_output[nodeid].cpu().detach().numpy())
        print("sub_maskimp_pred=",sub_maskimp_pred.cpu().detach().numpy())
        print("maskimp_pred=",maskimp_pred.cpu().detach().numpy())
        print("sub_masknotimp_pred=",sub_masknotimp_pred.cpu().detach().numpy())
        print("masknotimp_pred=",masknotimp_pred.cpu().detach().numpy())
        '''f_embed.write("node={}\n".format(node) )
        f_embed.write("sub_node={}\n".format(sub_node))
        f_embed.write("sub_node_totalnum={}\n".format(len(sub_node)))
        f_embed.write("origin_pred={}\n".format(sub_output[nodeid].cpu().detach().numpy().tolist()) )
        f_embed.write("factual_pred={}\n".format(masked_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("cf_pred={}\n".format(cf_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("sub_maskimp_pred={}\n".format(sub_maskimp_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("sub_masknotimp_pred={}\n".format(sub_masknotimp_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("maskimp_pred={}\n".format(maskimp_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("masknotimp_pred={}\n".format(masknotimp_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("delimp_pred={}\n".format(delimp_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("delnotimp_pred={}\n".format(delnotimp_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("fminus_pred={}\n".format(fminus_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("fplus_pred={}\n".format(fplus_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("random_fminus_pred={}\n".format(random_fminus_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("random_fplus_pred={}\n".format(random_fplus_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("label={}\n".format(label[node]) )
        f_embed.write("origin_label={}\n".format(origin_label.item()) )
        f_embed.write("sub_maskimp_embed={}\n".format(sub_maskimp_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("sub_masknotimp_emb={}\n".format(sub_masknotimp_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("maskimp_embed={}\n".format(maskimp_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("masknotimp_emb={}\n".format(masknotimp_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("delimp_embed={}\n".format(delimp_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("delnotimp_emb={}\n".format(delnotimp_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("fminus_embed={}\n".format(fminus_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("fplus_emb={}\n".format(fplus_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("random_fminus_embed={}\n".format(random_fminus_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("random_fplus_emb={}\n".format(random_fplus_emb.cpu().detach().numpy().tolist()) )'''
        fminus_pred_arr.append(fminus_pred.cpu().detach().numpy().tolist())
        fplus_pred_arr.append(fplus_pred.cpu().detach().numpy().tolist())
        fminus_emb_arr.append(fminus_emb[node].cpu().detach().numpy().tolist())
        fplus_emb_arr.append(fplus_emb[node].cpu().detach().numpy().tolist())
        random_fminus_pred_arr.append(random_fminus_pred.cpu().detach().numpy().tolist())
        random_fplus_pred_arr.append(random_fplus_pred.cpu().detach().numpy().tolist())
        random_fminus_emb_arr.append(random_fminus_emb[node].cpu().detach().numpy().tolist())
        random_fplus_emb_arr.append(random_fplus_emb[node].cpu().detach().numpy().tolist())
    
    f_embed.write("explain_node_id_arr={}\n".format(testnodes) )
    f_embed.write("origin_pred_arr={}\n".format(outputs.tolist()))
    f_embed.write("origin_emb_arr={}\n".format(embeds.tolist()))
    f_embed.write("label_arr={}\n".format(label.tolist()) )
    f_embed.write("origin_label_arr={}\n".format(outputs.argmax(1).cpu().numpy().tolist()) )
    f_embed.write("fminus_pred_arr={}\n".format(fminus_pred_arr) )
    f_embed.write("fplus_pred_arr={}\n".format(fplus_pred_arr) )
    f_embed.write("random_fminus_pred_arr={}\n".format(random_fminus_pred_arr) )
    f_embed.write("random_fplus_pred_arr={}\n".format(random_fplus_pred_arr) )
    f_embed.write("fminus_emb_arr={}\n".format(fminus_emb_arr) )
    f_embed.write("fplus_emb_arr={}\n".format(fplus_emb_arr) )
    f_embed.write("random_fminus_emb_arr={}\n".format(random_fminus_emb_arr) )
    f_embed.write("random_fplus_emb_arr={}\n".format(random_fplus_emb_arr) )
    f_embed.close()




if __name__ == "__main__":
    iteration = 5
    optimal_method_arr = ["MO-GradVac"]  #weightsum, MO-PCGrad, MO-GradVac, MO-CAGrad
    loss_type_arr = ["pdiff_hidden_CF_LM_conn"]    #"ce", "ce_hidden", "kl", "kl_hidden", "pl_value", "pl_value_hidden"
    for optimal_method in optimal_method_arr:
        for loss_type in loss_type_arr: 
            if "weightsum" in optimal_method:
                #coff_arr = [1.0, 5.0, 10.0, 50.0, 100.0]
                coff_arr = [10.0]
                for coff in coff_arr:
                    main(iteration, optimal_method, loss_type, "alllayer", (None,None), None, coff)
            elif "getGrad" in optimal_method:
                main(iteration, optimal_method, loss_type, "alllayer", ("pdiff-CF-mean",[0,2]), None)
            elif "MO" in optimal_method:
                #dominant_loss_dic = {"pdiff-LM-mean": [0,3], "hidden-LM-mean":[1,3]}     #{"KL":0, "hidden":1}  # "PL":0, "value":1, "hidden":2 , "mask":3, "con":4, "CF":5, "KL":0, "hidden":1
                dominant_loss_dic = {"CF": 2}   #None: None, "pdiff-CF-disector": [0,2]
                angle_arr = [45]    #[90, 60, 45, 30]
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

