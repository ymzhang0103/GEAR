#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import sys
sys.path.append('./codes/forgraph/')
from codes.forgraph.config import args
from sklearn.metrics import roc_auc_score
from codes.forgraph.models import GCN2 as GCN
from codes.forgraph.metricsHidden import *
from codes.forgraph.ExplainerGCMO import ExplainerGCMO
from codes.forgraph.utils import *
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch 
import torch.optim
from torch.optim import Adam
import time
from torch_geometric.data import Data
import os.path as osp
from torch_geometric.utils import to_networkx
from codes.mograd import MOGrad
from codes.load_GNNNets_hidden import load_gnnNets_GC
import math
import operator

def main(iteration_num, optimal_method, loss_type, hidden_layer, dominant_loss, angle, coff=None):
    skip = 5
    topk = 5

    def acc(sub_adj, sub_edge_label):
        mask = explainer.masked_adj.cpu().detach().numpy()
        real = []
        pred = []
        sub_adj = coo_matrix(sub_adj)
        sub_edge_label = sub_edge_label.todense()
        for r,c in list(zip(sub_adj.row, sub_adj.col)):
            d = sub_edge_label[r,c] + sub_edge_label[c,r]
            if d == 0:
                real.append(0)
            else:
                real.append(1)
            pred.append(mask[r][c]+mask[c][r])

        if len(np.unique(real))==1 or len(np.unique(pred))==1:
            return -1, [], []
        return roc_auc_score(real, pred), real, pred


    def test(iteration, indices, model, explainer, topk_arr, plot_flag=False):
        #global preds
        #global reals
        preds = []
        reals = []
        ndcgs =[]
        exp_dict={}
        pred_label_dict={}
        plotutils = PlotUtils(dataset_name=args.dataset)
        metric = MaskoutMetric(model, args)
        allnode_related_preds_dict = dict()
        allnode_mask_dict = dict()
        classify_acc = 0
        for graphid in indices:
            sub_features = dataset.data.x[dataset.slices['x'][graphid].item():dataset.slices['x'][graphid+1].item(), :]
            sub_edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][graphid].item():dataset.slices['edge_index'][graphid+1].item()]
            data = Data(x=sub_features, edge_index=sub_edge_index, batch = torch.zeros(sub_features.shape[0], dtype=torch.int64, device=sub_features.device))
            logits, prob, sub_embs, _, h_all = model(data)
            label = dataset.data.y[graphid]
            sub_adj = torch.sparse_coo_tensor(indices=sub_edge_index, values=torch.ones(sub_edge_index.shape[1]), size=(sub_features.shape[0], sub_features.shape[0])).to_dense().to(sub_edge_index.device)
            explainer.eval()
            masked_pred, cf_pred, masked_emb, masked_hidden_embs = explainer((sub_features, sub_embs, sub_adj, 1.0, label))
            '''if args.dataset == "MUTAG":
                insert = 20
                real, pred = acc(sub_adj, insert)
                reals.extend(real)
                preds.extend(pred)'''
            if args.dataset == "Mutagenicity" or args.dataset == "Mutagenicity_full":
                #sub_edge_label = dataset.data.edge_label[dataset.slices['edge_label'][graphid].item() : dataset.slices['edge_label'][graphid+1].item()]
                sub_edge_gt = dataset.data.edge_label_gt[dataset.slices['edge_label_gt'][graphid].item() : dataset.slices['edge_label_gt'][graphid+1].item()]
                sub_edge_gt_matrix= coo_matrix((sub_edge_gt,(sub_edge_index[0],sub_edge_index[1])),shape=(sub_features.shape[0],sub_features.shape[0]))
                #real, pred = acc(sub_adj, sub_edge_gt_matrix)
                auc_onegraph, real, pred = acc(sub_adj, sub_edge_gt_matrix)
                reals.extend(real)
                preds.extend(pred)
            origin_pred = prob.squeeze()
            ndcg_onegraph, r_mask, _, _ = rho_ndcg(origin_pred, masked_pred, len(masked_pred))
            ndcgs.append(ndcg_onegraph)

            mask = explainer.masked_adj
            pred_mask, related_preds_dict = metric.metric_del_edges_GC(topk_arr, sub_features, mask, sub_edge_index, origin_pred, masked_pred, label)
            if related_preds_dict[10][0]["origin_label"] == torch.argmax(related_preds_dict[10][0]["masked"]):
                classify_acc +=1
            allnode_related_preds_dict[graphid] = related_preds_dict
            allnode_mask_dict[graphid] = pred_mask
            
            exp_dict[graphid] = mask.detach()
            origin_label = torch.argmax(origin_pred)
            pred_label_dict[graphid]=origin_label

            if plot_flag:
                if args.dataset =="Mutagenicity" or args.dataset =="Mutagenicity_full" or args.dataset =="NCI1":
                    visual_imp_edge_count = 8
                else:
                    visual_imp_edge_count = 12
                #plot(sub_adj, label, graphid, iteration)
                edge_mask = mask[sub_edge_index[0], sub_edge_index[1]]
                edges_idx_desc = edge_mask.reshape(-1).sort(descending=True).indices
                #important_edges = sub_edge_index[:, edges_idx_desc]
                #important_nodes = list(set(important_edges[0].numpy()) | set(important_edges[1].numpy()))
                important_nodelist = []
                important_edgelist = []
                for idx in edges_idx_desc:
                    if len(important_edgelist)<visual_imp_edge_count:
                        if (sub_edge_index[0][idx].item(), sub_edge_index[1][idx].item()) not in important_edgelist:
                            important_nodelist.append(sub_edge_index[0][idx].item())
                            important_nodelist.append(sub_edge_index[1][idx].item())
                            important_edgelist.append((sub_edge_index[0][idx].item(), sub_edge_index[1][idx].item()))
                            important_edgelist.append((sub_edge_index[1][idx].item(), sub_edge_index[0][idx].item()))
                important_nodelist = list(set(important_nodelist))
                ori_graph = to_networkx(data, to_undirected=True)
                if hasattr(dataset, 'supplement'):
                    words = dataset.supplement['sentence_tokens'][str(graphid)]
                    if '$' in words:
                        continue
                    plotutils.plot_new(ori_graph, nodelist=important_nodelist, edgelist=important_edgelist, words=words,
                                figname=os.path.join(save_map + str(iteration), f"example_{graphid}_4.pdf"))
                else:
                    plotutils.plot_new(ori_graph, nodelist=important_nodelist, edgelist=important_edgelist, x=data.x,
                                figname=os.path.join(save_map + str(iteration), f"example_{graphid}_4.pdf"))

        if args.dataset == "Mutagenicity" or args.dataset =="Mutagenicity_full":  #args.dataset=="MUTAG":
            if len(np.unique(reals))<=1 or len(np.unique(preds))<=1:
                auc = -1
            else:
                auc = roc_auc_score(reals,preds)
        else:
            auc = 0
        ndcg = np.mean(ndcgs)
        classify_acc = classify_acc/len(indices)
        return auc, ndcg, classify_acc, allnode_related_preds_dict, allnode_mask_dict, exp_dict, pred_label_dict


    def train_MO(iteration):
        tik = time.time()
        epochs = args.eepochs
        t0 = args.coff_t0
        t1 = args.coff_te
        clip_value_min = -2.0
        clip_value_max = 2.0
        best_auc = 0
        explainer.train()

        best_auc = 0
        best_loss = 0
        best_ndcg = 0
        best_decline = 0
        best_F_fidelity =0
        best_del_F_fidelity = 0
        f_train = open(save_map + str(iteration) + "/" + "train_LOG_" + args.model_filename + "_BEST.txt", "w")
        
        optimizer = Adam(explainer.elayers.parameters(), lr=args.elr)
        optimizer = MOGrad(optimizer)
        #sim_obj = 0.0
        sim_obj = round( math.cos(math.radians(angle)), 3)
        dominant_index = dominant_loss[1]
        modify_index_arr = None
        coslist = None
        for epoch in range(epochs):
            loss = torch.tensor(0.0)
            pred_loss = 0
            mask_loss = 0
            con_loss = 0
            sort_loss = 0
            value_loss = 0
            hidden_loss = 0
            cf_loss = 0
            l, pl, ml, cl, hl, cfl = 0,0,0,0,0,0
            losses = []
            tmp = float(t0 * np.power(t1 / t0, epoch /epochs))
            #train_instances = [ins for ins in range(adjs.shape[0])]
            #np.random.shuffle(train_instances)
            for graphid in train_instances:
                sub_features = dataset.data.x[dataset.slices['x'][graphid].item():dataset.slices['x'][graphid+1].item(), :]
                sub_edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][graphid].item():dataset.slices['edge_index'][graphid+1].item()]
                data = Data(x=sub_features, edge_index=sub_edge_index, batch = torch.zeros(sub_features.shape[0], dtype=torch.int64, device=sub_features.device))
                logits, prob, sub_embs, graph_emb, hidden_embs = model(data)
                label = dataset.data.y[graphid]
                #pred_label = torch.argmax(prob.squeeze())
                sub_adj = torch.sparse_coo_tensor(indices=sub_edge_index, values=torch.ones(sub_edge_index.shape[1]), size=(sub_features.shape[0], sub_features.shape[0])).to_dense().to(sub_edge_index.device)
                new_pred, cf_pred, new_graph_emb, new_hidden_embs = explainer((sub_features, sub_embs, sub_adj, tmp, label))
                if "ce" in args.loss_flag:
                    l, pl, ml, cl, hl  = explainer.loss_ce_hidden(new_pred, label, new_hidden_embs, hidden_embs)
                    cfl = 0
                elif "kl" in args.loss_flag:
                    l, pl, ml, cl, hl = explainer.loss_kl_hidden(new_pred, prob.squeeze(), new_hidden_embs, hidden_embs)
                    cfl = 0
                elif "pdiff" in args.loss_flag:
                    l, pl, ml, cl, hl, cfl = explainer.loss_pdiff_hidden(new_pred, prob.squeeze(), cf_pred, new_hidden_embs, hidden_embs)

                loss = loss + l
                pred_loss = pred_loss + pl
                mask_loss = mask_loss + ml
                con_loss = con_loss + cl
                hidden_loss = hidden_loss + hl
                cf_loss = cf_loss + cfl

            train_variables = []
            for name, para in explainer.named_parameters():
                if "elayers" in name:
                    train_variables.append(para)

            optimizer.zero_grad()
            #MOGrad
            if args.loss_flag == "ce":
                losses = [pred_loss, mask_loss, con_loss]
            elif args.loss_flag == "hidden":
                losses = [hidden_loss, mask_loss, con_loss]   
            elif  args.loss_flag == "ce_hidden":
                losses = [pred_loss, hidden_loss, mask_loss, con_loss]
            elif  args.loss_flag == "kl":
                losses = [pred_loss, mask_loss, con_loss]
            elif  args.loss_flag == "kl_hidden":
                losses = [pred_loss, hidden_loss, mask_loss, con_loss]
            elif  args.loss_flag == "kl_hidden_cf":
                losses = [pred_loss, hidden_loss, mask_loss, con_loss, cf_loss]
            elif args.loss_flag == "pl_value":
                losses = [sort_loss, value_loss, mask_loss, con_loss]
            elif args.loss_flag == "pl_value_hidden":
                losses = [sort_loss, value_loss, hidden_loss, mask_loss, con_loss]
            elif args.loss_flag == "pl_value_hidden_cf":
                losses = [sort_loss, value_loss, hidden_loss, mask_loss, con_loss, cf_loss]
            elif args.loss_flag == "pdiff_hidden_CF_LM_conn":
                losses = [pred_loss, hidden_loss, cf_loss, mask_loss, con_loss]
            elif args.loss_flag == "pdiff_CF_LM_conn":
                losses = [pred_loss, cf_loss, mask_loss, con_loss]
            #optimizer.pc_backward(losses)
            if "PCGrad" in optimal_method:
                if dominant_loss[1] is not None:
                    dominant_index, modify_index_arr, coslist, grads, grads_new = optimizer.pc_backward_dominant(losses, dominant_index)
                else:
                    modify_index_arr, coslist, grads, grads_new = optimizer.pc_backward(losses)
            elif "GEAR" in optimal_method:
                if dominant_loss[1] is not None:
                    modify_index_arr, coslist, cur_sim_obj, grads, grads_new= optimizer.backward_adjust_grad_dominant(losses, dominant_index, sim_obj)
                    sim_obj = cur_sim_obj
                else:
                    modify_index_arr, coslist, cur_sim_obj, grads, grads_new = optimizer.backward_adjust_grad(losses, sim_obj)
                    sim_obj = cur_sim_obj
            #torch.nn.utils.clip_grad_value_(explainer.elayers.parameters(), clip_value_max)
            optimizer.step()

            #eval
            auc, ndcg, classify_acc, allnode_related_preds_dict, allnode_mask_dict, exp_dict, pred_label_dict = test(iteration, eval_indices, model, explainer, [10])

            x_collector = XCollector()
            for graphid in eval_indices:
                related_preds = allnode_related_preds_dict[graphid][10]
                mask = allnode_mask_dict[graphid]
                x_collector.collect_data(mask, related_preds, label=0)

            fidelityplus = x_collector.fidelity_complete
            fidelityminus = x_collector.fidelityminus_complete
            decline = fidelityplus-fidelityminus
            #F_fidelity = 2/(1/fidelityplus +1/(1-fidelityminus))
            F_fidelity = 2/(operator.truediv(1, fidelityplus) +1/operator.truediv(1, fidelityminus))

            del_fidelityplus = x_collector.del_fidelity_complete
            del_fidelityminus = x_collector.del_fidelityminus_complete
            #del_F_fidelity = 2/(1/del_fidelityplus +1/(1-del_fidelityminus))
            del_F_fidelity = 2/(operator.truediv(1, del_fidelityplus) +1/operator.truediv(1, del_fidelityminus))

            if epoch == 0:
                best_loss = loss
                best_decline = decline
                best_fidelity = fidelityplus
            if auc >= best_auc:
                print("saving best auc model...")
                f_train.write("saving best auc model...\n")
                best_auc = auc
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +'_BEST_auc.pt')
                torch.save(explainer.state_dict(), save_map + str(iteration) + "/" + args.model_filename +'_BEST_auc.pt')
            if decline >= best_decline:
                print("saving best decline model...")
                f_train.write("saving best decline model...\n")
                best_decline = decline
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +'_BEST_decline.pt')
                torch.save(explainer.state_dict(), save_map + str(iteration) + "/" + args.model_filename +'_BEST_decline.pt')
            if F_fidelity >= best_F_fidelity:
                print("saving best F_fidelity model...")
                f_train.write("saving best F_fidelity model...\n")
                best_F_fidelity = F_fidelity
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +'_BEST.pt')
                torch.save(explainer.state_dict(), save_map + str(iteration) + "/" + args.model_filename +'_BEST.pt')
            if del_F_fidelity >= best_del_F_fidelity:
                print("saving best del_F_fidelity model...")
                f_train.write("saving best del_F_fidelity model...\n")
                best_del_F_fidelity = del_F_fidelity
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +'_BEST_delF.pt')
                torch.save(explainer.state_dict(), save_map + str(iteration) + "/" + args.model_filename +'_BEST_delF.pt')
            
            print("epoch", epoch, "loss", loss, "hidden_loss",hidden_loss, "pred_loss",pred_loss, "mask_loss",mask_loss, "con_loss",con_loss, "sort_loss",sort_loss, "value_loss",value_loss, "cf_loss", cf_loss, "ndcg", ndcg, "auc", auc, "decline",decline, "F_fidelity",F_fidelity, "classify_acc",classify_acc)
            f_train.write("epoch,{}".format(epoch) + ",loss,{}".format(loss) + ",ndcg,{}".format(ndcg) + ",auc,{}".format(auc)+ ",2,{}".format(fidelityplus) + ",1,{}".format(fidelityminus)  + ",decline,{}".format(decline)  + ",hidden_loss,{}".format(hidden_loss) + ",pred_loss,{}".format(pred_loss) + ",size_loss,{}".format(mask_loss)+ ",con_loss,{}".format(con_loss)+ ",sort_loss,{}".format(sort_loss) + ",value_loss,{}".format(value_loss) + ",cf_loss,{}".format(cf_loss)+ ",F_fidelity,{}".format(F_fidelity)+",classify_acc,{}".format(classify_acc))
            if dominant_index is not None:
                dominant_index_str = ""
                if isinstance(dominant_index, list):
                    for i in dominant_index:
                        if dominant_index_str == "":
                            dominant_index_str = str(i)
                        else:
                            dominant_index_str = dominant_index_str+"+"+str(i)
                print("dominant_index:", dominant_index)
                f_train.write(",dominant_index,{}".format(dominant_index_str))
            if modify_index_arr is not None:
                print("modify_index_arr:", modify_index_arr)
                f_train.write(",modify_index,{}".format("_".join([str(midx) for midx in modify_index_arr])) )
            if coslist is not None:
                print("cosvalue:", coslist)
                f_train.write(",cosvalue,{}".format("_".join(coslist)) +  "\n")
            if dominant_index is None or modify_index_arr is None or coslist is None:
                f_train.write("\n")
            
        tok = time.time()
        f_train.write("train time,{}".format(tok - tik) + "\n")
        f_train.close()


    def train(iteration):
        tik = time.time()
        epochs = args.eepochs
        t0 = args.coff_t0
        t1 = args.coff_te
        clip_value_min = -2.0
        clip_value_max = 2.0
        best_auc = 0
        explainer.train()

        best_auc = 0
        best_loss = 0
        best_ndcg = 0
        best_decline = 0
        best_F_fidelity =0
        best_del_F_fidelity = 0
        f_train = open(save_map + str(iteration) + "/" + "train_LOG_" + args.model_filename + "_BEST.txt", "w")
        losses = []
        coslist = None
        dominant_index=None
        optimizer = Adam(explainer.elayers.parameters(), lr=args.elr)
        for epoch in range(epochs):
            loss = torch.tensor(0.0)
            pred_loss = 0
            mask_loss = 0
            con_loss = 0
            sort_loss = 0
            value_loss = 0
            hidden_loss = 0
            cf_loss = 0
            l, pl, ml, cl, hl, cfl = 0,0,0,0,0,0
            tmp = float(t0 * np.power(t1 / t0, epoch /epochs))
            #train_instances = [ins for ins in range(adjs.shape[0])]
            #np.random.shuffle(train_instances)
            for graphid in train_instances:
                sub_features = dataset.data.x[dataset.slices['x'][graphid].item():dataset.slices['x'][graphid+1].item(), :]
                sub_edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][graphid].item():dataset.slices['edge_index'][graphid+1].item()]
                data = Data(x=sub_features, edge_index=sub_edge_index, batch = torch.zeros(sub_features.shape[0], dtype=torch.int64, device=sub_features.device))
                logits, prob, sub_embs, graph_emb, hidden_embs = model(data)
                label = dataset.data.y[graphid]
                #pred_label = torch.argmax(prob.squeeze())
                sub_adj = torch.sparse_coo_tensor(indices=sub_edge_index, values=torch.ones(sub_edge_index.shape[1]), size=(sub_features.shape[0], sub_features.shape[0])).to_dense().to(sub_edge_index.device)
                new_pred, cf_pred, new_graph_emb, new_hidden_embs = explainer((sub_features, sub_embs, sub_adj, tmp, label))
                if "ce" in args.loss_flag:
                    l, pl, ml, cl, hl  = explainer.loss_ce_hidden(new_pred, label, new_hidden_embs, hidden_embs)
                    cfl = 0
                elif "kl" in args.loss_flag:
                    l, pl, ml, cl, hl = explainer.loss_kl_hidden(new_pred, prob.squeeze(), new_hidden_embs, hidden_embs)
                    cfl = 0
                elif "pdiff" in args.loss_flag:
                    l, pl, ml, cl, hl, cfl = explainer.loss_pdiff_hidden(new_pred, prob.squeeze(), cf_pred, new_hidden_embs, hidden_embs)

                loss = loss + l
                pred_loss = pred_loss + pl
                mask_loss = mask_loss + ml
                con_loss = con_loss + cl
                hidden_loss = hidden_loss + hl
                cf_loss = cf_loss + cfl

            train_variables = []
            for name, para in explainer.named_parameters():
                if "elayers" in name:
                    train_variables.append(para)

            if optimal_method=="getGrad":
                if args.loss_flag == "pdiff_hidden_CF_LM_conn":
                    losses = [pred_loss, hidden_loss, cf_loss, mask_loss, con_loss]
                elif args.loss_flag == "pdiff_CF_LM_conn":
                    losses = [pred_loss, cf_loss, mask_loss, con_loss]
                optimizer1 = MOGrad(optimizer)
                dominant_index = dominant_loss[1]
                coslist, grads, grads_new  = optimizer1.get_grads(losses, dominant_index)

            optimizer.zero_grad()
            loss.backward()   #original
            #torch.nn.utils.clip_grad_value_(explainer.elayers.parameters(), clip_value_max)
            optimizer.step()
            #eval
            auc, ndcg, classify_acc, allnode_related_preds_dict, allnode_mask_dict, exp_dict, pred_label_dict = test(iteration, eval_indices, model, explainer, [10])

            x_collector = XCollector()
            for graphid in eval_indices:
                related_preds = allnode_related_preds_dict[graphid][10]
                mask = allnode_mask_dict[graphid]
                x_collector.collect_data(mask, related_preds, label=0)

            fidelityplus = x_collector.fidelity_complete
            fidelityminus = x_collector.fidelityminus_complete
            decline = fidelityplus-fidelityminus
            #F_fidelity = 2/(1/fidelityplus +1/(1-fidelityminus))
            F_fidelity = 2/(operator.truediv(1, fidelityplus) +1/operator.truediv(1, fidelityminus))

            del_fidelityplus = x_collector.del_fidelity_complete
            del_fidelityminus = x_collector.del_fidelityminus_complete
            #del_F_fidelity = 2/(1/del_fidelityplus +1/(1-del_fidelityminus))
            del_F_fidelity = 2/(operator.truediv(1, del_fidelityplus) +1/operator.truediv(1, del_fidelityminus))

            if epoch == 0:
                best_loss = loss
                best_decline = decline
                best_fidelity = fidelityplus
            if auc >= best_auc:
                print("saving best auc model...")
                f_train.write("saving best auc model...\n")
                best_auc = auc
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +'_BEST_auc.pt')
                torch.save(explainer.state_dict(), save_map + str(iteration) + "/" + args.model_filename +'_BEST_auc.pt')
            if decline >= best_decline:
                print("saving best decline model...")
                f_train.write("saving best decline model...\n")
                best_decline = decline
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +'_BEST_decline.pt')
                torch.save(explainer.state_dict(), save_map + str(iteration) + "/" + args.model_filename +'_BEST_decline.pt')
            if F_fidelity >= best_F_fidelity:
                print("saving best F_fidelity model...")
                f_train.write("saving best F_fidelity model...\n")
                best_F_fidelity = F_fidelity
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +'_BEST.pt')
                torch.save(explainer.state_dict(), save_map + str(iteration) + "/" + args.model_filename +'_BEST.pt')
            if del_F_fidelity >= best_del_F_fidelity:
                print("saving best del_F_fidelity model...")
                f_train.write("saving best del_F_fidelity model...\n")
                best_del_F_fidelity = del_F_fidelity
                torch.save(explainer.state_dict(), f'model_weights/' + args.model_filename +'_BEST_delF.pt')
                torch.save(explainer.state_dict(), save_map + str(iteration) + "/" + args.model_filename +'_BEST_delF.pt')
            
            print("epoch", epoch, "loss", loss, "hidden_loss",hidden_loss, "pred_loss",pred_loss, "mask_loss",mask_loss, "con_loss",con_loss, "sort_loss",sort_loss, "value_loss",value_loss, "cf_loss", cf_loss, "ndcg", ndcg, "auc", auc, "decline",decline, "F_fidelity",F_fidelity, "classify_acc",classify_acc)
            f_train.write("epoch,{}".format(epoch) + ",loss,{}".format(loss) + ",ndcg,{}".format(ndcg) + ",auc,{}".format(auc)+ ",2,{}".format(fidelityplus) + ",1,{}".format(fidelityminus)  + ",decline,{}".format(decline)  + ",hidden_loss,{}".format(hidden_loss) + ",pred_loss,{}".format(pred_loss) + ",size_loss,{}".format(mask_loss)+ ",con_loss,{}".format(con_loss)+ ",sort_loss,{}".format(sort_loss) + ",value_loss,{}".format(value_loss) + ",cf_loss,{}".format(cf_loss)+ ",F_fidelity,{}".format(F_fidelity)+",classify_acc,{}".format(classify_acc))
            if dominant_index is not None:
                dominant_index_str = ""
                if isinstance(dominant_index, list):
                    for i in dominant_index:
                        if dominant_index_str == "":
                            dominant_index_str = str(i)
                        else:
                            dominant_index_str = dominant_index_str+"+"+str(i)
                else:
                    dominant_index_str = str(dominant_index)
                print("dominant_index:", dominant_index)
                f_train.write(",dominant_index,{}".format(dominant_index_str))
            if coslist is not None:
                print("cosvalue:", coslist)
                f_train.write(",cosvalue,{}".format("_".join(coslist)) +  "\n")
            else:
                f_train.write( "\n")
        tok = time.time()
        f_train.write("train time,{}".format(tok - tik) + "\n")
        f_train.close()






    args.elr = 0.01
    args.eepochs = 100  #30
    args.coff_t0=1.0   #5.0
    args.coff_te=0.05  #2.0
    args.coff_size = 100
    args.coff_ent = 1.0
    args.concat = False
    #args.bn = True
    args.graph_classification = True
    args.batch_size = 128
    args.random_split_flag = True
    args.data_split_ratio =  [0.8, 0.1, 0.1]  #None
    args.seed = 2023
    

    args.dataset_root = "datasets"
    args.dataset = "Mutagenicity"   #Graph_Twitter, Mutagenicity, NCI1
    save_map = "MO_LISA_TEST_LOGS_NEW/"+args.dataset.upper() +"_loss"
    save_map = save_map + "_" + loss_type
    if hidden_layer:
       save_map = save_map +"_hidden_" + hidden_layer
    save_map = save_map + "_iter"+str(iteration_num)+"_elr"+str(args.elr).replace(".","")+"_epoch"+str(args.eepochs)
    if optimal_method == "weightsum":
        args.coff_diff = coff
        args.coff_cf =coff
        save_map = save_map + "_coffdiff"+str(args.coff_diff).replace(".","")+ "_coffcf"+str(args.coff_cf).replace(".","")
    if "MO" in optimal_method or optimal_method=="getGrad":
        args.coff_diff = 1.0
        args.coff_cf =1.0
        #save_map = save_map + "_coffdiff"+str(args.coff_diff).replace(".","")+ "_coffcf"+str(args.coff_cf).replace(".","")
    save_map = save_map + "_"+optimal_method
    if angle:
        save_map = save_map + "_angle"+str(angle)
    if dominant_loss[0]:
         save_map = save_map + "_dominant_"+dominant_loss[0]
    save_map = save_map + "_fiveGCN_seed2023/"
    print("save_map: ", save_map)
    if not os.path.exists(save_map):
        os.makedirs(save_map)

    args.topk_arr = list(range(10))+list(range(10,101,5))
    args.loss_flag = loss_type
    args.hidden_layer = hidden_layer
    args.ema_beta = 0.01
    
    #train
    args.model_filename = args.dataset
    #test
    test_flag = False
    testmodel_filename = args.dataset + '_BEST'
    args.plot_flag=False

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
    for iteration in range(1):
        print("Starting iteration: {}".format(iteration))
        if not os.path.exists(save_map+str(iteration)):
            os.makedirs(save_map+str(iteration))
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        dataset = get_dataset(args.dataset_root, args.dataset)
        dataset.data.x = dataset.data.x.float()
        dataset.data.y = dataset.data.y.squeeze().long()
        #if args.graph_classification:
        dataloader_params = {'batch_size': args.batch_size,
                                'random_split_flag': args.random_split_flag,
                                'data_split_ratio': args.data_split_ratio,
                                'seed': args.seed}
        loader = get_dataloader(dataset, **dataloader_params)
        test_indices = loader['test'].dataset.indices
        train_instances = loader['train'].dataset.indices
        eval_indices = loader['eval'].dataset.indices
        if args.dataset == "Mutagenicity" or args.dataset == "Mutagenicity_full":
            test_indices = [i for i in test_indices if dataset.data.y[i]==0]
            train_instances = [i for i in train_instances if dataset.data.y[i]==0]
            eval_indices = [i for i in eval_indices if dataset.data.y[i]==0]

        #print("train_instances", train_instances)
        print("test_indices", test_indices)
        #print("eval_indices", eval_indices)

        #load trained GCN (survey)
        GNNmodel_ckpt_path = osp.join('GNN_checkpoint', args.dataset+'_'+str(iteration), 'gcn_best.pth') 
        model = load_gnnNets_GC(GNNmodel_ckpt_path, input_dim=dataset.num_node_features, output_dim=dataset.num_classes, device = args.device)
        model.eval()

        #CELL2
        explainer = ExplainerGCMO(model=model, args=args)
        explainer.to(args.device)
        
        # Training
        if test_flag:
            f = open(save_map + str(iteration) + "/LOG_"+testmodel_filename+"_test.txt", "w")
            explainer.load_state_dict(torch.load(save_map + str(iteration) + "/" + testmodel_filename+".pt") )
        else:
            f = open(save_map + str(iteration) + "/" + "LOG.txt", "w")
            tik = time.time()
            if optimal_method == "weightsum" or optimal_method == "getGrad":
                train(iteration)
            elif "MO" in  optimal_method:
                train_MO(iteration)
            tok = time.time()
            f.write("train time,{}".format(tok - tik) + "\n\n")
            explainer.load_state_dict(torch.load(save_map + str(iteration) + "/" + args.model_filename +'_BEST.pt'))

        f.write("test result.")
        # metrics
        tik = time.time()
        auc, ndcg, classify_acc, allnode_related_preds_dict, allnode_mask_dict, exp_dict, pred_label_dict = test(iteration, test_indices, model, explainer, args.topk_arr, plot_flag=args.plot_flag)
        auc_all.append(auc)
        ndcg_all.append(ndcg)

        PN = compute_pn(exp_dict, pred_label_dict, args, model, dataset)
        PS, ave_size = compute_ps(exp_dict, pred_label_dict, args, model, dataset)
        if PN + PS==0:
            FNS=0
        else:
            FNS = 2 * PN * PS / (PN + PS)
        acc_1, pre, rec, f1=0,0,0,0
        if args.dataset == "Mutagenicity" or args.dataset == "Mutagenicity_full":
            acc_1, pre, rec, f1 = compute_precision_recall(exp_dict, args, dataset)
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
            print("top_k: ", top_k)
            x_collector = XCollector()
            for graphid in test_indices:
                related_preds = allnode_related_preds_dict[graphid][top_k]
                mask = allnode_mask_dict[graphid]
                x_collector.collect_data(mask, related_preds, label=0)
                f.write("graphid,{}\n".format(graphid))
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
        f.write("PN: {}, PS: {}, FNS: {}, size:{}\n".format(PN, PS, FNS, ave_size))
        f.write("acc:{}, pre:{}, rec:{}, f1:{}\n".format(acc_1, pre, rec, f1))
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
    print("MO_fidelity_origin_mean =", list(fidelity_origin_mean))
    print("MO_fidelity_complete_mean =", list(fidelity_complete_mean))
    print("MO_fidelityminus_mean =", list(fidelityminus_mean))
    print("MO_fidelityminus_origin_mean =", list(fidelityminus_origin_mean))
    print("MO_fidelityminus_complete_mean =", list(fidelityminus_complete_mean))
    print("MO_finalfidelity_complete_mean =", list(finalfidelity_complete_mean))
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

def graph_mask(explainer, model, features, edge_index, edge_mask):
    explainer.__clear_masks__()
    explainer.__set_masks__(features, edge_index, edge_mask)    
    data = Data(x=features, edge_index=edge_index, batch = torch.zeros(features.shape[0], dtype=torch.int64, device=features.device))
    _, mask_pred, node_embs, graph_emb, h_all = model(data)
    explainer.__clear_masks__()
    return mask_pred.squeeze(0), graph_emb.squeeze(0)

def test_explainmodel():
    args.dataset_root = "datasets"
    args.dataset = "Mutagenicity"
    #args.dataset="Graph-Twitter"
    save_map = "MO_LISA_TEST_LOGS/Mutagenicity_loss_pdiff_hidden_CF_LM_hidden_alllayer_iter1_elr001_epoch100_MO-GEAR_angle45_dominant_pdiff-CF-disector_fiveGCN_seed2023/0/"
    explainModel_ckpt_path = save_map + args.dataset+"_BEST.pt"
    
    top_k = 10
    GNNmodel_ckpt_path = osp.join('model_weights', args.dataset, 'gcn_best.pth') 

    dataset = get_dataset(args.dataset_root, args.dataset)
    model = load_gnnNets_GC(GNNmodel_ckpt_path, input_dim=dataset.num_node_features, output_dim=dataset.num_classes, device = args.device)
    model.eval()

    explainer = ExplainerGCMO(model=model, args=args)
    explainer.to(args.device)
    explainer.load_state_dict(torch.load(explainModel_ckpt_path) )
    explainer.eval()

    explain_graph_arr = range(len(dataset.slices['x'])-1)
    f_embed = open(save_map + "test_embedding_LOG.txt", "w")
    explain_graph_id_arr = []
    origin_pred_arr = []
    fminus_pred_arr = []
    fplus_pred_arr = []
    random_fminus_pred_arr = []
    random_fplus_pred_arr = []
    label_arr = []
    origin_label_arr = []
    origin_emb_arr = []
    fminus_emb_arr = []
    fplus_emb_arr = []
    random_fminus_emb_arr = []
    random_fplus_emb_arr = []
    for explain_graph in explain_graph_arr:
        features = dataset.data.x[dataset.slices['x'][explain_graph].item():dataset.slices['x'][explain_graph+1].item(), :]
        edge_index = dataset.data.edge_index[:, dataset.slices['edge_index'][explain_graph].item():dataset.slices['edge_index'][explain_graph+1].item()]
        data = Data(x=features, edge_index=edge_index, batch = torch.zeros(features.shape[0], dtype=torch.int64, device=features.device))
        logits, prob, node_embs, graph_emb, hidden_embs = model(data)
        label = dataset.data.y[explain_graph]
        adj = torch.sparse_coo_tensor(indices=edge_index, values=torch.ones(edge_index.shape[1]), size=(features.shape[0], features.shape[0])).to_dense().to(edge_index.device)
        masked_pred, cf_pred, masked_emb, masked_hidden_embs = explainer((features, node_embs, adj, 1.0, label))
        
        edge_mask = explainer.masked_adj[edge_index[0], edge_index[1]]
        fminus_pred, fminus_emb = graph_mask(explainer, model, features, edge_index, edge_mask)
        fplus_pred, fplus_emb = graph_mask(explainer, model, features, edge_index, 1-edge_mask)

        random_edge_mask = torch.rand(edge_mask.shape[0]).to(args.device)
        random_fminus_pred, random_fminus_emb = graph_mask(explainer, model, features, edge_index, random_edge_mask)
        random_fplus_pred, random_fplus_emb = graph_mask(explainer, model, features, edge_index, 1-random_edge_mask)

        select_k = round(top_k/100 * len(edge_index[0]))
        selected_impedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[:select_k]        #按比例选择top_k%的重要边
        other_notimpedges_idx = edge_mask.reshape(-1).sort(descending=True).indices[select_k:]  

        maskimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(args.device) 
        maskimp_edge_mask[selected_impedges_idx] = 1-edge_mask[selected_impedges_idx]      #重要的top_k%的边置为1-mask
        maskimp_pred, maskimp_emb = graph_mask(explainer, model, features, edge_index, maskimp_edge_mask)
        
        masknotimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(args.device) 
        masknotimp_edge_mask[other_notimpedges_idx] = edge_mask[other_notimpedges_idx]      #除了重要的top_k%之外的其他边置为mask
        masknotimp_pred, masknotimp_emb = graph_mask(explainer, model, features, edge_index, masknotimp_edge_mask)
        
        delimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(args.device) 
        delimp_edge_mask[selected_impedges_idx] =  0.0   #remove important edges
        delimp_pred, delimp_emb = graph_mask(explainer, model, features, edge_index, delimp_edge_mask)
       
        delnotimp_edge_mask  = torch.ones(len(edge_mask), dtype=torch.float32).to(args.device) 
        delnotimp_edge_mask[other_notimpedges_idx] =  0.0   #remove not important edges
        delnotimp_pred, delnotimp_emb = graph_mask(explainer, model, features, edge_index, delnotimp_edge_mask)

        print("explain_graph",explain_graph)
        f_embed.write("explain_graph_id={}\n".format(explain_graph) )
        f_embed.write("origin_pred={}\n".format(prob.squeeze(0).cpu().detach().numpy().tolist()) )
        f_embed.write("factual_pred={}\n".format(masked_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("cf_pred={}\n".format(cf_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("fminus_pred={}\n".format(fminus_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("fplus_pred={}\n".format(fplus_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("maskimp_pred={}\n".format(maskimp_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("masknotimp_pred={}\n".format(masknotimp_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("delimp_pred={}\n".format(delimp_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("delnotimp_pred={}\n".format(delnotimp_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("random_fminus_pred={}\n".format(random_fminus_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("random_fplus_pred={}\n".format(random_fplus_pred.cpu().detach().numpy().tolist()) )
        f_embed.write("label={}\n".format(label.item()) )
        f_embed.write("origin_label={}\n".format(prob.squeeze(0).argmax(0).item()) )
        f_embed.write("origin_embed={}\n".format(graph_emb.squeeze(0).cpu().detach().numpy().tolist()) )
        f_embed.write("fminus_embed={}\n".format(fminus_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("fplus_emb={}\n".format(fplus_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("random_fminus_embed={}\n".format(random_fminus_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("random_fplus_emb={}\n".format(random_fplus_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("maskimp_embed={}\n".format(maskimp_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("masknotimp_emb={}\n".format(masknotimp_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("delimp_embed={}\n".format(delimp_emb.cpu().detach().numpy().tolist()) )
        f_embed.write("delnotimp_emb={}\n".format(delnotimp_emb.cpu().detach().numpy().tolist()) )
        for i in range(len(hidden_embs)):
            f_embed.write("hidden_emb_layer{}={}\n".format(str(i), hidden_embs[i].cpu().detach().numpy().tolist()) )
        for i in range(len(masked_hidden_embs)):
            f_embed.write("masked_hidden_emb_layer{}={}\n".format(str(i), masked_hidden_embs[i].cpu().detach().numpy().tolist()) )
        explain_graph_id_arr.append(explain_graph)
        origin_pred_arr.append(prob.squeeze(0).cpu().detach().numpy().tolist())
        fminus_pred_arr .append(fminus_pred.cpu().detach().numpy().tolist())
        fplus_pred_arr.append(fplus_pred.cpu().detach().numpy().tolist())
        random_fminus_pred_arr.append(random_fminus_pred.cpu().detach().numpy().tolist())
        random_fplus_pred_arr.append(random_fplus_pred.cpu().detach().numpy().tolist())
        label_arr.append(label.item())
        origin_label_arr.append(prob.squeeze(0).argmax(0).item())
        origin_emb_arr.append(graph_emb.squeeze(0).cpu().detach().numpy().tolist())
        fminus_emb_arr.append(fminus_emb.cpu().detach().numpy().tolist())
        fplus_emb_arr.append(fplus_emb.cpu().detach().numpy().tolist())
        random_fminus_emb_arr.append(random_fminus_emb.cpu().detach().numpy().tolist())
        random_fplus_emb_arr.append(random_fplus_emb.cpu().detach().numpy().tolist())
    f_embed.write("explain_graph_id_arr={}\n".format(explain_graph_id_arr) )
    f_embed.write("origin_pred_arr={}\n".format(origin_pred_arr) )
    f_embed.write("fminus_pred_arr={}\n".format(fminus_pred_arr) )
    f_embed.write("fplus_pred={}\n".format(fplus_pred_arr) )
    f_embed.write("random_fminus_pred={}\n".format(random_fminus_pred_arr) )
    f_embed.write("random_fplus_pred={}\n".format(random_fplus_pred_arr) )
    f_embed.write("label_arr={}\n".format(label_arr) )
    f_embed.write("origin_label_arr={}\n".format(origin_label_arr) )
    f_embed.write("origin_emb_arr={}\n".format(origin_emb_arr) )
    f_embed.write("fminus_emb_arr={}\n".format(fminus_emb_arr) )
    f_embed.write("fplus_emb_arr={}\n".format(fplus_emb_arr) )
    f_embed.write("random_fminus_emb_arr={}\n".format(random_fminus_emb_arr) )
    f_embed.write("random_fplus_emb_arr={}\n".format(random_fplus_emb_arr) )
    f_embed.close()
    


if __name__ == "__main__":
    iteration = 5
    optimal_method_arr = ["MO-GEAR"]  #weightsum, MO-PCGrad, MO-GEAR, MO-CAGrad
    loss_type_arr = ["pdiff_hidden_CF_LM_conn"]    #"ce", "ce_hidden", "kl", "kl_hidden", "pl_value", "pl_value_hidden"
    for optimal_method in optimal_method_arr:
        for loss_type in loss_type_arr: 
            if optimal_method == "weightsum":
                coff_arr =[1.0, 5.0, 10.0, 50.0, 100.0]
                for coff in coff_arr:
                    main(iteration, optimal_method, loss_type, "alllayer", (None,None), None, coff)
            elif  optimal_method == "getGrad":
                main(iteration, optimal_method, loss_type, "alllayer", ("pdiff-CF-mean",[0,2]), None, None)
            elif "MO" in optimal_method:
                #dominant_loss_dic = {"pdiff": 0, "hidden":1, "CF":2, "mask":3, "conn":4}
                dominant_loss_dic = {"pdiff-CF-disector": [0,2]}    #{"pdiff-CF-mean": [0,2]},  "pdiff-CF-disector": [0,2]
                #dominant_loss_dic = {None: None}
                angle_arr = [45] #[90, 75, 60, 45, 30, 15]
                for dominant_loss in dominant_loss_dic.items():
                        for angle in angle_arr:
                            if "hidden" in loss_type:
                                hidden_layer_arr = ["alllayer"]
                                for hidden_layer in hidden_layer_arr:
                                    main(iteration, optimal_method, loss_type, hidden_layer, dominant_loss, angle)
                            else:
                                main(iteration, optimal_method, loss_type, None, dominant_loss, angle)

    #test
    #test_explainmodel()