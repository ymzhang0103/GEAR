import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random
from torch.nn.functional import cosine_similarity
from scipy.optimize import minimize


class Grad():
    def __init__(self, optimizer):
        self._optim = optimizer
        return

    @property
    def optimizer(self):
        return self._optim

    def step(self):
        '''
        update the parameters with the gradient
        '''
        return self._optim.step()
    
    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def _retrieve_grad(self):
        grad, shape = [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
        return grad, shape

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad
    
    def _pack_grad(self, objectives):
        grads, shapes = [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            shapes.append(shape)
        return grads, shapes

    def get_grads(self, objectives):
        grads, shapes = self._pack_grad(objectives)
        coslist = []
        for g_i_idx in range(len(grads)):
            g_i = grads[g_i_idx]
            for g_j_idx in range(g_i_idx+1, len(grads)):
                g_j = grads[g_j_idx]
                cosvalue = torch.nn.CosineSimilarity(dim=0)(g_i, g_j)
                coslist.append(str(g_i_idx) + "-" + str(g_j_idx) + "-" + str(cosvalue.item()))
        return coslist, grads, grads


class MOGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        self.param_groups = optimizer.param_groups
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''
        return self._optim.step()
    
    def get_grads(self, objectives, dominant_idx):
        grads, shapes, has_grads = self._pack_grad(objectives)
        coslist = []
        for g_i_idx in range(len(grads)):
            g_i = grads[g_i_idx]
            for g_j_idx in range(g_i_idx+1, len(grads)):
                g_j = grads[g_j_idx]
                cosvalue = torch.nn.CosineSimilarity(dim=0)(g_i, g_j)
                coslist.append(str(g_i_idx) + "-" + str(g_j_idx) + "-" + str(cosvalue.item()))
        
        if dominant_idx:
            ori_grad = copy.deepcopy(grads)
            dominant_grad = torch.zeros(len(ori_grad[0])).to(ori_grad[0].device)
            if isinstance(dominant_idx, list):
                #F-value of dominant_idx grads
                '''for i in dominant_idx:
                        dominant_grad = dominant_grad + 1/(ori_grad[i] + eps)
                dominant_grad =  torch.true_divide(len(dominant_idx), (dominant_grad+ eps))'''
                domgrads_list = []
                for i in dominant_idx:
                    domgrads_list.append(ori_grad[i])
                dominant_grad =  torch.mean(torch.stack(domgrads_list),dim=0)
            elif dominant_idx == "mean":
                dominant_grad = torch.mean(torch.stack(grads),dim=0)
            else:
                dominant_grad = ori_grad[dominant_idx]
            print("dominant_grad",dominant_grad)
            for idx in range(len(grads)):
                if idx != dominant_idx:
                    g_i = grads[idx]
                    cosvalue = torch.nn.CosineSimilarity(dim=0)(g_i, dominant_grad)
                    coslist.append(str(idx) + "-" + str(cosvalue.item()))
        return coslist, grads, grads
    

    def cagrad_backward(self, objectives, dominant_idx):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''
        #random.shuffle(objectives)
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad, grads_new = self._cagrad(grads, dominant_idx)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return grads, grads_new

    def get_dominant_grad(self, grads, dominant_idx):
        #w_plus = 0.5
        #w_minus = 0.5
        # f = (w_plus + w_minus) / ( w_plus / fidelity_plus + w_minus / (1 - fidelity_minus) )
        #finalfidelity = ( (w_plus + w_minus) * fidelity_plus * (1 - fidelity_minus) ) / (w_plus * (1 - fidelity_minus) + w_minus * fidelity_plus)

        dominant_grad = None
        if isinstance(dominant_idx, list):
            for i in dominant_idx:
                if dominant_grad is None:
                    dominant_grad = 1/grads[:,i]
                else:
                    dominant_grad = dominant_grad + 1/grads[:,i]
            dominant_grad = len(dominant_idx) / dominant_grad
        else:
            dominant_grad = grads[:,dominant_idx]
        return dominant_grad


    def _cagrad(self, grads, dominant_idx, alpha=0.5, rescale=1):
        grads = torch.stack(grads)
        g0 = torch.mean(grads,dim=0)
        g0_norm = torch.norm(g0)
        obj_num = grads.shape[0]
        x_start = np.ones(obj_num) / obj_num
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        c = (alpha*g0_norm+1e-8).item()
        b = x_start.copy()
        A = grads.mm(grads.t()).cpu().numpy()
        def objfn(x):
            return x.reshape(1,obj_num).dot(A).dot(b.reshape(obj_num, 1)) + c * np.sqrt(x.reshape(1,obj_num).dot(A).dot(x.reshape(obj_num,1))+1e-8)
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads.t() * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-8)
        g = g0 + lmbda * gw
        if rescale== 0:
            merged_grad = g
        elif rescale == 1:
            merged_grad = g / (1+alpha**2)
        else:
            merged_grad = g / (1 + alpha)
        return merged_grad, grads


    def _cagrad_bak(self, grads, dominant_idx, alpha=0.5, rescale=1):
        grads = torch.stack(grads).t()
        obj_num = grads.shape[1]
        GG = grads.t().mm(grads).cpu() # [num_tasks, num_tasks]
        #g0_norm = (GG.mean()+1e-8).sqrt() # norm of the average gradient    
        g0 = self.get_dominant_grad(grads, dominant_idx)
        g0_norm = g0.norm()
        x_start = np.ones(obj_num) / obj_num
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha*g0_norm+1e-8).item()
        def objfn(x):
            return (x.reshape(1,obj_num).dot(A).dot(b.reshape(obj_num, 1)) + c * np.sqrt(x.reshape(1,obj_num).dot(A).dot(x.reshape(obj_num,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-8)
        #g = grads.mean(1) + lmbda * gw     #go is average gradient.
        g = g0 + lmbda * gw
        if rescale== 0:
            merged_grad = g
        elif rescale == 1:
            merged_grad = g / (1+alpha**2)
        else:
            merged_grad = g / (1 + alpha)
        return merged_grad, grads


    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad, modify_index_arr, coslist, grads_new = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return modify_index_arr, coslist, grads, grads_new

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        modify_index_arr = []
        #for g_i in pc_grad:
        for g_i_idx in range(len(pc_grad)):
            g_i = pc_grad[g_i_idx]
            #random.shuffle(grads)
            g_idx_arr = list(range(len(grads)))
            random.shuffle(g_idx_arr)
            #for g_j in grads:
            for g_j_idx in g_idx_arr:
                g_j = grads[g_j_idx]
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    modify_index_arr.append(str(g_i_idx) + "-" + str(g_j_idx))
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        coslist = []
        for g_i_idx in range(len(pc_grad)):
            g_i = pc_grad[g_i_idx]
            for g_j_idx in range(g_i_idx+1, len(pc_grad)):
                g_j = pc_grad[g_j_idx]
                cosvalue = torch.nn.CosineSimilarity(dim=0)(g_i, g_j)
                coslist.append(str(g_i_idx) + "-" + str(g_j_idx) + "-" + str(cosvalue.item()))
                '''if cosvalue < 0:
                    print(str(g_i_idx) + "-" + str(g_j_idx) + "-" + str(cosvalue.item()))'''
        return merged_grad, modify_index_arr, coslist, pc_grad


    def pc_backward_dominant(self, objectives, dominant_idx):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad, dominant_index, modify_index_arr, coslist, grads_new = self._project_conflicting_dominant(dominant_idx, grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return dominant_index, modify_index_arr, coslist, grads, grads_new

    def select_dominant_grad(self, grads):
        grad_score = torch.tensor([]).to(grads[0].device)
        for grad_i in grads:
            score = sum([torch.nn.CosineSimilarity(dim=0)(grad_i, grad_j) for grad_j in grads])
            grad_score = torch.cat((grad_score, score.unsqueeze(0)))
        grad_score_sorted, grad_score_index = torch.sort(grad_score, descending=True)
        dominant_grad = grads[grad_score_index[0]]
        return dominant_grad, grad_score_index[0].item()


    #project g_i to the normal plane of dominant grad
    def _project_conflicting_dominant(self, dominant_idx, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        #dominant_grad, dominant_index = self.select_dominant_grad(grads)
        ori_grad = copy.deepcopy(grads)
        dominant_grad, dominant_index = ori_grad[dominant_idx], dominant_idx
        modify_index_arr =  []
        for idx in range(len(grads)):
            g_i = grads[idx]
            g_i_g_j = torch.dot(g_i, dominant_grad)
            if g_i_g_j < 0:
                modify_index_arr.append(idx)
                g_i -= (g_i_g_j) * dominant_grad / (dominant_grad.norm()**2)
        
        coslist = []
        other_grads_idx = [i for i in range(len(grads)) if i != dominant_idx]
        del_grads_idx = []
        for g_i_idx in other_grads_idx:
            g_i = grads[g_i_idx]
            for g_j_idx in [j for j in other_grads_idx if j > g_i_idx]:
                g_j = grads[g_j_idx]
                cosvalue = torch.nn.CosineSimilarity(dim=0)(g_i, g_j)
                coslist.append(str(g_i_idx) + "-" + str(g_j_idx) + "-" + str(cosvalue.item()))
                '''if cosvalue <0:
                    if g_j.norm() < g_i.norm():
                        del_grads_idx.append(g_j_idx)
                    else:
                        del_grads_idx.append(g_i_idx)'''
                    #print(str(g_i_idx) + "-" + str(g_j_idx) + "-" + str(cosvalue.item()))

        '''for del_i in del_grads_idx:
            del grads[del_i]'''
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in grads]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in grads]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in grads]).sum(dim=0)
        return merged_grad, dominant_index, modify_index_arr, coslist, grads


    def pc_backward_dominant_add(self, objectives, dominant_idx):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad, dominant_index, modify_index_arr, coslist = self._project_conflicting_dominant_add(dominant_idx, grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return dominant_index, modify_index_arr, coslist


    #project g_i to the addtion of dominant grad and g_i
    def _project_conflicting_dominant_add(self, dominant_idx, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        ori_grad = copy.deepcopy(grads)
        dominant_grad, dominant_index = ori_grad[dominant_idx], dominant_idx
        modify_index_arr =  []
        for idx in range(len(grads)):
            g_i = grads[idx]
            g_i_g_j = torch.dot(g_i, dominant_grad)
            if g_i_g_j < 0:
                modify_index_arr.append(idx)
                g_add = torch.add(g_i, dominant_grad)
                g_i = torch.dot(g_i, g_add) * g_add / (g_add.norm()**2)
                if torch.dot(g_i, dominant_grad) < 0:
                    print("idx", idx, "cos", torch.nn.CosineSimilarity(dim=0)(g_i, dominant_grad))
                #g_i -= (g_i_g_j) * dominant_grad / (dominant_grad.norm()**2)
        
        coslist = []
        other_grads_idx = [i for i in range(len(grads)) if i != dominant_idx]
        del_grads_idx = []
        for g_i_idx in other_grads_idx:
            g_i = grads[g_i_idx]
            for g_j_idx in [j for j in other_grads_idx if j > g_i_idx]:
                g_j = grads[g_j_idx]
                cosvalue = torch.nn.CosineSimilarity(dim=0)(g_i, g_j)
                coslist.append(str(g_i_idx) + "-" + str(g_j_idx) + "-" + str(cosvalue.item()))
                if cosvalue <0:
                    if g_j.norm() < g_i.norm():
                        del_grads_idx.append(g_j_idx)
                    else:
                        del_grads_idx.append(g_i_idx)
                    print(str(g_i_idx) + "-" + str(g_j_idx) + "-" + str(cosvalue.item()))

        for del_i in del_grads_idx:
            del grads[del_i]
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in grads]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in grads]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in grads]).sum(dim=0)
        return merged_grad, dominant_index, modify_index_arr, coslist


    def pc_backward_gradvac(self, objectives, sim_obj):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad, modify_index_arr, coslist, cur_sim_obj, grads_new = self._project_conflicting_gradvac(grads, has_grads, sim_obj)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return modify_index_arr, coslist, cur_sim_obj, grads, grads_new

    def _project_conflicting_gradvac(self, grads, has_grads, sim_obj=0, beta=0.01, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        modify_index_arr = []
        #for g_i in pc_grad:
        for g_i_idx in range(len(pc_grad)):
            g_i = pc_grad[g_i_idx]
            #random.shuffle(grads)
            g_idx_arr = list(range(len(grads)))
            random.shuffle(g_idx_arr)
            #for g_j in grads:
            for g_j_idx in g_idx_arr:
                g_j = grads[g_j_idx]
                '''g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    modify_index_arr.append(str(g_i_idx) + "-" + str(g_j_idx))
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)'''
                sim_obj_t =  torch.nn.CosineSimilarity(dim=0)(g_i, g_j)
                if sim_obj_t < sim_obj:
                    modify_index_arr.append(str(g_i_idx) + "-" + str(g_j_idx))
                    a2 = g_i.norm() * (sim_obj* torch.sqrt(1-sim_obj_t**2) - sim_obj_t* torch.sqrt(torch.tensor(1.0)-sim_obj**2))
                    a2 /= (g_j.norm() * torch.sqrt(torch.tensor(1.0)-sim_obj**2))
                    g_i += a2* g_j
                #sim_obj = (1-beta)*sim_obj + beta * sim_obj_t

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        coslist = []
        for g_i_idx in range(len(pc_grad)):
            g_i = pc_grad[g_i_idx]
            for g_j_idx in range(g_i_idx+1, len(pc_grad)):
                g_j = pc_grad[g_j_idx]
                cosvalue = torch.nn.CosineSimilarity(dim=0)(g_i, g_j)
                coslist.append(str(g_i_idx) + "-" + str(g_j_idx) + "-" + str(cosvalue.item()))
                '''if cosvalue < 0:
                    print(str(g_i_idx) + "-" + str(g_j_idx) + "-" + str(cosvalue.item()))'''
        return merged_grad, modify_index_arr, coslist, sim_obj, pc_grad


    def pc_backward_gradvac_dominant(self, objectives, dominant_idx, sim_obj):
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad, modify_index_arr, coslist, cur_sim_obj, grads_new = self._project_conflicting_gradvac_dominant(grads, has_grads, dominant_idx, sim_obj)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return modify_index_arr, coslist, cur_sim_obj, grads, grads_new


    def _project_conflicting_gradvac_dominant(self, grads, has_grads, dominant_idx, sim_obj=0, beta=0.01, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        #dominant_grad, dominant_index = self.select_dominant_grad(grads)
        eps = 1e-10
        ori_grad = copy.deepcopy(grads)
        dominant_grad = torch.zeros(len(ori_grad[0])).to(ori_grad[0].device)
        if isinstance(dominant_idx, list):
            #F-value of dominant_idx grads
            '''for i in dominant_idx:
                    dominant_grad = dominant_grad + 1/(ori_grad[i] + eps)
            dominant_grad =  torch.true_divide(len(dominant_idx), (dominant_grad+ eps))'''
            domgrads_list = []
            for i in dominant_idx:
                #domgrads_list.append(ori_grad[i])
                domgrads_list.append(ori_grad[i]/torch.norm(ori_grad[i]))   #normalize
            dominant_grad =  torch.mean(torch.stack(domgrads_list),dim=0)
        elif dominant_idx == "mean":
            dominant_grad = torch.mean(torch.stack(grads),dim=0)
        else:
            dominant_grad = ori_grad[dominant_idx]
        print("dominant_grad",dominant_grad)
        coslist = []
        other_grads_idx = [i for i in range(len(grads)) if i != dominant_idx]
        for g_i_idx in other_grads_idx:
            g_i = grads[g_i_idx]
            cosvalue = torch.nn.CosineSimilarity(dim=0)(g_i, dominant_grad)
            if g_i_idx in [0,2] and cosvalue<0:
                print(cosvalue)
            coslist.append("s-"+str(g_i_idx) + "-" + str(cosvalue.item()))
        
        modify_index_arr =  []
        for idx in range(len(grads)):
            if idx != dominant_idx:
                g_i = grads[idx]
                sim_obj_t =  torch.nn.CosineSimilarity(dim=0)(g_i, dominant_grad)
                if sim_obj_t < sim_obj:
                    modify_index_arr.append(idx)
                    a2 = g_i.norm() * (sim_obj* torch.sqrt(1-sim_obj_t**2) - sim_obj_t* torch.sqrt(torch.tensor(1.0)-sim_obj**2))
                    a2 /= ((dominant_grad.norm()+eps) * torch.sqrt(torch.tensor(1.0)-sim_obj**2))
                    g_i += a2* dominant_grad
                #sim_obj = (1-beta)*sim_obj + beta * sim_obj_t

        other_grads_idx = [i for i in range(len(grads)) if i != dominant_idx]
        for g_i_idx in other_grads_idx:
            g_i = grads[g_i_idx]
            cosvalue = torch.nn.CosineSimilarity(dim=0)(g_i, dominant_grad)
            coslist.append("t-"+str(g_i_idx) + "-" + str(cosvalue.item()))
            for g_j_idx in [j for j in other_grads_idx if j > g_i_idx]:
                g_j = grads[g_j_idx]
                cosvalue = torch.nn.CosineSimilarity(dim=0)(g_i, g_j)
                coslist.append("e-"+str(g_i_idx) + "-" + str(g_j_idx) + "-" + str(cosvalue.item()))

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in grads]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in grads]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in grads]).sum(dim=0)
        return merged_grad, modify_index_arr, coslist, sim_obj, grads


    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            #obj.backward(retain_graph=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad



class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


if __name__ == '__main__':

    # fully shared network test
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = TestNet()
    y_pred = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

    print('-' * 80)
    # seperated shared network test

    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2 = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.MSELoss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

