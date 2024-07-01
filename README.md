# GEAR
Code for the paper "GEAR: Learning Graph Neural Network Explainer via Adjusting Gradients"

## Overview
The project contains the following folders and files.
- datasets:  the datasets for the node classification is contained in this fold.
- codes
	- load_dataset.py: contains functions for loading datasets.
  - train_GNNNets.py: the code for the GNN model to be explained.
	- Configures.py:  parameter configuration of the GNN model to be explained.
  - fornode
	  - ExplainerNCMO.py: the explainer for the node classification task.
	  - config.py: parameter configuration of the GNN explainer for the node classification task.
	  - metricsHidden.py:  metrics of the evaluation for the node classification task.
  - forgraph
	  - ExplainerGCMO.py: the explainer for the graph classification task.
	  - config.py: parameter configuration of the GNN explainer for the graph classification task.
	  - metricsHidden.py:  metrics of the evaluation for the graph classification task.
- main_NCExplain.py: the code for the explainer in the node classification task.
- main_GCExplain.py: the code for the explainer in the graph classification task.

## Prerequisites
python   3.9
torch >= 1.12.1+cu113

## To run
- Train GNN Model
	- Run codes/train_GNNNets.py to train the GNN model. Change parameter **dataset** per demand.
- Train GEAR Model
	- Run NC-Explain_MO.py to train and test the explainer in the node classification task. Change **dataset** per demand.
	- Run GC-Explain_MO.py to train and test the explainer in the graph classification. task Change **dataset** per demand.
- Integrate GEAR into existing GNN frameworks
  	- Initializing MO optimizer
  	  	- optimizer = Adam(explainer.elayers.parameters(), lr=args.elr)  #Line 71 in NC-Explain_MO.py
		- optimizer = MOGrad(optimizer)  #Line 72 in NC-Explain_MO.py
  	- Packaging losses list
  	  	- losses = [pred_loss, hidden_loss, cf_loss, lap_loss, con_loss]  #Line 183 in NC-Explain_MO.py
  	- Adjusting Gradients
  	  	- optimizer.backward_adjust_grad_dominant(losses, dominant_index, sim_obj)  #Line 200 in NC-Explain_MO.py
