# GEAR
Code for the paper "GEAR: Learning Graph Neural Network Explainer via Adjusting Gradients"

## Overview
The project contains the following folders and files.
- datasets: the datasets except for the Graph-Twitter dataset are contained in this fold. The Graph-Twitter dataset is stored on a cloud drive.URL: https://pan.baidu.com/s/1vtzVGU2WVD5YdFs1REkG_w?pwd=usis. Access code: usis
- codes
	- load_dataset.py: Load datasets.
  	- train_GNNNets.py: Train GNN models to be explained.
	- Configures.py: Parameter configuration of the GNN model to be explained.
	- fornode
		- ExplainerMO.py: Explainer for the node classification task.
		- ExplainerMO_batch.py: Explainer for the large dataset of the node classification task (Amazon-Computers).
		- config.py: Parameter configuration of the GNN explainer for the node classification task.
	  	- metricsHidden.py: Metrics of the evaluation for the node classification task.
	- forgraph
	  	- ExplainerGCMO.py: Explainer for the graph classification task.
	  	- config.py: Parameter configuration of the GNN explainer for the graph classification task.
	  	- metricsHidden.py: Metrics of the evaluation for the graph classification task.
- GNN_checkpoint: To facilitate the reproduction of the experimental results in the paper, we provide the trained GNNs to be explainer in this fold.

## Prerequisites
- python >= 3.9
- torch >= 1.12.1+cu113
- torch-geometric >= 2.2.0

## To run
- Train GNN Model
	- Run codes/train_GNNNets.py to train the GNNs to be explained. Change parameter **dataset** per demand.
- Train GEAR Model
	- Run NC-Explain_MO.py to train and test the explainer on small datasets of the node classification task. Change **parameters** per demand.
	- Run NC-Explain_MO_batch.py to train and test the explainer large datasets of the node classification task(Amazon-Computers). Change **parameters** per demand.
	- Run GC-Explain_MO.py to train and test the explainer in the graph classification task. Change **parameters** per demand.
- Integrate GEAR into existing GNN frameworks
  	- Initializing MO optimizer
  	  	- optimizer = Adam(explainer.elayers.parameters(), lr=args.elr)  #Line 71 in NC-Explain_MO.py
		- optimizer = MOGrad(optimizer)  #Line 72 in NC-Explain_MO.py
  	- Packaging losses list
  	  	- losses = [pred_loss, hidden_loss, cf_loss, lap_loss, con_loss]  #Line 183 in NC-Explain_MO.py
  	- Adjusting Gradients
  	  	- optimizer.backward_adjust_grad_dominant(losses, dominant_index, sim_obj)  #Line 200 in NC-Explain_MO.py
