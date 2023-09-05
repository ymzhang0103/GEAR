import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import shutil
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from GNNmodelsHidden import GnnNets, GnnNets_NC
from load_dataset import get_dataset, get_dataloader
from Configures import data_args, train_args, model_args


# train for graph classification
def train_GC():
    # attention the multi-task here
    print('start loading data====================')
    dataset = get_dataset(data_args)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)
    dataloader = get_dataloader(dataset, data_args, train_args)
    
    print('start training model==================')
    gnnNets = GnnNets(input_dim, output_dim, model_args)
    gnnNets.to_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]
    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index/2 :.4f}")

    best_acc = 0.0
    data_size = len(dataset)
    print(f'The total num of dataset is {data_size}')

    # save path for model
    '''if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', data_args.dataset_name)):
        os.mkdir(os.path.join('checkpoint', f"{data_args.dataset_name}"))'''
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    early_stop_count = 0
    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []
        gnnNets.train()
        for batch in dataloader['train']:
            logits, probs, emb, graph_emb, h_all = gnnNets(batch)
            loss = criterion(logits, batch.y)

            # optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        # report train msg
        print(f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f} | "
              f"Acc: {np.concatenate(acc, axis=0).mean():.3f}")

        # report eval msg
        eval_state = evaluate_GC(dataloader['eval'], gnnNets, criterion)
        print(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f}")

        # only save the best model
        is_best = (eval_state['acc'] > best_acc)

        if eval_state['acc'] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_acc = eval_state['acc']
            early_stop_count = 0
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, eval_state['acc'], is_best)

    print(f"The best validation accuracy is {best_acc}.")
    # report test msg
    checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_best.pth'))
    #gnnNets.update_state_dict(checkpoint['net'])
    gnnNets.load_state_dict(checkpoint['net'])
    test_state, _, _ = test_GC(dataloader['test'], gnnNets, criterion)
    print(f"Test: | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")


def evaluate_GC(eval_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            logits, probs, emb, graph_emb, h_all = gnnNets(batch)
            loss = criterion(logits, batch.y)

            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        eval_state = {'loss': np.average(loss_list),
                      'acc': np.concatenate(acc, axis=0).mean()}

    return eval_state


def test_GC(test_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    pred_probs = []
    predictions = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            logits, probs, emb, graph_emb, h_all = gnnNets(batch)
            loss = criterion(logits, batch.y)

            # record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())
            predictions.append(prediction)
            pred_probs.append(probs)

    test_state = {'loss': np.average(loss_list),
                  'acc': np.average(np.concatenate(acc, axis=0).mean())}

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return test_state, pred_probs, predictions


def predict_GC(test_dataloader, gnnNets):
    """
    return: pred_probs --  np.array : the probability of the graph class
            predictions -- np.array : the prediction class for each graph
    """
    pred_probs = []
    predictions = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            logits, probs, _ = gnnNets(batch)

            ## record
            _, prediction = torch.max(logits, -1)
            predictions.append(prediction)
            pred_probs.append(probs)

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return pred_probs, predictions


# train for node classification task
def train_NC():
    print('start loading data====================')
    #import pdb; pdb.set_trace()
    dataset = get_dataset(data_args)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', f"{data_args.dataset_name}")):
        os.mkdir(os.path.join('checkpoint', f"{data_args.dataset_name}"))
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"

    data = dataset[0]
    gnnNets_NC = GnnNets_NC(input_dim, output_dim, model_args)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets_NC.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    best_val_loss = float('inf')
    best_acc = 0
    val_loss_history = []
    early_stop_count = 0
    for epoch in range(1, train_args.max_epochs + 1):
        gnnNets_NC.to(model_args.device)
        gnnNets_NC.train()
        logits, prob, _ = gnnNets_NC(data)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        eval_info = evaluate_NC(data, gnnNets_NC, criterion)
        eval_info['epoch'] = epoch

        if eval_info['val_loss'] < best_val_loss:
            best_val_loss = eval_info['val_loss']
            val_acc = eval_info['val_acc']

        val_loss_history.append(eval_info['val_loss'])

        # only save the best model
        is_best = (eval_info['val_acc'] > best_acc)

        if eval_info['val_acc'] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_acc = eval_info['val_acc']
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets_NC, model_args.model_name, eval_info['val_acc'], is_best)
        print(f'Epoch {epoch}, Train Loss: {eval_info["train_loss"]:.4f}, '
                        f'Train Accuracy: {eval_info["train_acc"]:.3f}, '
                        f'Val Loss: {eval_info["val_loss"]:.3f}, '
                        f'Val Accuracy: {eval_info["val_acc"]:.3f}')


    # report test msg
    checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_best.pth'))
    #gnnNets_NC.update_state_dict(checkpoint['net'])
    gnnNets_NC.load_state_dict(checkpoint['net'])
    gnnNets_NC.to(model_args.device)
    eval_info = evaluate_NC(data, gnnNets_NC, criterion)
    print(f'Test Loss: {eval_info["test_loss"]:.4f}, Test Accuracy: {eval_info["test_acc"]:.3f}')


def evaluate_NC(data, gnnNets_NC, criterion):
    eval_state = {}
    gnnNets_NC.eval()

    with torch.no_grad():
        for key in ['train', 'val', 'test']:
            mask = data['{}_mask'.format(key)]
            logits, probs, _, _ = gnnNets_NC(data)
            loss = criterion(logits[mask], data.y[mask]).item()
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            ## record
            eval_state['{}_loss'.format(key)] = loss
            eval_state['{}_acc'.format(key)] = acc

    return eval_state

def test_NC():
    print('start loading data====================')
    #import pdb; pdb.set_trace()
    dataset = get_dataset(data_args)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)

    # save path for model
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"

    data = dataset[0]
    gnnNets_NC = GnnNets_NC(input_dim, output_dim, model_args)
    criterion = nn.CrossEntropyLoss()

     # report test msg
    checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_best.pth'))
    #gnnNets_NC.update_state_dict(checkpoint['net'])
    gnnNets_NC.load_state_dict(checkpoint['net'])
    gnnNets_NC.to(model_args.device)
    eval_info = evaluate_NC(data, gnnNets_NC, criterion)
    print(f'Test Loss: {eval_info["test_loss"]:.4f}, Test Accuracy: {eval_info["test_acc"]:.3f}')


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    print('saving....')
    #gnnNets.to('cpu')
    state = {
        'net': gnnNets.state_dict(),
        'epoch': epoch,
        'acc': eval_acc
    }
    pth_name = f"{model_name}_latest.pth"
    best_pth_name = f'{model_name}_best.pth'
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))
        torch.save(gnnNets, os.path.join(ckpt_dir, f'{model_name}_net.pt'))
    #gnnNets.to_device()


if __name__ == '__main__':
    #train_GC()
    train_NC()
    #test_NC()
