import argparse
import sys

import torch
from GNN_module.scripts import *
from GNN_module.stats import *
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

print('Starting hyperparameter sweep...')
sys.stdout.flush()
print('--------------------------------------------------------------------------')
sys.stdout.flush()

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--sweep_name', type=str, default='sweep_results')
parser.add_argument('--hidden_channels', type=list, default=[1, 2, 3, 4])
parser.add_argument('--num_layers', type=list, default=[1, 2, 4, 8])
parser.add_argument('--heads', type=list, default=[1, 2, 4, 8, 16])
parser.add_argument('--dropout', type=list, default=[0.0, 0.2, 0.4, 0.6, 0.8])
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--model_type', type=str, default='GAT')

args = parser.parse_args()
num_epochs = args.epochs
sweep_name = args.sweep_name
hidden_channels = args.hidden_channels
num_layers = args.num_layers
heads = args.heads
dropout = args.dropout
dataset_name = args.dataset
model_type = args.model_type

args = parser.parse_args()

# Load dataset
dataset_cora = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
dataset_citeseer = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
dataset_pubmed = Planetoid(root='data/Planetoid', name='Pubmed', transform=NormalizeFeatures())

dataset = None
if dataset_name == 'Cora':
    dataset = dataset_cora
elif dataset_name == 'CiteSeer':
    dataset = dataset_citeseer
elif dataset_name == 'Pubmed':
    dataset = dataset_pubmed

data = dataset[0]  # Get the first graph object.
dataset_statistics(dataset)
criterion = torch.nn.CrossEntropyLoss()

if model_type == 'GAT':
    print(f'Running GAT hyperparameter sweep on {dataset_name}', flush=True)
    hyperparameters_gat = {'hidden_channels': hidden_channels, 'num_layers': num_layers, 'heads': heads,
                           'dropout': dropout}
    file_name = sweep_name + '_gat_' + dataset_name + '.pkl'
    sweep_results = hp_sweep(num_epochs, data.val_mask, dataset, hyperparameters_gat, criterion, file_name,
                             model_type='GAT')
elif model_type == 'GATv2':
    print(f'Running GATv2 hyperparameter sweep on {dataset_name}', flush=True)
    hyperparameters_gat = {'hidden_channels': hidden_channels, 'num_layers': num_layers, 'heads': heads,
                           'dropout': dropout}
    file_name = sweep_name + '_gatv2_' + dataset_name + '.pkl'
    sweep_resultsv2 = hp_sweep(num_epochs, data.val_mask, dataset, hyperparameters_gat, criterion, file_name,
                               model_type='GATv2')
