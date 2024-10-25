from GNN_module.scripts import *
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
import argparse
import sys



print('Starting hyperparameter sweep...')
sys.stdout.flush()
print('--------------------------------------------------------------------------')
sys.stdout.flush()

parser = argparse.ArgumentParser()


parser.add_argument( '--epochs', type=int, default=500)
parser.add_argument('--sweep_name', type=str, default='sweep_results')
parser.add_argument('--hidden_channels', type=list, default=[1, 2, 3, 4])
parser.add_argument('--num_layers', type=list, default=[1, 2, 4, 8])
parser.add_argument('--heads', type=list, default=[1, 2, 4, 8, 16])
parser.add_argument('--dropout', type=list, default=[0.0, 0.2, 0.4, 0.6, 0.8])


args = parser.parse_args()
num_epochs = args.epochs
sweep_name = args.sweep_name
hidden_channels = args.hidden_channels
num_layers = args.num_layers
heads = args.heads
dropout = args.dropout


args = parser.parse_args()


#Load dataset
dataset_cora = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
dataset_citeseer = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
dataset_pubmed = Planetoid(root='data/Planetoid', name='Pubmed', transform=NormalizeFeatures())


dataset = dataset_cora
data = dataset[0]  # Get the first graph object.
dataset_statistics(dataset)
criterion = torch.nn.CrossEntropyLoss()

hyperparameters_gat = {'hidden_channels': hidden_channels, 'num_layers': num_layers, 'heads': heads, 'dropout': dropout}

file_name = sweep_name + 'gat_cora.pkl'
sweep_results = hp_sweep(num_epochs, data.val_mask, dataset, hyperparameters_gat, criterion, file_name, model_type='GAT')

file_name = sweep_name + 'gatv2_cora.pkl'
sweep_resultsv2 = hp_sweep(num_epochs, data.val_mask, dataset, hyperparameters_gat, criterion, file_name, model_type='GATv2')