from torch_geometric.nn.models import GAT, GCN
import torch
from sklearn.model_selection import ParameterGrid
import pickle
import os
import torch.nn.functional as F

def train_model(num_epochs, model, data, optimizer, criterion, return_loss = True, print_status = True):
    loss_list = []
    if print_status:
        print("Training model...\n")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(data.x, data.edge_index)
        loss = criterion(y_pred[data.train_mask], data.y[data.train_mask])
        if return_loss:
            loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        if print_status:
            print(f"Epoch {epoch}, Loss: {loss.item()}", end="\r", flush=True)
    if return_loss:
        return loss_list


def test_model(mask, model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
    acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    return acc






# Updated hyperparameter sweep function
def hp_sweep(num_epochs, val_masks, dataset, hyperparameters_gat, criterion, filename, model_type='GAT'):
    results = []
    data = dataset[0]

    # Use sklearn's ParameterGrid for hyperparameter combinations
    param_grid = ParameterGrid(hyperparameters_gat)
    n = len(param_grid)

    for idx, params in enumerate(param_grid):
        hidden_channels = params['hidden_channels']
        num_layers = params['num_layers']
        heads = params['heads']
        dropout = params['dropout']

        #print(f'Running with hidden_channels: {hidden_channels}, num_layers: {num_layers}, heads: {heads}, dropout: {dropout}')
        

        model = None
        if model_type == 'GAT':
            model = GAT(in_channels=dataset.num_features, hidden_channels=hidden_channels * heads, 
                  num_layers=num_layers, out_channels=dataset.num_classes, heads=heads, dropout=dropout, act=F.elu)
        elif model_type == 'GATv2':
            model = GAT(v2=True, in_channels=dataset.num_features, hidden_channels=hidden_channels * heads, 
                    num_layers=num_layers, out_channels=dataset.num_classes, heads=heads, dropout=dropout, act=F.elu)

        assert model is not None

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        # Train model and save training loss and test accuracy
        loss_list = train_model(num_epochs, model, data, optimizer, criterion, print_status=False)
        test_accuracy = test_model(val_masks, model, dataset)

        
        # Append the result for this set of hyperparameters
        result = {
            'hyperparameters': params,
            'train_loss': loss_list,
            'test_accuracy': test_accuracy
        }

        results.append(result)

        print(f'Finished hyperparameter combination {idx+1}/{n}', end='\r', flush=True)

    # Ensure the 'sweeps' directory exists
    os.makedirs('sweeps', exist_ok=True)
    
    # Check if the file already exists and modify the filename if necessary
    base_filename, extension = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join('sweeps', new_filename)):
        new_filename = f"{base_filename}_{counter}{extension}"
        counter += 1

    # Save the results using pickle
    with open(os.path.join('sweeps', new_filename), 'wb') as f:
        pickle.dump(results, f)
    
    return results













#Statistics

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def dataset_statistics(dataset):
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')



