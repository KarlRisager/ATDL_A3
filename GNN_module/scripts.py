from torch_geometric.nn.models import GAT, GCN
import torch
from sklearn.model_selection import ParameterGrid
import pickle
import os
import torch.nn.functional as F
from carbontracker.tracker import CarbonTracker
import numpy as np
import copy
import matplotlib.pyplot as plt


def train_model(num_epochs, model, data, optimizer, criterion, return_loss = True, print_status = True, track_emissions = False):
    loss_list = []
    tracker = None
    if track_emissions:
        tracker = CarbonTracker(epochs=num_epochs)
    if print_status:
        print("Training model...\n")
    for epoch in range(num_epochs):
        if track_emissions:
            tracker.epoch_start()
        optimizer.zero_grad()
        y_pred = model(data.x, data.edge_index)
        loss = criterion(y_pred[data.train_mask], data.y[data.train_mask])
        if return_loss:
            loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        if print_status:
            print(f"Epoch {epoch}, Loss: {loss.item()}", end="\r", flush=True)
        if track_emissions:
            tracker.epoch_end()
    if return_loss:
        return loss_list
    if track_emissions:
        tracker.stop()


def test_model(mask, model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()  # Use the class with highest probability and move to CPU.
        correct = np.equal(pred[mask.cpu().numpy()], data.y[mask].cpu().numpy())  # Check against ground-truth labels.
        acc = np.mean(correct)  # Derive ratio of correct predictions.
    return acc





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
        number_of_parameters = count_trainable_parameters(model)
        
        # Append the result for this set of hyperparameters
        result = {
            'hyperparameters': params,
            'train_loss': loss_list,
            'test_accuracy': test_accuracy,
            'number_of_parameters': number_of_parameters
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






#Robustness:

def add_noise_to_features(data, scale):
    data_copy = copy.deepcopy(data)
    noise = torch.randn_like(data_copy.x) * scale
    data_copy.x += noise
    return data_copy

def add_noise_to_features_local(data, scale):
    data_copy = copy.deepcopy(data)
    n_nodes = data_copy.x.shape[0]
    n_features = data_copy.x.shape[1]
    for i in range(n_nodes):
        data_copy.x[i] += torch.randn(n_features)*scale
    return data_copy

def feature_noise(data, data_noisy):
    mse = np.mean((data.x.detach().cpu().numpy() - data_noisy.x.detach().cpu().numpy()) ** 2)
    return mse



def test_feature_noise_robustness(model, data, global_noise = True, scale_range = 0.2, scale_step = 0.01):
    scale = [i/(1/scale_step) for i in range(0, int((scale_range/scale_step)+1))]
    noisy_data_list = None
    if global_noise:
        noisy_data_list = [add_noise_to_features(data, s) for s in scale]
    else:
        noisy_data_list = [add_noise_to_features_local(data, s) for s in scale]
    assert noisy_data_list is not None
    acc = [test_model(data.test_mask, model, noisy_data) for noisy_data in noisy_data_list]
    return scale, acc







def do_n_tests(test_fun, model, data, n, global_noise=True):
    tests = []
    scale = None
    for i in range(n):
        scale, acc = test_fun(model, data, global_noise)
        tests.append(acc)
    assert scale is not None
    return np.array(scale), np.array(tests)







#plotting

def plot_test(scale, tests, add_std_shading = False,  more_plots_coming=False, repeating = True, titel='', xlabel='Noise scale', ylabel='Accuracy', line_label='Mean Accuracy', std_label='std'):
    mean_data = np.mean(tests, axis=0)
    std_data = np.std(tests, axis=0)
    if repeating:
        plt.plot(scale, mean_data, label=line_label)
    else:
        plt.plot(scale, tests, label=line_label)

    if add_std_shading:
        plt.fill_between(scale, mean_data - std_data, mean_data + std_data, alpha=0.2, label=std_label)
    if not(more_plots_coming):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(titel)
        plt.legend()
        plt.show()


def plot_n_tests(tests, scale, repeating = True, titels = None):
    if len(tests.shape) == 1:
        plt.plot(scale, tests)
    elif len(tests.shape) == 2:
        if not repeating:
            for i, test in enumerate(tests):
                if titels is not None:
                    plot_test(scale, test, more_plots_coming= i < len(tests) - 1, repeating=False, line_label=titels[i])
                else:
                    plot_test(scale, test, more_plots_coming= i < len(tests) - 1, repeating=False)
        else:
            plot_test(scale, tests, add_std_shading=True)
    else:
        for i, test in enumerate(tests):
            if titels is not None:
                plot_test(scale, test, add_std_shading=True, more_plots_coming= i < len(tests) - 1, line_label=titels[i])
            else:
                plot_test(scale, test, add_std_shading=True, more_plots_coming= i < len(tests) - 1)

    
