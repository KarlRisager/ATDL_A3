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
        test_accuracy = test_model(val_masks, model, data)
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

def get_best(sweep):
    idx_best = 0
    best_acc = 0
    for i, test in enumerate(sweep):
        acc = test['test_accuracy']
        if acc > best_acc:
            best_acc = acc
            idx_best = i
    return sweep[idx_best]













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




def test_feature_noise_robustness(model, data, global_noise = True, scale_range = 0.25, scale_step = 0.01):
    scale = [i/(1/scale_step) for i in range(0, int((scale_range/scale_step)+1))]
    noisy_data_list = None
    if global_noise:
        noisy_data_list = [add_noise_to_features(data, s) for s in scale]
    else:
        noisy_data_list = [add_noise_to_features_local(data, s) for s in scale]
    assert noisy_data_list is not None
    acc = [test_model(data.test_mask, model, noisy_data) for noisy_data in noisy_data_list]
    return scale, acc


def test_edge_noise_robustness(model, data, max_added_edges=250, check_existing=True):
    num_added_edges = [i for i in range(max_added_edges+1)]
    noisy_data_list = [add_random_edges(data, n, check_existing=check_existing) for n in num_added_edges]
    acc = [test_model(data.test_mask, model, noisy_data) for noisy_data in noisy_data_list]
    return num_added_edges, acc

def test_edge_removal_robustness(model, data, max_removed_edges=250):
    num_removed_edges = [i for i in range(max_removed_edges+1)]
    noisy_data_list = [remove_edges(data, n) for n in num_removed_edges]
    acc = [test_model(data.test_mask, model, noisy_data) for noisy_data in noisy_data_list]
    return num_removed_edges, acc

def remove_edges(data, n):
    data_copy = copy.deepcopy(data)
    num_edges = data_copy.num_edges
    indices_to_be_removed = np.random.choice(num_edges, n, replace=False)
    data_copy.edge_index = torch.tensor(np.array([np.delete(data_copy.edge_index[0].numpy(), indices_to_be_removed),
                                         np.delete(data_copy.edge_index[1].numpy(), indices_to_be_removed)]))
    return data_copy

def add_random_edges(data, n, check_existing=True):
    '''This adds n random edges to the edge_index of the data object. If check_existing is True, it ensures the edges do not already exist in the graph.'''
    data_copy = copy.deepcopy(data)
    existing_edges = set(map(tuple, data.edge_index.t().tolist()))
    
    random_edges = []
    while len(random_edges) < n:
        edge = tuple(torch.randint(low=0, high=data.num_nodes, size=(2,)).tolist())
        if not check_existing or edge not in existing_edges:
            random_edges.append(edge)
            existing_edges.add(edge)
    
    random_edges = torch.tensor(random_edges, dtype=torch.long).t()  # Ensure the tensor is of type long
    data_copy.edge_index = torch.cat((data.edge_index, random_edges), dim=1)
    return data_copy









def do_n_tests(test_fun, model, data, n, global_noise=True):
    tests = []
    scale = None
    for i in range(n):
        scale, acc = test_fun(model, data)
        tests.append(acc)
        print(f'Finished test {i+1}/{n}', end='\r', flush=True)
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


def plot_n_tests(tests, scale, repeating = True, titels = None, xlabel='Noise scale', ylabel='Accuracy', std_label='std', titel='', line_label='Mean Accuracy'):
    if len(tests.shape) == 1:
        plt.plot(scale, tests)
    elif len(tests.shape) == 2:
        if not repeating:
            for i, test in enumerate(tests):
                if titels is not None:
                    plot_test(scale, test, more_plots_coming= i < len(tests) - 1, repeating=False, line_label=titels[i], xlabel=xlabel, ylabel=ylabel, std_label=std_label, titel=titel)
                else:
                    plot_test(scale, test, more_plots_coming= i < len(tests) - 1, repeating=False, xlabel=xlabel, ylabel=ylabel, line_label=line_label, std_label=std_label, titel=titel)
        else:
            plot_test(scale, tests, add_std_shading=True)
    else:
        for i, test in enumerate(tests):
            if titels is not None:
                plot_test(scale, test, add_std_shading=True, more_plots_coming= i < len(tests) - 1, line_label=titels[i], xlabel=xlabel, ylabel=ylabel, std_label=std_label, titel=titel)
            else:
                plot_test(scale, test, add_std_shading=True, more_plots_coming= i < len(tests) - 1, xlabel=xlabel, ylabel=ylabel, line_label=line_label, std_label=std_label, titel=titel)

    
