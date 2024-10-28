from torch_geometric.nn.models import GAT
import torch
from sklearn.model_selection import ParameterGrid
import pickle
import os
import torch.nn.functional as F
from carbontracker.tracker import CarbonTracker
import numpy as np
#import copy
#import matplotlib.pyplot as plt
from .metrics import *
from .pertubate import *
from .stats import *



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

def load_sweep(path_to_sweep):
    with open(path_to_sweep, 'rb') as f:
        sweep = pickle.load(f)
    return sweep


def load_best_sweep(path_to_sweep):
    sweep = load_sweep(path_to_sweep)
    best = get_best(sweep)
    return best

def initialize_model_from_sweep(path_to_sweep, model_type, dataset):
    model = None
    optimizer = None
    s = load_best_sweep(path_to_sweep)['hyperparameters']
    if model_type=='GAT':
        model = GAT(in_channels=dataset.num_features, hidden_channels=s['hidden_channels']*s['heads'], num_layers=s['num_layers'], out_channels=dataset.num_classes, heads=s['heads'], dropout=s['dropout'], act=F.elu)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    elif model_type=='GATv2':
        model = GAT(v2 = True, in_channels=dataset.num_features, hidden_channels=s['hidden_channels']*s['heads'], num_layers=s['num_layers'], out_channels=dataset.num_classes, heads=s['heads'], dropout=s['dropout'], act=F.elu)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    return model, optimizer




def normalize_accuracy(accuracy):
    non_perturbed = accuracy[0]
    return np.array([((acc/non_perturbed)) for  acc in accuracy])

def normalize_accuracy_n_tests(accuracy_list):
    
    return np.array([normalize_accuracy(accuracy) for accuracy in accuracy_list])




def do_n_tests(test_fun, model, data, n, global_noise=True):
    tests = []
    scale = None
    for i in range(n):
        scale, acc = test_fun(model, data)
        tests.append(acc)
        print(f'Finished test {i+1}/{n}', end='\r', flush=True)
    assert scale is not None
    return np.array(scale), np.array(tests)

