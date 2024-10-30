from .scripts import test_model
import torch
import numpy as np
import copy


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

def test_feature_noise_robustness(model, data, scale_range = 0.25, scale_step = 0.01):
    acc_list = []
    scale_list = []
    for i in range(0, int((scale_range/scale_step)+1)):
        scale = i/(1/scale_step)
        scale_list.append(scale)

        noisy_data = add_noise_to_features(data, scale)

        acc = test_model(data.test_mask, model, noisy_data)
        acc_list.append(acc)
        del noisy_data

    return np.array(scale_list), np.array(acc_list)




def test_edge_adding_robustness(model, data, max_added_edges=150, step_size=1, check_existing=True):
    scale_list = []
    acc_list = []
    for i in range(0, max_added_edges, step_size):
        scale_list.append(i)

        noisy_data = add_random_edges(data, i, check_existing=check_existing)

        acc = test_model(data.test_mask, model, noisy_data)
        acc_list.append(acc)
        del noisy_data
    return np.array(scale_list), np.array(acc_list)
        
        



def test_edge_removal_robustness(model, data, max_removed_edges=1000, step_size=5):
    scale_list = []
    acc_list = []
    for i in range(0, max_removed_edges, step_size):
        scale_list.append(i)

        noisy_data = remove_edges(data, i)

        acc = test_model(data.test_mask, model, noisy_data)
        acc_list.append(acc)
        del noisy_data
    return np.array(scale_list), np.array(acc_list)

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

    del existing_edges, random_edges
    return data_copy
