from .scripts import test_model
import torch
import numpy as np
import copy



# region pertubation functions
def add_noise_to_features(data, scale):
    data_copy = copy.deepcopy(data)
    noise = torch.randn_like(data_copy.x) * scale
    data_copy.x += noise
    del noise
    return data_copy


def remove_edges(data, n):
    data_copy = copy.deepcopy(data)
    num_edges = data_copy.num_edges
    indices_to_be_removed = np.random.choice(num_edges, n, replace=False)
    data_copy.edge_index = torch.tensor(np.array([np.delete(data_copy.edge_index[0].numpy(), indices_to_be_removed),
                                         np.delete(data_copy.edge_index[1].numpy(), indices_to_be_removed)]))
    del num_edges, indices_to_be_removed
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
    
    random_edges = torch.tensor(random_edges, dtype=torch.long).t()
    data_copy.edge_index = torch.cat((data.edge_index, random_edges), dim=1)

    del existing_edges, random_edges
    return data_copy

def remove_random_features(data, n):
    data_copy = copy.deepcopy(data)
    num_features = data_copy.num_features
    indicies_to_zero = np.random.choice(num_features, n, replace=False)
    data_copy.x[:,indicies_to_zero] = 0
    del num_features, indicies_to_zero
    return data_copy

def remove_nodes2(data, n): #Remove this function
    data_copy = copy.deepcopy(data)
    num_nodes = data.num_nodes
    indices_to_remove = np.random.choice(num_nodes, n, replace=False)
    
    # Removing nodes
    data_copy.x = torch.tensor(np.delete(data_copy.x.numpy(), indices_to_remove, axis=0))
    data_copy.y = torch.tensor(np.delete(data_copy.y.numpy(), indices_to_remove))
    data_copy.train_mask = torch.tensor(np.delete(data_copy.train_mask.numpy(), indices_to_remove))
    data_copy.val_mask = torch.tensor(np.delete(data_copy.val_mask.numpy(), indices_to_remove))
    data_copy.test_mask = torch.tensor(np.delete(data_copy.test_mask.numpy(), indices_to_remove))
    
    # Removing all edges to removed nodes
    mask = np.isin(data_copy.edge_index[0].numpy(), indices_to_remove) | np.isin(data_copy.edge_index[1].numpy(), indices_to_remove)
    data_copy.edge_index = torch.tensor(data_copy.edge_index[:, ~mask])
    
    return data_copy


def remove_nodes(data, n):
    data_copy = copy.deepcopy(data)
    num_nodes = data.num_nodes
    indices_to_remove = np.random.choice(num_nodes, n, replace=False)
    
    # Create a mapping from old node indices to new node indices
    mask = np.ones(num_nodes, dtype=bool)
    mask[indices_to_remove] = False
    mapping = np.cumsum(mask) - 1  # Map old indices to new ones
    
    # Removing nodes
    data_copy.x = torch.tensor(np.delete(data_copy.x.numpy(), indices_to_remove, axis=0))
    data_copy.y = torch.tensor(np.delete(data_copy.y.numpy(), indices_to_remove))
    data_copy.train_mask = torch.tensor(np.delete(data_copy.train_mask.numpy(), indices_to_remove))
    data_copy.val_mask = torch.tensor(np.delete(data_copy.val_mask.numpy(), indices_to_remove))
    data_copy.test_mask = torch.tensor(np.delete(data_copy.test_mask.numpy(), indices_to_remove))
    
    # Update edge_index to remove edges to removed nodes and remap the remaining indices
    edge_index = data_copy.edge_index.numpy()
    mask_edges = np.isin(edge_index[0], indices_to_remove) | np.isin(edge_index[1], indices_to_remove)
    edge_index = edge_index[:, ~mask_edges]  # Keep edges not connected to removed nodes
    
    # Remap the edge indices based on the new node indices
    edge_index = mapping[edge_index]
    
    # Assign the updated edge_index back to data_copy
    data_copy.edge_index = torch.tensor(edge_index)
    
    return data_copy






# endregion



# region robustness tests



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

def test_feature_removal_robustness(model, data, max_removed_features=1000, step_size=5):
    scale_list = []
    acc_list = []
    for i in range(0, max_removed_features, step_size):
        scale_list.append(i)

        noisy_data = remove_random_features(data, i)

        acc = test_model(data.test_mask, model, noisy_data)
        acc_list.append(acc)
        del noisy_data
    return np.array(scale_list), np.array(acc_list)

def test_node_removal_robustness(model, data, max_removed_nodes=1000, step_size=5):
    scale_list = []
    acc_list = []
    for i in range(0, max_removed_nodes, step_size):
        scale_list.append(i)

        noisy_data = remove_nodes(data, i)

        acc = test_model(noisy_data.test_mask, model, noisy_data)
        acc_list.append(acc)
        del noisy_data
    return np.array(scale_list), np.array(acc_list)


# endregion









def feature_noise(data, data_noisy):
    mse = np.mean((data.x.detach().cpu().numpy() - data_noisy.x.detach().cpu().numpy()) ** 2)
    return mse