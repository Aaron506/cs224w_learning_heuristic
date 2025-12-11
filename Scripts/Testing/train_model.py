import os
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from DatasetCreation.data_generator import GraphDatum
from DatasetCreation.data_to_torch import GraphDatumDataset
from Learning.heuristic_nn import HeuristicNN
import Learning.training_tools as tt
import Configs.load_from_config as lc

def plot_loss(train_losses, val_losses=None, show=True):
    fig = plt.figure()
    plt.plot(train_losses, label='Training Loss')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')

    if show:
        plt.show()

    return fig

if __name__ == '__main__':
    # 0. User settings
    TRAIN = True # True to fit the model
    SAVE = True # True to save the model
    LOAD = False # True to load a pre-existing model

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ml_config_path = 'Scripts/Configs/ml_config.yaml'
    ml_cfg = lc.load_config(ml_config_path)
    layer_type = 'gcn' # 'gcn','gat'

    # 0. Load the data and convert to PyG
    timestamp = '20251208_225420'
    foldername = f'Data/{timestamp}'
    dataset = pickle.load(open(os.path.join(foldername, 'dataset'), 'rb'))
    py_dataset = GraphDatumDataset(dataset)

    # 1. Create the relevant data loaders
    split_idx = tt.from_splits(ml_cfg['split_fracs'], len(py_dataset))
    train_dataset = Subset(py_dataset, split_idx["train"])
    val_dataset   = Subset(py_dataset, split_idx["val"])
    test_dataset  = Subset(py_dataset, split_idx["test"])
    train_loader = DataLoader(train_dataset, batch_size=ml_cfg['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=ml_cfg['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=ml_cfg['batch_size'], shuffle=False, num_workers=0)

    # 2. Instantiate the model
    if LOAD:
        model = pickle.load(open(os.path.join(foldername, f'model_{layer_type}'), 'rb'))
        model.to(device)
        model.device = device
    else:
        layer_args = {}
        if layer_type == 'gat':
            layer_args['heads'] = ml_cfg['num_heads']
            layer_args['concat'] = False
        model = HeuristicNN(ml_cfg, layer_type, device=device, **layer_args)

    # 3. Train the model
    loss_fn = tt.build_custom_loss(ml_cfg['reg'])
    if TRAIN:
        optimizer = torch.optim.Adam(model.parameters(), lr=ml_cfg['lr'])
        # Overwrite the current model with best seen in training
        train_losses, val_losses, model = tt.train(model, ml_cfg['epochs'], train_loader, 
                                                     loss_fn, optimizer, val_loader, verbose=True)
        # Plot the loss curve
        plot_loss(train_losses, val_losses, show=True)
        if SAVE:
            pickle.dump(model, open(os.path.join(foldername, f'model_{layer_type}'), 'wb'))

    # 4. Compute the final losses
    train_loss = tt.one_epoch(model, train_loader, loss_fn, None, False)
    val_loss = tt.one_epoch(model, val_loader, loss_fn, None, False)
    test_loss = tt.one_epoch(model, test_loader, loss_fn, None, False)
    print(f'Losses (train/val/test): {train_loss}, {val_loss}, {test_loss}')