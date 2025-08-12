import pandas as pd
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset
import torch.optim as optim
from torchinfo import summary

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure
NUMPOINTS = 21504
NUMCHANNELS = 4
LATENTSIZE=128
SCALEVEL = True
SCALEFACTOR = 1000000
MODELPATH = "/lus/grand/projects/visualization/azy4/catalyst_based_ml/autoencoder/model_weights/autoenc_paper.pth"

class PointCloudAE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAE, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        
        self.conv1 = torch.nn.Conv1d(NUMCHANNELS, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)
        
        self.dec1 = nn.Linear(self.latent_size,256)
        self.dec2 = nn.Linear(256,256)
        self.dec3 = nn.Linear(256,self.point_size*NUMCHANNELS)

    def encoder(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, NUMCHANNELS)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

def calculate_frobenius_error(output, labels):
    num_pts = np.shape(output)[0]
    num_channels = np.shape(output)[-1]
    assert(NUMPOINTS == num_pts)
    assert(NUMCHANNELS== num_channels)
   
    rel_loss = []
    gt_grid = []
    for ii in range(num_pts):
        for jj in range(num_channels):
            rel_loss.append((output[ii, jj] - labels[ii, jj]) ** 2)
            gt_grid.append(labels[ii, jj] ** 2)
    top_frac = np.sqrt(np.sum(np.array(rel_loss)))
    bottom_frac = np.sqrt(np.sum(np.array(gt_grid)))
    frob_err = (top_frac/bottom_frac)
    return frob_err
def calculate_sep_mse_error(output, labels):
    num_pts = np.shape(output)[0]
    num_channels = np.shape(output)[-1]
    assert(NUMPOINTS == num_pts)
    assert(NUMCHANNELS== num_channels)
    
    rel_loss_grid = []
    rel_loss_vel = []
    for ii in range(num_pts):
        for jj in range(num_channels-1): ## Last channel is vel
            rel_loss_grid.append((output[ii, jj] - labels[ii, jj]) ** 2)
        rel_loss_vel.append((output[ii, -1] - labels[ii, -1]) ** 2)
    mse_grid = np.sum(np.array(rel_loss_grid))/num_pts
    mse_vel = np.sum(np.array(rel_loss_vel))/num_pts
    return mse_grid, mse_vel 
def prep_dataset(with_scaling=False):
    
    column_names = ['X Coord', 'Y Coord', 'Z Coord', 'Vel Mag']

    dir_path = "/lus/grand/projects/visualization/azy4/catalyst_based_ml/Mini-Apps/lbm_data_1rank/"
    main_str = "coordVel_0000000_"

    timesteps_of_interest = np.arange(10, 1010, 10)
    file_list = []
    os.listdir()
    X_train_tensors = []
    X_train_tensors_copy = []

    for entry in timesteps_of_interest:
        full_str = main_str + str(entry) + ".csv"
        full_path = dir_path + full_str
        if os.path.isfile(full_path):
            file_list.append(full_path)

    for file in file_list:
        df = pd.read_csv(file,  header=None, names=column_names)
        numpy_array = df.values
        if (with_scaling):
            numpy_array[:, -1] = numpy_array[:, -1] * SCALEFACTOR
        torch_tensor = torch.tensor(np.expand_dims(numpy_array, axis=0), dtype=torch.float32)
        X_train_tensors.append(torch_tensor)
    cat_Torch = concatenated_tensor = torch.cat(X_train_tensors, dim=0) 
    cat_Torch_copy = torch.transpose(torch.cat(X_train_tensors, dim=0), 1 ,2) 
    return cat_Torch, cat_Torch_copy


def split_into_train_test_in_order(whole_dataset):
    train_size = int(0.8 * len(whole_dataset))
    val_size = len(whole_dataset) - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(whole_dataset)))

    train_subset = Subset(whole_dataset, train_indices)
    val_subset = Subset(whole_dataset, val_indices)

    return train_subset, val_subset

def main():
    
    device = torch.device("cuda:{}".format(0))

    X_train, X_train_perm = prep_dataset(SCALEVEL)
    X_train = X_train.to(device)
    X_train_perm = X_train_perm.to(device)
    
    whole_dataset = TensorDataset(X_train_perm, X_train)
    
    model = PointCloudAE(NUMPOINTS, LATENTSIZE)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    num_epochs = 100
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_X, batch_y in train_loader: ## x and y are the same, reconstruction problem
            # Forward pass
            optimizer.zero_grad()  # Clear gradients from previous step
            #print("Batch size: " + str(batch_X.shape))
            outputs = model(batch_X)
            #print("outputs size: " + str(outputs.shape))
            loss = criterion(outputs, batch_y)
            running_loss += loss.item()
            # Backward and optimize
            loss.backward()        # Compute gradients
            optimizer.step()       # Update weights

        if (epoch % 10 == 0):
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}')
        loss_tensor = loss.clone().detach() # Detach to prevent gradient flow through this operation

    torch.save(model.state_dict(),MODELPATH)



if __name__ == "__main__":
    main()
