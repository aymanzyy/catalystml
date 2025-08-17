from mpi4py import MPI
import pandas as pd
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import random_split
from torch.utils.data import Subset
import torch.optim as optim
from torchinfo import summary

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure
NUMPOINTS = 5376
NUMCHANNELS = 4
LATENTSIZE=128
SCALEVEL = True
SCALEFACTOR = 1000000
MODELPATH = "dist_autoenc.pth"
LBMDATAPATH = ""

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
        #print("Encoded shape")
        #print(x.shape)
        x = self.decoder(x)
        return x
            
def prep_dataset(myRank_, with_scaling=False):
    
    column_names = ['X Coord', 'Y Coord', 'Z Coord', 'Vel Mag']
    main_str = "coordVel_000000{}_".format(myRank_)
    timesteps_of_interest = np.arange(10, 810, 10)
    file_list = []
    os.listdir()
    X_train_tensors = []
    X_train_tensors_copy = []

    for entry in timesteps_of_interest:
        full_str = main_str + str(entry) + "_runSize4.csv"
        full_path = LBMDATAPATH + full_str
        if os.path.isfile(full_path):
            file_list.append(full_path)

    for file in file_list:
        df = pd.read_csv(file,  header=None, names=column_names)
        numpy_array = df.values
        if (with_scaling):
            numpy_array[:, -1] = numpy_array[:, -1] * SCALEFACTOR
        torch_tensor = torch.tensor(np.expand_dims(numpy_array, axis=0), dtype=torch.float32)
        X_train_tensors.append(torch_tensor)
    cat_Torch = torch.cat(X_train_tensors, dim=0) 
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
    
    myRank_ = int(MPI.COMM_WORLD.Get_rank())
    worldSize_ = int(MPI.COMM_WORLD.Get_size())
    device_count = torch.cuda.device_count()
    dist.init_process_group(backend="nccl", world_size=worldSize_, rank=myRank_) 
    local_rank = myRank_ % device_count
    torch.cuda.set_device(0)     

    device = torch.device("cuda:{}".format(0)) 

    X_train, X_train_perm = prep_dataset(myRank_, SCALEVEL)
    X_train = X_train.to(device)
    X_train_perm = X_train_perm.to(device)
    
    print(X_train)
    print(X_train.shape)
    
    
    whole_dataset = TensorDataset(X_train_perm, X_train) 

    model = PointCloudAE(NUMPOINTS, LATENTSIZE)
    model = model.to(device)
    
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], output_device=0) 
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    
    train_loader = DataLoader(whole_dataset, batch_size=1, shuffle=True)
    train_losses =list()

    num_epochs = 80
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_X, batch_y in train_loader: 
            optimizer.zero_grad() 
            outputs = ddp_model(batch_X)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item()
            loss.backward()       
            optimizer.step()      
        epoch_loss = running_loss / len(train_loader)
        if (epoch % 10 == 0):
            if (myRank_ == 0):
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        train_losses.append(epoch_loss)

    plt.figure(figsize=(18, 12))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("static_dataset_insitu_train_loss_w_scaled_rank{}.png".format(myRank_))

if __name__ == "__main__":
    main()
