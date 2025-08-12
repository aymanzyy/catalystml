
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

sys.path.append("/lus/grand/projects/visualization/azy4/catalyst_based_ml/autoencoder/")
from dynamic_dataset_class_definition import DynamicDataset

from models.pn_autoencoder import PointCloudAE
NUMPOINTS_PER_RANK=5376
NUMCHANNELS = 4
LATENTSIZE=128
SCALEVEL = True
SCALEFACTOR = 1000000
MODELPATH = "/lus/grand/projects/visualization/azy4/catalyst_based_ml/autoencoder/model_weights/autoenc.pth"

myRank_ = int(MPI.COMM_WORLD.Get_rank())
worldSize_ = int(MPI.COMM_WORLD.Get_size())
device_count = torch.cuda.device_count()
dist.init_process_group(backend="nccl", world_size=worldSize_, rank=myRank_) 
local_rank = myRank_ % device_count
torch.cuda.set_device(0)

model = PointCloudAE(NUMPOINTS_PER_RANK, LATENTSIZE)
device = torch.device("cuda:{}".format(0)) 
model = model.to(device)
ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], output_device=0) 

training_cycles = 0

criterion = nn.MSELoss()
optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

globalDataset = DynamicDataset()
global_epoch = 0
train_losses =list()

def inject_data(xdata, ydata):
    globalDataset.append_data(xdata,ydata)

def train_loop(train_loader_):
    num_epochs = 80
    global global_epoch 
    global training_losses
    for epoch in range(num_epochs):
        batch_temp = 0
        running_loss = 0.0
        for batch_X, batch_y in train_loader_:
            optimizer.zero_grad()  
            outputs = ddp_model(batch_X)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item()
            loss.backward()        
            optimizer.step()     
            batch_temp+=1
            
        epoch_loss = running_loss / len(train_loader_)
        train_losses.append(epoch_loss)
    global_epoch+=num_epochs
    
def main(pythInput, cat_step):

    torch_tensor = torch.tensor(pythInput, dtype=torch.float32)
    torch_tensor = torch_tensor.to(device)
    torch_tensor_perm = torch.transpose(torch_tensor, 0, 1)
    
    inject_data(torch_tensor,torch_tensor_perm)
        
    if (cat_step == 800):
        train_loader = DataLoader(globalDataset, batch_size=1, shuffle=True)
        train_loop(train_loader)
        plt.figure(figsize=(18, 12))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
if __name__ == "__main__":
    main(); 