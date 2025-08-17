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

from pn_autoencoder import PointCloudAE
NUMPOINTS = 21504
NUMCHANNELS = 4
LATENTSIZE=128
SCALEVEL = True
SCALEFACTOR = 1000000
MODELPATH = "autoenc_paper.pth"

model = PointCloudAE(NUMPOINTS, LATENTSIZE)
device = torch.device("cuda:{}".format(0))
model.load_state_dict(torch.load(MODELPATH, weights_only=True))
model = model.to(device)
model.eval()

criterion = nn.MSELoss()
model.eval()

def infer(imported_data, timestep):
    torch_tensor = torch.tensor(imported_data, dtype=torch.float32)
    torch_tensor = torch_tensor.to(device)
    with torch.no_grad():
        encoded_tensor = model.encoder(torch_tensor)
    file_path = 'offline_trained_insitu_encoding_timestep_{}.pt'.format(timestep)
    torch.save(encoded_tensor, file_path)

def main():
    print("rerun")

if __name__ == "__main__":
    main(); 