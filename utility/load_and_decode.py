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
MODELPATH = "autoenc_paper.pth"
from pn_autoencoder import PointCloudAE

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
        for jj in range(num_channels-1):
            rel_loss_grid.append((output[ii, jj] - labels[ii, jj]) ** 2)
        rel_loss_vel.append((output[ii, -1] - labels[ii, -1]) ** 2)
    mse_grid = np.sum(np.array(rel_loss_grid))/num_pts
    mse_vel = np.sum(np.array(rel_loss_vel))/num_pts
    return mse_grid, mse_vel 



def main():
    device = torch.device("cuda:{}".format(0))
    column_names = ['X Coord', 'Y Coord', 'Z Coord', 'Vel Mag']

    model = PointCloudAE(NUMPOINTS, LATENTSIZE)
    model = model.to(device)
    
    model.load_state_dict(torch.load(MODELPATH, weights_only=True))
    loaded_tensor = torch.load('offline_trained_insitu_encoding_timestep_990.pt')
    loaded_tensor = loaded_tensor.to(device)
    model.eval()
    with torch.no_grad():
        decoded_tensor = model.decoder(loaded_tensor)
        
    gt_string = "coordVel_0000000_990_runSize1.csv"
    labels = pd.read_csv(gt_string, header=None, names=column_names).values
    labels = np.squeeze(np.asarray(labels)) 
    
    decoded_numpy = np.squeeze(decoded_tensor.cpu().numpy()) 

    if (SCALEVEL):
        decoded_numpy[:, -1] = decoded_numpy[:, -1]/SCALEFACTOR 
        #labels[:, -1] = labels[:, -1]/SCALEFACTOR
    
    raveled_ground_truth = np.squeeze(labels[:, -1])
    raveled_prediction =  np.squeeze(decoded_numpy[:, -1])
    
    minmin = np.min([np.min(raveled_ground_truth), np.min(raveled_prediction)])
    maxmax = np.max([np.max(raveled_ground_truth), np.max(raveled_prediction)])

    azim_par = 240
    elev_par = 30
    figure(figsize=(80, 48))
    fig = plt.figure()
    ax1 = fig.add_subplot(121,projection='3d')
    ax1.view_init(elev=elev_par, azim=azim_par)
    ax1plot = ax1.scatter3D(labels[:, 0],labels[:, 1], labels[:, 2], s=5, c = labels[ :, 3], vmin=minmin, vmax=maxmax,
            cmap='magma', edgecolor='none')
            
    ax2 = fig.add_subplot(122,projection='3d')
    ax2.view_init(elev=elev_par, azim=azim_par)
    ax2plot = ax2.scatter3D(decoded_numpy[:, 0],decoded_numpy[:, 1], decoded_numpy[:, 2], s=5, c = decoded_numpy[:, 3], vmin=minmin, vmax=maxmax,
            cmap='magma', edgecolor='none')

    ax1.set_title(
            "Label Grnd Truth")
        
    ax2.set_title(
            "Label Prediction")
    plt.tight_layout()

    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])  # [left, bottom, width, height]
    fig.colorbar(ax2plot, cax=cbar_ax, orientation='horizontal')
    plt.show()
    plt.savefig("manual_encode_decode_sample_prediction.png")
    header = ['X', 'Y', 'Z', 'Velocity']
    
    fro_error = calculate_frobenius_error(decoded_numpy, labels)
    mse_grid_, mse_vel_ = calculate_sep_mse_error(decoded_numpy, labels)
    print("Frobenius Error [Whole]: {}, Mean Squared Error [Grid]: {}, Mean Squared Error [Velocity]: {}".format(fro_error, mse_grid_, mse_vel_))
    
    df_pred = pd.DataFrame(np.squeeze(decoded_numpy), columns=header)
    df_gt = pd.DataFrame(np.squeeze(labels), columns=header)    

if __name__ == "__main__":
    main()