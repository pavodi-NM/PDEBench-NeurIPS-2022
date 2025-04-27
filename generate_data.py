import torch
import sys 
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Import PDEBench dataloader
from pdebench.utils.dataset import PDEBenchDataset
random_seed = 3407
torch.manual_seed(random_seed)
np.random.seed(random_seed)


def create_and_save_dataloaders(file_names, reduced_resolution, reduced_resolution_t, reduced_batch, initial_step, base_path, batch_size, num_workers):
    # Create datasets
    train_data = PDEBenchDataset(file_names, reduced_resolution=reduced_resolution, reduced_resolution_t=reduced_resolution_t,
                                 reduced_batch=reduced_batch, initial_step=initial_step, saved_folder=base_path)
    
    val_data = PDEBenchDataset(file_names, reduced_resolution=reduced_resolution, reduced_resolution_t=reduced_resolution_t,
                               reduced_batch=reduced_batch, initial_step=initial_step, if_test=True, saved_folder=base_path)
    
    print(f"len of the train data, {len(train_data)}") # 9000 points are used to train
    print(f"len of the val data: {len(val_data)}")     # 1000 points are used to test
    #sys.exit()
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    # Save dataloaders for reaction
    # torch.save(train_loader, '/home/maniamfu/PDEBench/save_data/reaction/train_0.5_bs50.pth')
    # torch.save(val_loader, '/home/maniamfu/PDEBench/save_data/reaction/val_0.5_bs50.pth')
    
    # save data for burgers 
    torch.save(train_loader, '/home/maniamfu/PDEBench/save_data/burgers/train_0.01_32.pth')
    torch.save(val_loader, '/home/maniamfu/PDEBench/save_data/burgers/val_0.01_32.pth')
    
    #torch.save(train_loader, 'save_train_data/1D_burgers/train_loader_bv1.0.ICs10.pth')
    #torch.save(val_loader, 'save_train_data/1D_burgers/val_loader_bv1.0.ICs10.pth')



base_path = "/home/maniamfu/PDEBench/pdebench/data_download/pdebench/data/1D/Burgers/Train/"   #"pdebench/data/"
file_names = ["1D_Burgers_Sols_Nu0.01.hdf5"]    # ["ReacDiff_Nu0.5_Rho1.0.hdf5"] # 0.001 is the viscosity (Burger's parameter), ["1D_Burgers_Sols_Nu0.001.hdf5"] 1D_Burgers_Sols_Nu1.0
initial_step = 10 # one is the first way, than 10 for comparison purpose
reduced_resolution = 4
reduced_resolution_t = 5
reduced_batch = 1

num_workers = 8
batch_size = 32 # basic 32, 50

random_seed = 3407


create_and_save_dataloaders(file_names=file_names, reduced_resolution=reduced_resolution, reduced_resolution_t=reduced_resolution_t,
                            reduced_batch=reduced_batch, initial_step=initial_step, base_path=base_path, batch_size=batch_size,
                            num_workers=num_workers)