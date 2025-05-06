# -*- coding: utf-8 -*-

import sys
import os 
import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

from timeit import default_timer 
 
# torch.manual_seed(0)
# np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pdebench.models.unet.unet import UNet1d, UNet2d, UNet3d
from pdebench.models.unet.utils import UNetDatasetSingle, UNetDatasetMult
from pdebench.models.metrics import metrics
from pdebench.utils.dataset import PDEBenchDataset

def run_training(if_training,
                 continue_training,
                 num_workers,
                 initial_step,
                 t_train,
                 in_channels,
                 out_channels, 
                 batch_size,
                 unroll_step,
                 ar_mode,
                 pushforward,
                 epochs,
                 learning_rate,
                 scheduler_step,
                 scheduler_gamma,
                 model_update,
                 flnm,
                 single_file,
                 reduced_resolution,
                 reduced_resolution_t,
                 reduced_batch,
                 plot,
                 channel_plot,
                 x_min,
                 x_max,
                 y_min,
                 y_max,
                 t_min,
                 t_max,
                 base_path='../data/',
                 training_type='autoregressive'                
                 ):

    print(f'Epochs = {epochs}, learning rate = {learning_rate}, scheduler step = {scheduler_step}, scheduler gamma = {scheduler_gamma}')
    
    ################################################################
    # load data
    ################################################################
    
    if single_file: 
        # Load data from my own path
        """ Double check the path either to ReactionDiffusion or Burgers """
        if flnm.split('_')[0] == 'ReacDiff':
            base_path_me  = "../PDEBench/pdebench/data_download/pdebench/data/1D/ReactionDiffusion/Train/" # change either to ReactionDiffusion or Burgers
            
        elif flnm.split('_')[0] == 'Burgers':
            base_path_me  = "../PDEBench/pdebench/data_download/pdebench/data/1D/Burgers/Train/" # change either to ReactionDiffusion or Burgers
            
        else:
            raise ValueError(f"flnm: {flnm} is not a valid file name")
        
        pre_trained_base_me = "../PDEBench/pdebench/models/pre-trained-models/unet/"
        # filename
        model_name = flnm[:-5] + '_Unet'
    
        # Initialize the dataset and dataloader
        train_data = UNetDatasetSingle(flnm,
                                saved_folder= base_path_me, #base_path,
                                reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step)
        val_data = UNetDatasetSingle(flnm,
                              saved_folder=base_path_me, #base_path,
                              reduced_resolution=reduced_resolution,
                              reduced_resolution_t=reduced_resolution_t,
                              reduced_batch=reduced_batch,
                              initial_step=initial_step,
                              if_test=True)
        
    else:
        # filename
        model_name = flnm + '_Unet'
    
        train_data = UNetDatasetMult(flnm,
                                reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                saved_folder=base_path)
        val_data = UNetDatasetMult(flnm,
                              reduced_resolution=reduced_resolution,
                              reduced_resolution_t=reduced_resolution_t, 
                              reduced_batch=reduced_batch,
                              if_test=True,
                              saved_folder=base_path)

    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    #                                            num_workers=num_workers, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
    #                                          num_workers=num_workers, shuffle=False)   

    # # t_train should be 21 too,
    
    """ Reaction Diffusion """
    train_loader = torch.load("save_data/reaction/train_0.5_bs50.pth") # train_5_10
    val_loader = torch.load("save_data/reaction/val_0.5_bs50.pth") # val_5_10

    """ Burgers """
    # train_loader = torch.load('save_data/burgers/train_0.01_32.pth')
    # val_loader = torch.load('save_data/burgers/val_0.01_32.pth')
    
    print(f"train_loader: {len(train_loader)}, val_loader: {len(val_loader)}")


    # batch = next(iter(val_loader))

    # # Print detailed information about the batch
    # print("\n=== Batch Structure Investigation ===")
    # print(f"Type of batch: {type(batch)}")
    # print(f"Length of batch (number of elements): {len(batch)}")
    # print("\nContents of batch:")
    # for i, item in enumerate(batch):
    #     print(f"\nElement {i}:")
    #     print(f"Type: {type(item)}")
    #     if torch.is_tensor(item):
    #         print(f"Shape: {item.shape}")
    #         print(f"Data type: {item.dtype}")
    #         print(f"Device: {item.device}")
    #     else:
    #         print(f"Value: {item}")

    # # Try unpacking to see what happens
    # try:
    #     _, _data, _, _ = batch
    #     print("\nUnpacking successful!")
    #     print(f"Shape of _data: {_data.shape if torch.is_tensor(_data) else 'Not a tensor'}")
    # except ValueError as e:
    #     print(f"\nUnpacking failed: {e}")

    # sys.exit()

    ################################################################
    # training and evaluation
    ################################################################
    
    #model = UNet2d(in_channels, out_channels).to(device)

    """ Start comment """
    #_, _data, _, _ = next(iter(val_loader)) # shape of _data: torch.Size([50, 256, 101, 1])
    _, _data, _, _ = next(iter(val_loader))
    first_batch = next(iter(train_loader)) 
    print(f"shape of first_batch: {type(first_batch)}")
    print(f"shape of _data: {_data.shape}")
    # sys.exit()
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)
    if training_type in ['autoregressive']:
        if dimensions == 4:
            model = UNet1d(in_channels*initial_step, out_channels).to(device) # U-Net receives the feature + the initial step values
        elif dimensions == 5:
            model = UNet2d(in_channels*initial_step, out_channels).to(device)
        elif dimensions == 6:
            model = UNet3d(in_channels*initial_step, out_channels).to(device)
    if training_type in ['single']:
        if dimensions == 4:
            model = UNet1d(in_channels, out_channels).to(device)
        elif dimensions == 5:
            model = UNet2d(in_channels, out_channels).to(device)
        elif dimensions == 6:
            model = UNet3d(in_channels, out_channels).to(device)
    
    # Set maximum time step of the data to train
    if t_train > _data.shape[-2]:
        t_train = _data.shape[-2]
    # Set maximum of unrolled time step for the pushforward trick
    if t_train - unroll_step < 1:
        unroll_step = t_train - 1

    if training_type in ['autoregressive']:
        if ar_mode:
            if pushforward:
                model_name = model_name + '-PF-' + str(unroll_step)
            if not pushforward:
                unroll_step = _data.shape[-2]
                model_name = model_name + '-AR'
        else:
            model_name = model_name + '-1-step'

    """ End comment """
        
    model_path = pre_trained_base_me + model_name + ".pt"
    pre_trained_model = os.path.abspath(model_path)
    
    if not os.path.exists(pre_trained_base_me):
        print("The Path Does not Exist, creating the path for the model training...")
        os.makedirs(pre_trained_base_me, exist_ok=True)
    else:
        print("The Path Exists")
        
    
    #sys.exit()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')
    #print(f'if training value: {if_training}, continue training value: {continue_training}, initial step value: {initial_step}, t_train value: {t_train}, unroll step value: {unroll_step}, ar_mode value: {ar_mode}, pushforward value: {pushforward}, scheduler step value: {scheduler_step}, scheduler gamma value: {scheduler_gamma}, model update value: {model_update}, flnm value: {flnm}, single_file value: {single_file}, reduced_resolution value: {reduced_resolution}, reduced_resolution_t value: {reduced_resolution_t}, reduced_batch value: {reduced_batch}, plot value: {plot}, channel_plot value: {channel_plot}, x_min value: {x_min}, x_max value: {x_max}, y_min value: {y_min}, y_max value: {y_max}, t_min value: {t_min}, t_max value: {t_max}, base_path value: {base_path}, training_type value: {training_type}')
    print(f'plot value: {plot}, batch size value: {batch_size}')
    #sys.exit()  # counting the number of parameters.
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    
    loss_fn = nn.MSELoss(reduction="mean")
    loss_val_min = np.infty
    
    start_epoch = 0
    # print(f"if training value: {if_training}")
    # sys.exit()
    
    if not if_training:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"The Pre-Trained model")
        print(model)
        print(f"Initial step: {initial_step}") # The UNet-1D takes 10 as the initial step but my appraoch takes 1 as initial step. 
        #sys.exit()
        Lx, Ly, Lz = 1., 1., 1.
        # updating x_min, x_max, y_min, y_max, t_min, t_max to suit our problem setup
        x_min, x_max, y_min, y_max, t_min, t_max = 0., 1., 0, 1, 0., 2.

        errs = metrics(val_loader, model, Lx, Ly, Lz, plot, channel_plot, 
                       model_name, x_min, x_max, y_min, y_max,
                       t_min, t_max, mode='Unet', initial_step=initial_step)
        pickle.dump(errs, open(model_name+'.pickle', "wb"))
            
        return

   
    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if continue_training:
        print('Restoring model (that is the network\'s weights) from file...')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.train()
        
        # Load optimizer state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
        start_epoch = checkpoint['epoch']
        loss_val_min = checkpoint['loss']

    print('start training...')
    print(f"This is the value of ar_mode : {ar_mode}")
    # change training type to single
    #training_type = "single"
    #sys.exit()
    if ar_mode:
        #print(f"This is the the autoregressive mode or pushforward training and training type is : {training_type}") # autoregressive
        #sys.exit()
        # print(f"This is the value of t_train: {t_train}")
        # sys.exit()
        for ep in range(start_epoch, epochs):
            model.train()
            t1 = default_timer()
            train_l2_step = 0
            train_l2_full = 0
            
            for xx, yy in train_loader:
                loss = 0
                
                # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
                # yy: target tensor [b, x1, ..., xd, t, v]
                # grid: meshgrid [b, x1, ..., xd, dims]
                xx = xx.to(device)
                yy = yy.to(device)
                
                #print(f"shape of xx: {xx.shape} and shape of yy: {yy.shape}") #  torch.Size([50, 256, 10, 1])  and  torch.Size([50, 256, 41, 1])

                
                if training_type in ['autoregressive']:

                    # Initialize the prediction tensor
                    pred = yy[..., :initial_step, :] # it takes 10 different different time steps. 
                    # print(f"initialized prediction variable shape: {pred.shape}") # torch.Size([50, 256, 10, 1])
                    # sys.exit()
                    
                    # Extract shape of the input tensor for reshaping (i.e. stacking the
                    # time and channels dimension together)
                    inp_shape = list(xx.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)
                    #print(f"This is the input_shape variable : {inp_shape}") #  [50, 256, -1]
                    #sys.exit()
            
                    # Autoregressive loop
                    for t in range(initial_step, t_train): # initial step 10, t_train = 41,
                        #print(f"initial t value: {t}, t_train: {t_train}, unroll steps: {unroll_step}") # initial t value: 10, t_train: 41, unroll steps: 20
                        #sys.exit()
                        if t < t_train-unroll_step: # if t = 10 < 41 - 20 
                            with torch.no_grad(): # The reason we have this is because it is using Truncating Backpropagation Through Time (TBTT) to save computation and memory 
                                # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                                inp = xx.reshape(inp_shape)
                                temp_shape = [0, -1]
                                temp_shape.extend([i for i in range(1,len(inp.shape)-1)])
                                #print(f"inp shape before permutation: {inp.shape}, temp variable used for permutation shape: {temp_shape}") # inp : torch.Size([50, 256, 10]), temp_shape : [0, -1, 1], [50, 10, 256]
                                inp = inp.permute(temp_shape) # U-net takes as input a matrix  but outputs a single value. 
                                #print(f"inp shape after permutting with temp_shape: {inp.shape}") # torch.Size([50, 10, 256])
                                
                                # Extract target at current time step
                                y = yy[..., t:t+1, :]
                                #print(f"y at the current time step shape: {y.shape}") #  # torch.Size([50, 256, 1, 1])
        
                        
                                # Model run
                                temp_shape = [0]
                                temp_shape.extend([i for i in range(2,len(inp.shape))])
                                temp_shape.append(1)
                                #print(f"inp shape before permutation: {inp.shape}, temp variable used for permutation shape: {temp_shape}") #  input: torch.Size([50, 10, 256]), temp_shape: [0, 2, 1]
                                temp_output_ = model(inp) 
                                #print(f"shape of temp model output: {temp_output_.shape}") # shape of temp model output: torch.Size([50, 1, 256])
                                im = model(inp).permute(temp_shape).unsqueeze(-2)
                                #print(f"the shape of the model output: {im.shape}") # im shape:  torch.Size([50, 256, 1, 1])
                               
                                
                                # Concatenate the prediction at current time step into the
                                # prediction tensor
                                pred = torch.cat((pred, im), -2) # pred shape: torch.Size([50, 256, 10, 1]), im shape:  torch.Size([50, 256, 1, 1])
                                #print(f"pred shape after concatenating pred and model's output along -2 dimension: {pred.shape}") # pred shape: torch.Size([50, 256, 11, 1])
                    
                                # Concatenate the prediction at the current time step to be used
                                # as input for the next time step
                                xx = torch.cat((xx[..., 1:, :], im), dim=-2)
                                #print(f"xx shape as the next input: {xx.shape}") # xx shape:  torch.Size([50, 256, 10, 1])
                                #sys.exit()
                        else:
                            # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                            inp = xx.reshape(inp_shape)
                            temp_shape = [0, -1]
                            temp_shape.extend([i for i in range(1,len(inp.shape)-1)])
                            inp = inp.permute(temp_shape) # input: torch.Size([50, 10, 256]),
                            
                            # Extract target at current time step
                            y = yy[..., t:t+1, :] #  # torch.Size([50, 256, 1, 1])
                        
                            # Model run
                            temp_shape = [0]
                            temp_shape.extend([i for i in range(2,len(inp.shape))])
                            temp_shape.append(1)
                            im = model(inp).permute(temp_shape).unsqueeze(-2) # torch.Size([50, 256, 1, 1])
                            
                            # Loss calculation
                            loss += loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                
                            # Concatenate the prediction at current time step into the
                            # prediction tensor
                            pred = torch.cat((pred, im), -2)
                
                            # Concatenate the prediction at the current time step to be used
                            # as input for the next time step
                            xx = torch.cat((xx[..., 1:, :], im), dim=-2)
                                
                    train_l2_step += loss.item()
                    _batch = yy.size(0)
                    _yy = yy[..., :t_train, :]
                    #print(f"pred shape: {pred.shape}, _yy shape: {_yy.shape}")
                    #sys.exit()
                    l2_full = loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
                    train_l2_full += l2_full.item()
            
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if training_type in ['single']:
                print("Single training type being used...")
                x = xx[..., 0 , :]
                y = yy[..., t_train-1:t_train, :]
                #print(f"xx: {xx.shape}, x: {x.shape}, yy: {yy.shape}, y: {y.shape} ") # xx: torch.Size([50, 256, 10, 1]), x: torch.Size([50, 256, 1]), yy: torch.Size([50, 256, 41, 1]), y: torch.Size([50, 256, 1, 1]) 
                #sys.exit()
                pred = model(x.permute([0, 2, 1])).permute([0, 2, 1])
                _batch = yy.size(0)
                loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))
    
                train_l2_step += loss.item()
                train_l2_full += loss.item()
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            
            if ep % model_update == 0:
                val_l2_step = 0
                val_l2_full = 0
                with torch.no_grad():
                    for xx, yy in val_loader:
                        loss = 0
                        xx = xx.to(device)
                        yy = yy.to(device)
                        
                        if training_type in ['autoregressive']:
                            pred = yy[..., :initial_step, :]
                            inp_shape = list(xx.shape)
                            inp_shape = inp_shape[:-2]
                            inp_shape.append(-1)
                    
                            for t in range(initial_step, t_train):
                                inp = xx.reshape(inp_shape)
                                temp_shape = [0, -1]
                                temp_shape.extend([i for i in range(1,len(inp.shape)-1)])
                                inp = inp.permute(temp_shape)
                                y = yy[..., t:t+1, :]
                                temp_shape = [0]
                                temp_shape.extend([i for i in range(2,len(inp.shape))])
                                temp_shape.append(1)
                                im = model(inp).permute(temp_shape).unsqueeze(-2)
                                loss += loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                    
                                pred = torch.cat((pred, im), -2)
                    
                                xx = torch.cat((xx[..., 1:, :], im), dim=-2)
                
                            val_l2_step += loss.item()
                            _batch = yy.size(0)
                            _pred = pred[..., initial_step:t_train, :]
                            _yy = yy[..., initial_step:t_train, :]
                            #print(f"pred shape: {pred.shape}, _yy shape: {_yy.shape}")
                            #sys.exit()
                            val_l2_full += loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                    
                    if training_type in ['single']:
                        x = xx[..., 0 , :]
                        y = yy[..., t_train-1:t_train, :]
                        #print(f"x : {x.shape}, y : {y.shape}")
                        #sys.exit()
                        pred = model(x.permute([0, 2, 1])).permute([0, 2, 1])
                        _batch = yy.size(0)
                        loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))
            
                        val_l2_step += loss.item()
                        val_l2_full += loss.item()

                    if  val_l2_full < loss_val_min:
                        loss_val_min = val_l2_full
                        torch.save({
                            'epoch': ep,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss_val_min
                            }, model_path)
     
            t2 = default_timer()
            #print("Done, Success")
            #sys.exit()
            scheduler.step()
            print('epoch: {0}, loss: {1:.5f}, t2-t1: {2:.5f}, trainL2: {3:.5f}, testL2: {4:.5f}'\
                    .format(ep, loss.item(), t2 - t1, train_l2_step, val_l2_step))

    else:
        for ep in range(start_epoch, epochs):
            model.train()
            t1 = default_timer()
            train_l2_step = 0
            train_l2_full = 0
            
            for xx, yy in train_loader:
                loss = 0
                
                # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
                # yy: target tensor [b, x1, ..., xd, t, v]
                xx = xx.to(device)
                yy = yy.to(device)
                
                # Initialize the prediction tensor
                pred = yy[..., :initial_step, :]
                
                # Extract shape of the input tensor for reshaping (i.e. stacking the
                # time and channels dimension together)
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)
        
                # Autoregressive loop
                for t in range(initial_step, t_train):
                                        
                    # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                    inp = yy[..., t-initial_step:t, :].reshape(inp_shape)
                    temp_shape = [0, -1]
                    temp_shape.extend([i for i in range(1,len(inp.shape)-1)])
                    inp = inp.permute(temp_shape)
                    inp = torch.normal(inp, 0.001)
                    
                    # Extract target at current time step
                    y = yy[..., t:t+1, :]
                
                    # Model run
                    temp_shape = [0]
                    temp_shape.extend([i for i in range(2,len(inp.shape))])
                    temp_shape.append(1)
                    im = model(inp).permute(temp_shape).unsqueeze(-2)
                    
                    # Loss calculation
                    loss += loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
        
                    # Concatenate the prediction at current time step into the
                    # prediction tensor
                    pred = torch.cat((pred, im), -2)
        
                    # Concatenate the prediction at the current time step to be used
                    # as input for the next time step
                    # xx = torch.cat((xx[..., 1:, :], im), dim=-2)
        
                train_l2_step += loss.item()
                _batch = yy.size(0)
                _yy = yy[..., :t_train, :]  # if t_train is not -1
                l2_full = loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
                train_l2_full += l2_full.item()
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if ep % model_update == 0 or ep == epochs:
                val_l2_step = 0
                val_l2_full = 0
                with torch.no_grad():
                    for xx, yy in val_loader:
                        loss = 0
                        xx = xx.to(device)
                        yy = yy.to(device)
                        
                        pred = yy[..., :initial_step, :]
                        inp_shape = list(xx.shape)
                        inp_shape = inp_shape[:-2]
                        inp_shape.append(-1)
                
                        for t in range(initial_step, t_train):
                            inp = yy[..., t-initial_step:t, :].reshape(inp_shape)
                            temp_shape = [0, -1]
                            temp_shape.extend([i for i in range(1,len(inp.shape)-1)])
                            inp = inp.permute(temp_shape)
                            y = yy[..., t:t+1, :]
                            temp_shape = [0]
                            temp_shape.extend([i for i in range(2,len(inp.shape))])
                            temp_shape.append(1)
                            im = model(inp).permute(temp_shape).unsqueeze(-2)
                            loss += loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                
                            pred = torch.cat((pred, im), -2)
            
                        val_l2_step += loss.item()
                        _batch = yy.size(0)
                        _pred = pred[..., initial_step:t_train, :]
                        _yy = yy[..., initial_step:t_train, :]  # if t_train is not -1
                        val_l2_full += loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                    
                    if  val_l2_full < loss_val_min:
                        loss_val_min = val_l2_full
                        torch.save({
                            'epoch': ep,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss_val_min
                            }, model_path)
                    
                
            t2 = default_timer()
            scheduler.step()
            print('epoch: {0}, loss: {1:.5f}, t2-t1: {2:.5f}, trainL2: {3:.5f}, testL2: {4:.5f}'\
                    .format(ep, loss.item(), t2 - t1, train_l2_step, val_l2_step))


if __name__ == "__main__":
    
    run_training()  
    print("Done.")