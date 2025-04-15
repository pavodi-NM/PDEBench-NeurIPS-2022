"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os, sys
import torch

from typing import Tuple

from pdebench.models.pinn.utils import (
    PINNDatasetRadialDambreak,
    PINNDatasetDiffReact,
    PINNDataset2D,
    PINNDatasetDiffSorption,
    PINNDatasetBump,
    PINNDataset1Dpde,
    PINNDataset2Dpde,
    PINNDataset3Dpde,
)
from pdebench.models.pinn.pde_definitions import (
    pde_diffusion_reaction,
    pde_swe2d,
    pde_diffusion_sorption,
    pde_swe1d,
    pde_adv1d,
    pde_diffusion_reaction_1d,
    pde_burgers1D,
    pde_CFD1d,
    pde_CFD2d,
    pde_CFD3d,
)

from pdebench.models.metrics import metrics, metric_func


def setup_diffusion_sorption(filename, seed):
    # TODO: read from dataset config file
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 500.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    D = 5e-4

    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    bc_d = dde.icbc.DirichletBC(
        geomtime,
        lambda x: 1.0,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0),
    )

    def operator_bc(inputs, outputs, X):
        # compute u_t
        du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        return outputs - D * du_x

    bc_d2 = dde.icbc.OperatorBC(
        geomtime,
        operator_bc,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 1.0),
    )

    dataset = PINNDatasetDiffSorption(filename, seed)

    ratio = int(len(dataset) * 0.3)

    data_split, _ = torch.utils.data.random_split(
        dataset,
        [ratio, len(dataset) - ratio],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    data_gt = data_split[:]

    bc_data = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1])

    data = dde.data.TimePDE(
        geomtime,
        pde_diffusion_sorption,
        [ic, bc_d, bc_d2, bc_data],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([2] + [40] * 6 + [1], "tanh", "Glorot normal")

    def transform_output(x, y):
        return torch.relu(y)

    net.apply_output_transform(transform_output)

    model = dde.Model(data, net)

    return model, dataset

def setup_diffusion_reaction(filename, seed):
    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((-1, -1), (1, 1))
    timedomain = dde.geometry.TimeDomain(0, 5.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

    dataset = PINNDatasetDiffReact(filename, seed)
    initial_input, initial_u, initial_v = dataset.get_initial_condition()

    ic_data_u = dde.icbc.PointSetBC(initial_input, initial_u, component=0)
    ic_data_v = dde.icbc.PointSetBC(initial_input, initial_v, component=1)

    ratio = int(len(dataset) * 0.3)

    data_split, _ = torch.utils.data.random_split(
        dataset,
        [ratio, len(dataset) - ratio],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    data_gt = data_split[:]

    bc_data_u = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1], component=0)
    bc_data_v = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[2], component=1)

    data = dde.data.TimePDE(
        geomtime,
        pde_diffusion_reaction,
        [bc, ic_data_u, ic_data_v, bc_data_u, bc_data_v],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([3] + [40] * 6 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset


def setup_swe_2d(filename, seed) -> Tuple[dde.Model, PINNDataset2D]:

    dataset = PINNDatasetRadialDambreak(filename, seed)

    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((-2.5, -2.5), (2.5, 2.5))
    timedomain = dde.geometry.TimeDomain(0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic_h = dde.icbc.IC(
        geomtime,
        dataset.get_initial_condition_func(),
        lambda _, on_initial: on_initial,
        component=0,
    )
    ic_u = dde.icbc.IC(
        geomtime, lambda x: 0.0, lambda _, on_initial: on_initial, component=1
    )
    ic_v = dde.icbc.IC(
        geomtime, lambda x: 0.0, lambda _, on_initial: on_initial, component=2
    )

    ratio = int(len(dataset) * 0.3)

    data_split, _ = torch.utils.data.random_split(
        dataset,
        [ratio, len(dataset) - ratio],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    data_gt = data_split[:]

    bc_data = dde.icbc.PointSetBC(data_gt[0].cpu(), data_gt[1], component=0)

    data = dde.data.TimePDE(
        geomtime,
        pde_swe2d,
        [bc, ic_h, ic_u, ic_v, bc_data],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([3] + [40] * 6 + [3], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset

def _boundary_r(x, on_boundary, xL, xR):
    return (on_boundary and np.isclose(x[0], xL)) or (on_boundary and np.isclose(x[0], xR))

def setup_pde1D(filename="1D_Advection_Sols_beta0.1.hdf5",
                root_path='data',
                val_batch_idx=-1,
                input_ch=2,
                output_ch=1,
                hidden_ch=40,
                xL=0.,
                xR=1.,
                if_periodic_bc=True,
                aux_params=[0.1]):

    print(f"setting up pde 1D")
    #sys.exit()
    
    # TODO: read from dataset config file
    geom = dde.geometry.Interval(xL, xR)
    boundary_r = lambda x, on_boundary: _boundary_r(x, on_boundary, xL, xR)
    if filename[0] == 'R':
        timedomain = dde.geometry.TimeDomain(0, 1.0)
        pde = lambda x, y : pde_diffusion_reaction_1d(x, y, aux_params[0], aux_params[1])
    else:
        if filename.split('_')[1][0]=='A':
            timedomain = dde.geometry.TimeDomain(0, 2.0)
            pde = lambda x, y: pde_adv1d(x, y, aux_params[0])
        elif filename.split('_')[1][0] == 'B':
            timedomain = dde.geometry.TimeDomain(0, 2.0)
            # Debug the viscosity parameter
            nu_value = float(aux_params[0]) if aux_params[0] is not None else 0.001
            print(f"Using viscosity parameter: {nu_value} (from {aux_params[0]})")
            pde = lambda x, y: pde_burgers1D(x, y, nu_value)
        elif filename.split('_')[1][0]=='C':
            timedomain = dde.geometry.TimeDomain(0, 1.0)
            pde = lambda x, y: pde_CFD1d(x, y, aux_params[0])
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    dataset = PINNDataset1Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
    print(f"Data set loaded successfully: {dataset}")
    #sys.exit()
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    #print(f"initial input: {initial_input.shape}, initial_u: {initial_u.shape}")
    if filename.split('_')[1][0] == 'C':
        ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[:,0].unsqueeze(1), component=0)
        ic_data_v = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[:,1].unsqueeze(1), component=1)
        ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[:,2].unsqueeze(1), component=2)
    else:
        ic_data_u = dde.icbc.PointSetBC(initial_input.cpu(), initial_u, component=0)
    # prepare boundary condition
    if if_periodic_bc:
        if filename.split('_')[1][0] == 'C':
            print(f"in C")
            bc_D = dde.icbc.PeriodicBC(geomtime, 0, boundary_r)
            bc_V = dde.icbc.PeriodicBC(geomtime, 1, boundary_r)
            bc_P = dde.icbc.PeriodicBC(geomtime, 2, boundary_r)

            data = dde.data.TimePDE(
                geomtime,
                pde,
                [ic_data_d, ic_data_v, ic_data_p, bc_D, bc_V, bc_P],
                num_domain=1000,
                num_boundary=1000,
                num_initial=5000,
            )
        else:
            print(f"in BC")
            bc = dde.icbc.PeriodicBC(geomtime, 0, boundary_r)
            data = dde.data.TimePDE(
                geomtime,
                pde,
                [ic_data_u, bc],
                num_domain=1000,
                num_boundary=1000,
                num_initial=5000,
            )
    else:
        ic = dde.icbc.IC(
            geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
        )
        bd_input, bd_uL, bd_uR = dataset.get_boundary_condition()
        bc_data_uL = dde.icbc.PointSetBC(bd_input.cpu(), bd_uL, component=0)
        bc_data_uR = dde.icbc.PointSetBC(bd_input.cpu(), bd_uR, component=0)
        print(f"boundary input: {bd_input.shape}")
        data = dde.data.TimePDE(
            geomtime,
            pde,
            [ic, bc_data_uL, bc_data_uR],
            num_domain=1000,
            num_boundary=1000,
            num_initial=5000,
        )
    
    print(f"Data: {data}, pde: {pde}")
    
    # Print debug information
    print(f"Input dimension needs to be: {initial_input.shape[1]}")
    
    # Create the network with correct input dimension
    # For Burgers equation with DeepXDE 1.12.1
    if filename.split('_')[1][0] == 'B':
        # For TimePDE, DeepXDE expects the input dimension to match
        # the input data shape (which is x and t for 1D PDEs)
        net = dde.nn.FNN([initial_input.shape[1]] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
        print(f"Creating Burgers NN with input shape: {initial_input.shape[1]}")
    else:
        # Original code
        net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
    
    model = dde.Model(data, net)

    return model, dataset

def setup_CFD2D(filename="2D_CFD_RAND_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5",
                root_path='data',
                val_batch_idx=-1,
                input_ch=2,
                output_ch=4,
                hidden_ch=40,
                xL=0.,
                xR=1.,
                yL=0.,
                yR=1.,
                if_periodic_bc=True,
                aux_params=[1.6667]):

    # TODO: read from dataset config file
    geom = dde.geometry.Rectangle((-1, -1), (1, 1))
    timedomain = dde.geometry.TimeDomain(0., 1.0)
    pde = lambda x, y: pde_CFD2d(x, y, aux_params[0])
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    dataset = PINNDataset2Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,0].unsqueeze(1), component=0)
    ic_data_vx = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,1].unsqueeze(1), component=1)
    ic_data_vy = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,2].unsqueeze(1), component=2)
    ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,3].unsqueeze(1), component=3)
    # prepare boundary condition
    bc = dde.icbc.PeriodicBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_data_d, ic_data_vx, ic_data_vy, ic_data_p],#, bc],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset

def setup_CFD3D(filename="3D_CFD_RAND_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5",
                root_path='data',
                val_batch_idx=-1,
                input_ch=2,
                output_ch=4,
                hidden_ch=40,
                aux_params=[1.6667]):

    # TODO: read from dataset config file
    geom = dde.geometry.Cuboid((0., 0., 0.), (1., 1., 1.))
    timedomain = dde.geometry.TimeDomain(0., 1.0)
    pde = lambda x, y: pde_CFD2d(x, y, aux_params[0])
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    dataset = PINNDataset3Dpde(filename, root_path=root_path, val_batch_idx=val_batch_idx)
    # prepare initial condition
    initial_input, initial_u = dataset.get_initial_condition()
    ic_data_d = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,0].unsqueeze(1), component=0)
    ic_data_vx = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,1].unsqueeze(1), component=1)
    ic_data_vy = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,2].unsqueeze(1), component=2)
    ic_data_vz = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,3].unsqueeze(1), component=3)
    ic_data_p = dde.icbc.PointSetBC(initial_input.cpu(), initial_u[...,4].unsqueeze(1), component=4)
    # prepare boundary condition
    bc = dde.icbc.PeriodicBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_data_d, ic_data_vx, ic_data_vy, ic_data_vz, ic_data_p, bc],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )
    net = dde.nn.FNN([input_ch] + [hidden_ch] * 6 + [output_ch], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    return model, dataset

def _run_training(scenario, epochs, learning_rate, model_update, flnm,
                  input_ch, output_ch,
                  root_path, val_batch_idx, if_periodic_bc, aux_params,
                  if_single_run,
                  seed):
    if scenario == "swe2d":
        model, dataset = setup_swe_2d(filename=flnm, seed=seed)
        n_components = 1
    elif scenario == "diff-react":
        model, dataset = setup_diffusion_reaction(filename=flnm, seed=seed)
        n_components = 2
    elif scenario == "diff-sorp":
        model, dataset = setup_diffusion_sorption(filename=flnm, seed=seed)
        n_components = 1
    elif scenario == "pde1D":
        model, dataset = setup_pde1D(filename=flnm,
                                     root_path=root_path,
                                     input_ch=input_ch,
                                     output_ch=output_ch,
                                     val_batch_idx=val_batch_idx,
                                     if_periodic_bc=if_periodic_bc,
                                     aux_params=aux_params)
        if flnm.split('_')[1][0] == 'C':
            n_components = 3
        else:
            n_components = 1
    elif scenario == "CFD2D":
        model, dataset = setup_CFD2D(filename=flnm,
                                     root_path=root_path,
                                     input_ch=input_ch,
                                     output_ch=output_ch,
                                     val_batch_idx=val_batch_idx,
                                     aux_params=aux_params)
        n_components = 4
    elif  scenario == "CFD3D":
        model, dataset = setup_CFD3D(filename=flnm,
                                     root_path=root_path,
                                     input_ch=input_ch,
                                     output_ch=output_ch,
                                     val_batch_idx=val_batch_idx,
                                     aux_params=aux_params)
        n_components = 5
    else:
        raise NotImplementedError(f"PINN training not implemented for {scenario}")

    # filename
    if if_single_run:
        model_name = flnm + "_PINN"
    else:
        model_name = flnm[:-5] + "_PINN"

    checker = dde.callbacks.ModelCheckpoint(
        f"{model_name}.pt", save_better_only=True, period=5000
    )

    model.compile("adam", lr=learning_rate)
    losshistory, train_state = model.train(
        epochs=epochs, display_every=model_update, callbacks=[checker]
    )

    test_input, test_gt = dataset.get_test_data(
        n_last_time_steps=20, n_components=n_components
    )
    # select only n_components of output
    # dirty hack for swe2d where we predict more components than we have data on
    test_pred = torch.tensor(model.predict(test_input.cpu())[:, :n_components])

    # prepare data for metrics eval
    test_pred = dataset.unravel_tensor(
        test_pred, n_last_time_steps=20, n_components=n_components
    )
    test_gt = dataset.unravel_tensor(
        test_gt, n_last_time_steps=20, n_components=n_components
    )

    if if_single_run:
        # Calculate metrics
        errs = metric_func(test_pred, test_gt)
        errors = [np.array(err.cpu()) for err in errs]
        print("Metrics:")
        metric_names = ["RMSE", "Normalized RMSE", "RMSE of conserved variables", 
                      "Maximum value of RMS error", "RMSE at boundaries", "RMSE in Fourier space"]
        for i, (name, error) in enumerate(zip(metric_names, errors)):
            print(f"  {name}: {error}")
        print(errors)
        pickle.dump(errors, open(model_name + ".pickle", "wb"))

        try:
            # Safely handle plot generation with CUDA tensor protection
            try:
                if scenario == "pde1D" and filename.split('_')[1][0] == 'B':  # Burgers equation
                    # Create 2D visualization like the FNO/UNet plots
                    print("Creating 2D visualization for Burgers equation")
                    # Get the spatial domain
                    if hasattr(dataset, 'xL') and isinstance(dataset.xL, torch.Tensor) and dataset.xL.is_cuda:
                        dataset.xL = dataset.xL.cpu()
                    if hasattr(dataset, 'xR') and isinstance(dataset.xR, torch.Tensor) and dataset.xR.is_cuda:
                        dataset.xR = dataset.xR.cpu()
                    
                    # Define grid points in space and time
                    x_points = np.linspace(float(dataset.xL), float(dataset.xR), dataset.xdim)
                    t_points = np.linspace(0, 2.0, 100)  # Use 100 time points for smoother visualization
                    
                    # Create meshgrid
                    X, T = np.meshgrid(x_points, t_points)
                    X_flat = X.flatten()
                    T_flat = T.flatten()
                    
                    # Prepare input for the model
                    input_points = np.vstack((X_flat, T_flat)).T
                    
                    # Make predictions
                    predictions = model.predict(input_points)[:, 0]
                    
                    # Reshape to 2D grid
                    solution_field = predictions.reshape(T.shape)
                    
                    # Plot the solution
                    fig, ax = plt.subplots(figsize=(10, 8))
                    cf = ax.pcolormesh(T, X, solution_field.T, cmap='rainbow', shading='auto')
                    fig.colorbar(cf, ax=ax)
                    ax.set_xlabel('$t$', fontsize=14)
                    ax.set_ylabel('$x$', fontsize=14)
                    ax.set_title('Prediction', fontsize=18)
                    
                    # Save figure
                    plt.savefig(f"{model_name}_prediction.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Also create the traditional 1D slice plot
                    try:
                        # Try to get plot input for a specific time slice
                        plot_input = dataset.generate_plot_input(time=1.0)
                        
                        if plot_input is not None:
                            y_pred = model.predict(plot_input)[:, 0]
                            plt.figure()
                            plt.plot(x_points, y_pred)
                            plt.xlabel('x')
                            plt.ylabel('u')
                            plt.title(f'Solution at t=1.0')
                            plt.savefig(f"{model_name}_slice.png")
                            plt.close()
                    except Exception as e:
                        print(f"Warning: Could not generate 1D slice plot - {str(e)}")
                else:
                    # Original plotting code for other cases
                    # Try to get plot input
                    plot_input = dataset.generate_plot_input(time=1.0)
                    
                    # If tensor attributes are on GPU, move them to CPU first
                    if hasattr(dataset, 'xL') and isinstance(dataset.xL, torch.Tensor) and dataset.xL.is_cuda:
                        dataset.xL = dataset.xL.cpu()
                    if hasattr(dataset, 'xR') and isinstance(dataset.xR, torch.Tensor) and dataset.xR.is_cuda:
                        dataset.xR = dataset.xR.cpu()
                    
                    if scenario == "pde1D":
                        xdim = dataset.xdim
                        dim = 1
                    else:
                        dim = dataset.config["plot"]["dim"]
                        xdim = dataset.config["sim"]["xdim"]
                        if dim == 2:
                            ydim = dataset.config["sim"]["ydim"]

                    if plot_input is not None:
                        y_pred = model.predict(plot_input)[:, 0]
                        if dim == 1:
                            plt.figure()
                            plt.plot(y_pred)
                        elif dim == 2:
                            im_data = y_pred.reshape(xdim, ydim)
                            plt.figure()
                            plt.imshow(im_data)

                        plt.savefig(f"{model_name}.png")
                    plt.close()
            except Exception as e:
                print(f"Warning: Could not generate plot - {str(e)}")
                # Continue execution even if plotting fails
                
        except Exception as e:
            print(f"Error in visualization part: {str(e)}")
            # Continue with the rest of the function
            
        # TODO: implement function to get specific timestep from dataset
        # y_true = dataset[:][1][-xdim * ydim :]
    else:
        return test_pred, test_gt, model_name

def run_training(scenario, epochs, learning_rate, model_update, flnm,
                 input_ch=1, output_ch=1,
                 root_path='../data/', val_num=10, if_periodic_bc=True, aux_params=[None], seed='0000'):
    print(val_num, scenario)
    root_path =  "../PDEBench/pdebench/data_download/pdebench/data/1D/Burgers/Train/"
    flnm = "1D_Burgers_Sols_Nu0.001.hdf5"
    #print(f"flnm: {flnm}")
    #sys.exit()

    if val_num == 1:  # single job
        _run_training(scenario, epochs, learning_rate, model_update, flnm,
                      input_ch, output_ch,
                      root_path, -val_num, if_periodic_bc, aux_params,
                      if_single_run=True, seed=seed)
    else:
        for val_batch_idx in range(-1, -val_num, -1):
            test_pred, test_gt, model_name = _run_training(scenario, epochs, learning_rate, model_update, flnm,
                                                           input_ch, output_ch,
                                                           root_path, val_batch_idx, if_periodic_bc, aux_params,
                                                           if_single_run=False, seed=seed)
            if val_batch_idx == -1:
                pred, target = test_pred.unsqueeze(0), test_gt.unsqueeze(0)
            else:
                pred = torch.cat([pred, test_pred.unsqueeze(0)], 0)
                target = torch.cat([target, test_gt.unsqueeze(0)], 0)

        errs = metric_func(test_pred, test_gt)
        errors = [np.array(err.cpu()) for err in errs]
        print(errors)
        pickle.dump(errors, open(model_name + ".pickle", "wb"))
    
    # Generate standard PDF plots like FNO and UNet
    try:
        from pdebench.models.pinn.pinn_visualization import generate_pinn_plots_from_loaded_model
        # Extract nu value from filename if it exists
        nu_value = 0.01  # Default value
        if "Nu" in flnm:
            try:
                nu_value = float(flnm.split("Nu")[1].split("_")[0])
            except:
                pass
        
        # Load the latest model
        model_path = f"{flnm[:-5]}_PINN.pt" if "." in flnm else f"{flnm}_PINN.pt"
        if os.path.exists(model_path):
            model = dde.Model.from_checkpoint(model_path)
            print(f"Generating standardized plots for {model_path}...")
            plot_paths = generate_pinn_plots_from_loaded_model(model, flnm)
            print(f"Plots saved to: {plot_paths['prediction']}")
        else:
            print(f"Warning: Could not find model file {model_path} for visualization")
    except Exception as e:
        print(f"Warning: Could not generate standard PDF plots: {str(e)}")


if __name__ == "__main__":
    # run_training(
    #     scenario="diff-sorp",
    #     epochs=100,
    #     learning_rate=1e-3,
    #     model_update=500,
    #     flnm="2D_diff-sorp_NA_NA_0000.h5",
    #     seed="0000",
    # )
    run_training(
        scenario="diff-react",
        epochs=100,
        learning_rate=1e-3,
        model_update=500,
        flnm="2D_diff-react_NA_NA.h5",
        seed="0000",
    )
    # run_training(
    #     scenario="swe2d",
    #     epochs=100,
    #     learning_rate=1e-3,
    #     model_update=500,
    #     flnm="radial_dam_break_0000.h5",
    #     seed="0000",
    # )
