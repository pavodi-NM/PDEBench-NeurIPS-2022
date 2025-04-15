"""
Visualization utilities for PINN models.

This module provides functions to create and save visualizations of PINN model
predictions in the same style as FNO and UNet models in PDEBench.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import os

def plot_prediction_2d(model, filename, root_path=None, domain_size=(1.0, 2.0), 
                      resolution=(100, 100), viscosity=0.01, 
                      colormap='rainbow', save_format='pdf'):
    """
    Create a 2D visualization of a PINN model's prediction for the Burgers equation.
    
    Args:
        model: The trained PINN model (deepxde model)
        filename: Filename containing the dataset info (used for naming output files)
        root_path: Path to save the output plots (default: current directory)
        domain_size: Tuple (x_size, t_size) specifying the domain size
        resolution: Tuple (nx, nt) specifying the resolution of the visualization grid
        viscosity: Viscosity parameter for the Burgers equation
        colormap: Matplotlib colormap to use for the visualization
        save_format: Format to save the plots ('pdf' or 'png')
        
    Returns:
        Tuple (prediction_path, data_path) with the paths to the saved plots
    """
    # Extract model name from filename
    if isinstance(filename, str):
        model_name = filename.split('.')[0] + "_PINN"
    else:
        model_name = "PINN_model"
    
    # Create output directory if it doesn't exist
    if root_path is not None:
        os.makedirs(root_path, exist_ok=True)
        model_name = os.path.join(root_path, model_name)
    
    # Set up the domain with the correct ranges
    x_size, t_size = domain_size
    nx, nt = resolution
    x_min, x_max = 0.0, x_size
    t_min, t_max = 0.0, t_size
    
    # Create meshgrid for the domain
    x_points = np.linspace(x_min, x_max, nx)
    t_points = np.linspace(t_min, t_max, nt)
    
    # Create meshgrid with proper dimensions
    # For PINN, inputs are expected as (spatial_dim, time_dim)
    X, T = np.meshgrid(x_points, t_points)  
    
    # Print meshgrid shapes for debugging
    print(f"X meshgrid shape: {X.shape}, X range: [{X.min()}, {X.max()}]")
    print(f"T meshgrid shape: {T.shape}, T range: [{T.min()}, {T.max()}]")
    
    # Flatten the meshgrid for input to the model
    X_flat = X.flatten()
    T_flat = T.flatten()
    
    # Prepare input for the model - first dimension is x (space), second is t (time)
    input_points = np.vstack((X_flat, T_flat)).T
    print(f"Input points shape: {input_points.shape}, first few points: {input_points[:5]}")
    
    # Run the model to get predictions
    try:
        print(f"Running prediction with input shape: {input_points.shape}")
        if hasattr(model, 'predict') and callable(model.predict):
            predictions = model.predict(input_points)
            if predictions is not None:
                print(f"Raw prediction shape: {predictions.shape}")
                predictions = predictions[:, 0]
                print(f"Processed prediction shape: {predictions.shape}")
            else:
                raise ValueError("Model prediction returned None")
        else:
            raise AttributeError("Model has no callable predict method")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {dir(model)}")
        raise
    
    # Reshape to 2D grid and swap axes for correct orientation
    solution_field = predictions.reshape(T.shape)
    print(f"Solution field shape: {solution_field.shape}")
    print(f"X shape: {X.shape}, T shape: {T.shape}")
    
    # Create the prediction plot
    plt.ioff()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    
    # For PDEBench format, we need t on x-axis and x on y-axis
    # So we transpose the solution field and use t_min/t_max for x extent
    # and x_min/x_max for y extent
    h = ax.imshow(solution_field.T, 
                 extent=[t_min, t_max, x_min, x_max], 
                 origin='lower', 
                 aspect='auto',
                 cmap=colormap)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=30)
    
    # Set up labels and title to match other PDEBench plots
    ax.set_title("Prediction", fontsize=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.set_xlabel("$t$", fontsize=30)  # t is on x-axis
    ax.set_ylabel("$x$", fontsize=30)  # x is on y-axis
    
    plt.tight_layout()
    prediction_path = f"{model_name}_pred.{save_format}"
    plt.savefig(prediction_path)
    plt.close()
    
    # Add information about the model and parameters to a text file
    info_text = f"""PINN Model: {model_name}
Viscosity: {viscosity}
Domain: x ∈ [0, {x_size}], t ∈ [0, {t_size}]
Resolution: {nx} x {nt}
"""
    
    with open(f"{model_name}_info.txt", "w") as f:
        f.write(info_text)
    
    return prediction_path

def visualize_pinn_model(model_path, filename, root_path=None, domain_size=(1.0, 2.0),
                        resolution=(100, 100), viscosity=0.01, plot_format='pdf'):
    """
    Load a saved PINN model and create visualizations.
    
    Args:
        model_path: Path to the saved PINN model file (.pt)
        filename: Original dataset filename used for training
        root_path: Path to save the output plots
        domain_size: Tuple (x_size, t_size) specifying the domain size
        resolution: Tuple (nx, nt) specifying the resolution of the visualization grid
        viscosity: Viscosity parameter for the Burgers equation
        plot_format: Format to save the plots ('pdf' or 'png')
        
    Returns:
        Paths to the saved visualization files
    """
    import deepxde as dde
    import torch
    
    # Check DeepXDE version and load model accordingly
    try:
        # Try to load the model
        print(f"Loading model from {model_path}...")
        
        # For DeepXDE 1.12.1, we need to recreate the model and load state
        # First determine the model architecture
        if "Burgers" in filename:
            # Setup for Burgers equation
            geom = dde.geometry.Interval(0, domain_size[0])
            timedomain = dde.geometry.TimeDomain(0, domain_size[1])
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)
            
            # Create a PDE problem
            def pde_burgers(x, y, nu=viscosity):
                dy_x = dde.grad.jacobian(y, x, i=0, j=0)
                dy_t = dde.grad.jacobian(y, x, i=0, j=1)
                dy_xx = dde.grad.hessian(y, x, i=0, j=0)
                return dy_t + y * dy_x - nu / np.pi * dy_xx
            
            pde = pde_burgers
            
            # Define boundary conditions (simplified for visualization)
            # For Burgers equation we need both IC and BC
            def initial_condition(x):
                return -np.sin(np.pi * x[:, 0:1])
            
            ic = dde.icbc.IC(geomtime, initial_condition, lambda _, on_initial: on_initial)
            bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
            
            # Create a PDE problem - include both ic and bc
            data = dde.data.TimePDE(geomtime, pde, [ic, bc], num_domain=100)
            
            # Create a model with same architecture as training
            net = dde.nn.FNN([2] + [40] * 6 + [1], "tanh", "Glorot normal")
            model = dde.Model(data, net)
            
            # Print model architecture information
            print(f"Model architecture: {type(net)}")
            try:
                # In newer versions of DeepXDE these might be accessible directly
                print(f"Network layers: [2] + [40] * 6 + [1]")
                print(f"Activation: tanh")
            except Exception as e:
                print(f"Could not print detailed model architecture: {str(e)}")
            
            # Ensure model is compiled before loading weights
            model.compile("adam", lr=0.001, metrics=["l2 relative error"])
            
            # Load weights from saved model
            try:
                # Try loading with standard PyTorch approach
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                
                # Check if the checkpoint contains a dictionary with 'model_state_dict' key
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.net.load_state_dict(checkpoint["model_state_dict"])
                    print("Model loaded successfully using model_state_dict from checkpoint")
                else:
                    # Try direct loading if it's just the model state
                    model.net.load_state_dict(checkpoint)
                    print("Model loaded successfully using PyTorch load_state_dict")
                
                # The model needs to be compiled after loading weights
                model.compile("adam", lr=0.001, metrics=["l2 relative error"])
                print("Model compiled successfully")
            except Exception as e1:
                # If that fails, try DeepXDE's restore method if available
                try:
                    model.restore(model_path)
                    print("Model loaded successfully using DeepXDE restore method")
                except Exception as e2:
                    print(f"Could not load model weights: {str(e1)}, {str(e2)}")
                    raise
        else:
            # For other equation types, we would define different architectures
            # Just raising an error for now
            raise NotImplementedError(f"Visualization for {filename} is not implemented yet")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    # Generate and save visualization
    prediction_path = plot_prediction_2d(
        model, 
        filename, 
        root_path=root_path,
        domain_size=domain_size,
        resolution=resolution,
        viscosity=viscosity,
        save_format=plot_format
    )
    
    return prediction_path

def generate_pinn_plots_from_loaded_model(model, flnm, save_dir=None, domain_size=(1.0, 2.0), resolution=(100, 100), viscosity=None, plot_format='pdf'):
    """
    Generate standard plots for a loaded PINN model.
    
    Args:
        model: A trained DeepXDE model
        flnm: The dataset filename (used for naming)
        save_dir: Directory to save the plots (default: current directory)
        domain_size: Tuple (x_size, t_size) specifying the domain size
        resolution: Tuple (nx, nt) specifying the resolution of the visualization grid
        viscosity: Viscosity parameter for the Burgers equation (will be extracted from filename if None)
        plot_format: Format to save the plots ('pdf' or 'png')
        
    Returns:
        Dictionary with paths to the created visualization files
    """
    # Extract viscosity parameter from filename if not provided
    if viscosity is None:
        viscosity = 0.01  # Default value
        if "Nu" in flnm:
            try:
                viscosity = float(flnm.split("Nu")[1].split("_")[0].split('.')[0] + '.' + flnm.split("Nu")[1].split("_")[0].split('.')[1])
                print(f"Detected viscosity parameter: {viscosity}")
            except:
                print(f"Could not extract viscosity from filename {flnm}, using default: {viscosity}")
                pass
    
    # Determine domain size based on the problem type if not provided
    if domain_size is None:
        if "Burgers" in flnm:
            domain_size = (1.0, 2.0)  # Typical for Burgers equation in PDEBench
            print(f"Using domain size for Burgers equation: {domain_size}")
        else:
            domain_size = (1.0, 1.0)  # Default domain size
            print(f"Using default domain size: {domain_size}")
    
    # Generate the plots
    try:
        plot_path = plot_prediction_2d(
            model,
            flnm,
            root_path=save_dir,
            domain_size=domain_size,
            resolution=resolution,
            viscosity=viscosity,
            save_format=plot_format
        )
        print(f"Successfully generated plot at: {plot_path}")
    except Exception as e:
        print(f"Error generating plot: {str(e)}")
        plot_path = None
    
    # Return dictionary with plot paths
    return {
        'prediction': plot_path,
    }

if __name__ == "__main__":
    # Example usage
    import sys
    import deepxde as dde
    
    if len(sys.argv) < 2:
        print("Usage: python pinn_visualization.py model_path [filename] [output_dir]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    filename = sys.argv[2] if len(sys.argv) > 2 else "1D_Burgers_Sols_Nu0.001.hdf5"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "."
    
    # Load model and create visualization
    visualize_pinn_model(model_path, filename, root_path=output_dir) 