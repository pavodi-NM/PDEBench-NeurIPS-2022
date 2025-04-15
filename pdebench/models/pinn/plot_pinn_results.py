#!/usr/bin/env python
"""
Generate visualizations for trained PINN models in PDEBench.

This script loads trained PINN models and generates standardized visualizations
in the same format as FNO and UNet models in PDEBench.

Example usage:
    python plot_pinn_results.py --model_path path/to/model.pt --filename 1D_Burgers_Sols_Nu0.001.hdf5
"""

import os
import sys
import argparse
import deepxde as dde
from pinn_visualization import visualize_pinn_model, generate_pinn_plots_from_loaded_model

def parse_args():
    parser = argparse.ArgumentParser(description="Generate visualizations for trained PINN models")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the saved PINN model (.pt file)")
    parser.add_argument("--filename", type=str, default=None,
                       help="Original dataset filename (for naming plots)")
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Directory to save the plots")
    parser.add_argument("--resolution", type=str, default="100,100",
                       help="Resolution of the visualization grid (nx,nt)")
    parser.add_argument("--viscosity", type=float, default=None,
                       help="Viscosity parameter (will try to extract from filename if not provided)")
    parser.add_argument("--format", type=str, default="pdf",
                       choices=["pdf", "png"],
                       help="Output format for plots")
    parser.add_argument("--domain_size", type=str, default="1.0,2.0",
                       help="Domain size for x and t (x_size,t_size)")
    parser.add_argument("--use_loaded_model", action="store_true",
                       help="Use if model is already loaded, will be passed directly to plotting function")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Extract model name from model path if filename not provided
    if args.filename is None:
        args.filename = os.path.basename(args.model_path).replace(".pt", "")
    
    # Parse resolution and domain size
    resolution = tuple(map(int, args.resolution.split(",")))
    domain_size = tuple(map(float, args.domain_size.split(",")))
    
    # For Burgers in PDEBench, the typical domain size is (1.0, 2.0) for (x, t)
    # Ensure domain_size is in the correct order
    if "Burgers" in args.filename:
        if domain_size[0] > domain_size[1]:  # If x is larger than t, swap them
            domain_size = (domain_size[1], domain_size[0])
            print(f"Swapped domain size to match Burgers equation format: {domain_size}")
    
    # Extract viscosity from filename if not provided
    viscosity = args.viscosity
    if viscosity is None and "Nu" in args.filename:
        try:
            viscosity = float(args.filename.split("Nu")[1].split("_")[0])
        except:
            viscosity = 0.01  # Default value
    
    if viscosity is None:
        viscosity = 0.01  # Default if not specified or couldn't extract
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate the visualizations
    try:
        print(f"Generating visualizations for {args.model_path}...")
        
        if args.use_loaded_model:
            # This branch is for when the model is already loaded and passed as model_path
            # In this case, model_path is actually the loaded model object
            model = args.model_path
            print("Using already loaded model for visualization")
            result = generate_pinn_plots_from_loaded_model(
                model,
                args.filename,
                save_dir=args.output_dir,
                domain_size=domain_size,
                resolution=resolution,
                viscosity=viscosity,
                plot_format=args.format
            )
            prediction_path = result['prediction']
        else:
            # This branch is for when we need to load the model from a file
            prediction_path = visualize_pinn_model(
                args.model_path,
                args.filename,
                root_path=args.output_dir,
                domain_size=domain_size,
                resolution=resolution,
                viscosity=viscosity,
                plot_format=args.format
            )
            
        print(f"Plots saved to: {prediction_path}")
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 