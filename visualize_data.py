import h5py
import matplotlib.pyplot as plt
import numpy as np

# Open the HDF5 file
file_path = "pdebench/data_download/pdebench/data/1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5"
f = h5py.File(file_path, 'r')

# Print basic information
print(f"Dataset keys: {list(f.keys())}")
print(f"x-coordinate shape: {f['x-coordinate'].shape}")  # Spatial coordinates
print(f"t-coordinate shape: {f['t-coordinate'].shape}")  # Time coordinates
print(f"tensor shape: {f['tensor'].shape}")  # Solution values (samples, time, space)

# Load data
x = f['x-coordinate'][:]
t = f['t-coordinate'][:]
# Take first sample, all time steps, all spatial points
u = f['tensor'][0, :, :]

# Create some visualizations
plt.figure(figsize=(12, 10))

# Plot 1: Solution profile at different times
plt.subplot(2, 2, 1)
time_indices = [0, 50, 100, 150, 200]  # Different time points
for idx in time_indices:
    if idx < len(t):
        plt.plot(x, u[idx, :], label=f't={t[idx]:.2f}')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution profiles at different times')
plt.legend()
plt.grid(True)

# Plot 2: Spacetime plot (heatmap)
plt.subplot(2, 2, 2)
plt.pcolormesh(t[:100], x, u[:100, :].T, shading='auto', cmap='viridis')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Spacetime plot (first 100 timesteps)')
plt.colorbar(label='u')

# Plot 3: Time evolution at specific spatial points
plt.subplot(2, 2, 3)
space_indices = [256, 384, 512, 640, 768]  # Different spatial points
for idx in space_indices:
    if idx < len(x):
        plt.plot(t, u[:, idx], label=f'x={x[idx]:.3f}')
plt.xlabel('t')
plt.ylabel('u')
plt.title('Time evolution at specific spatial points')
plt.legend()
plt.grid(True)

# Plot 4: 3D surface plot
plt.subplot(2, 2, 4)
from mpl_toolkits.mplot3d import Axes3D
ax = plt.gca(projection='3d')
T, X = np.meshgrid(t[::4][:50], x[::4])  # Sample fewer points for clarity
U = u[::4, ::4][:50].T
surf = ax.plot_surface(T, X, U, cmap='viridis', edgecolor='none')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')
ax.set_title('3D surface plot (sampled)')

plt.tight_layout()
plt.savefig('burgers_data_visualization.png', dpi=300)
print("Visualization saved to burgers_data_visualization.png")

# Close the file
f.close() 