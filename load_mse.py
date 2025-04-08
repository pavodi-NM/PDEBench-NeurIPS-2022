import numpy as np 


# Load numpy array from .pnz file 
data = np.load('1D_Burgers_Sols_Nu1.0_Unet-PF-20mse_time.npz')

# Print the keys of the dictionary
print(data['mse'])

