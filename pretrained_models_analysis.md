
UNet Model pretrained model downloaded from the following link: 
on Burger's case with viscosity 1.0, filename: 1D_Burgers_Sols_Nu1.0.hdf5

The metrics are as follows:

    Initial step: 10
    RMSE: 0.05174
    normalized RMSE: 0.34638
    RMSE of conserved variables: 0.06335
    Maximum value of rms error: 0.31827
    RMSE at boundaries: 0.05242
    RMSE in Fourier space: [0.02118811 0.00252238 0.00047786]


==================================================================================================================================
FNO model has been trained on Synapse with the same conditions as the UNet model (as the pretrained downloaded model did not work well), the training tool about 3 hours.
==================================================================================================================================

The metrics are as follows:

    RMSE: 0.00120
    normalized RMSE: 0.00402
    RMSE of conserved variables: 0.00013
    Maximum value of rms error: 0.00806
    RMSE at boundaries: 0.00120
    RMSE in Fourier space: [4.1821165e-04 1.7442682e-05 2.2127570e-06]


====================================================================================================================================
Training a new UNet model on Synapse with the same conditions as the pretrained model, starts at time 12:45 and takes around 7 hours.
=========================================================================================================== ================= ======

The metrics results are as follows:

    Initial step: 10
    RMSE: 0.02487
    normalized RMSE: 0.25068
    RMSE of conserved variables: 0.01834
    Maximum value of rms error: 0.15413
    RMSE at boundaries: 0.03131
    RMSE in Fourier space: [0.00909343 0.00250319 0.00020331]



Training PINN with the same conditions as the original model