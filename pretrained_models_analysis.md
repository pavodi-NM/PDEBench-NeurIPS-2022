UNet Model pretrained model downloaded from the following link: 
on Burger's case with viscosity 1.0, filename: 1D_Burgers_Sols_Nu1.0.hdf5

## Comparison of Model Performance Metrics

| Metric | UNet (Pretrained) | UNet (Retrained on Synapse) | FNO (Trained on Synapse) |
|--------|-------------------|----------------------------|--------------------------|
| Initial step | 10 | 10 | - |
| RMSE | 0.05174 | 0.02487 | 0.00120 |
| Normalized RMSE | 0.34638 | 0.25068 | 0.00402 |
| RMSE of conserved variables | 0.06335 | 0.01834 | 0.00013 |
| Maximum value of RMS error | 0.31827 | 0.15413 | 0.00806 |
| RMSE at boundaries | 0.05242 | 0.03131 | 0.00120 |
| RMSE in Fourier space | [0.02118811, 0.00252238, 0.00047786] | [0.00909343, 0.00250319, 0.00020331] | [4.1821165e-04, 1.7442682e-05, 2.2127570e-06] |
| Training time | - | ~7 hours | ~3 hours |

## Model Details

**FNO model**: Trained on Synapse with the same conditions as the UNet model (as the pretrained downloaded model did not work well), the training took about 3 hours.

**UNet model (retrained)**: Trained on Synapse with the same conditions as the pretrained model, started at time 12:45 and took around 7 hours.

**PINN model**: Training PINN with the same conditions as the original model (results pending) 