import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable







def plot_heatmap(pred_plot, target_plot, channel_plot=0, t_min=0., t_max=2., x_min=0., x_max=1., model_name="PINN_RESULTS_B_0.01"):



    fig, ax = plt.subplots(figsize=(6.5,6))
    h = ax.imshow(pred_plot[...,channel_plot].squeeze(),
                extent=[t_min, t_max, x_min, x_max], origin='lower', aspect='auto')
    h.set_clim(target_plot[...,channel_plot].min(), target_plot[...,channel_plot].max())
    divider = make_axes_locatable(ax)

    data = target_plot[...,channel_plot].squeeze()
    print(f"Target Data shape: {data.shape}, Pred Data shape: {pred_plot[...,channel_plot].squeeze().shape}")
    print(f"Data range: {data.min()} to {data.max()}")


    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=30)
    ax.set_title(f"PINN", fontsize=30)
    ax.tick_params(axis='x',labelsize=30)
    ax.tick_params(axis='y',labelsize=30)
    ax.set_ylabel("$x$", fontsize=30)
    ax.set_xlabel("$t$", fontsize=30)
    plt.tight_layout()
    filename = model_name + '_32_pred.pdf'
    plt.savefig(filename)

    fig, ax = plt.subplots(figsize=(6.5,6))
    h = ax.imshow(target_plot[...,channel_plot].squeeze(),
                extent=[t_min, t_max, x_min, x_max], origin='lower', aspect='auto')
    h.set_clim(target_plot[...,channel_plot].min(), target_plot[...,channel_plot].max())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=30)
    ax.set_title("Data", fontsize=30)
    ax.tick_params(axis='x',labelsize=30)
    ax.tick_params(axis='y',labelsize=30)
    ax.set_ylabel("$x$", fontsize=30)
    ax.set_xlabel("$t$", fontsize=30)
    plt.tight_layout()
    filename = model_name + '_32_data.pdf'
    plt.savefig(filename)
    
    return 