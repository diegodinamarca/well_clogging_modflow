from pathlib import Path

import flopy
from matplotlib import pyplot as plt
import numpy as np
from utils import get_last_head, get_head_at_time




def plot_last_head(folder):
    """Plot all layers of the final head timestep from a .hds file."""
    folder = Path(folder)
    #print(f"Looking for .hds files in {folder}")
    last_head = get_last_head(folder)
    simulation = flopy.mf6.MFSimulation.load(
        sim_ws=str(folder),
        verbosity_level=0,
    )
    model = simulation.get_model("flow_model")
    if model is None:
        raise ValueError(f"Flow model was not found in {folder}.")

    layer_count = last_head.shape[0]
    figure, axes = plt.subplots(
        1, layer_count, figsize=(5 * layer_count, 5), squeeze=False
    )
    for layer, axis in enumerate(axes[0]):
        head_map = flopy.plot.PlotMapView(model=model, layer=layer, ax=axis)
        layer_head = np.ma.masked_where(
            ~np.isfinite(last_head[layer]) | (last_head[layer] < -1.0e20),
            last_head[layer],
        )
        image = head_map.plot_array(layer_head, cmap="viridis")
        head_map.plot_grid(color="0.25", linewidth=0.3, alpha=0.35)
        axis.set_title(f"Layer {layer + 1}")
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
        figure.colorbar(image, ax=axis, label="Head")

    figure.tight_layout()
    figures_folder = folder / "figures"
    figures_folder.mkdir(exist_ok=True)
    output_path = figures_folder / "last_head.png"
    figure.savefig(output_path)
    plt.close(figure)
    return output_path

def plot_first_head(folder):
    """Plot all layers of the first head timestep from a .hds file."""
    folder = Path(folder)
    #print(f"Looking for .hds files in {folder}")
    first_head = get_head_at_time(folder, timestp=0)
    simulation = flopy.mf6.MFSimulation.load(
        sim_ws=str(folder),
        verbosity_level=0,
    )
    model = simulation.get_model("flow_model")
    if model is None:
        raise ValueError(f"Flow model was not found in {folder}.")

    layer_count = first_head.shape[0]
    figure, axes = plt.subplots(
        1, layer_count, figsize=(5 * layer_count, 5), squeeze=False
    )
    for layer, axis in enumerate(axes[0]):
        head_map = flopy.plot.PlotMapView(model=model, layer=layer, ax=axis)
        layer_head = np.ma.masked_where(
            ~np.isfinite(first_head[layer]) | (first_head[layer] < -1.0e20),
            first_head[layer],
        )
        image = head_map.plot_array(layer_head, cmap="viridis")
        head_map.plot_grid(color="0.25", linewidth=0.3, alpha=0.35)
        axis.set_title(f"Layer {layer + 1}")
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
        figure.colorbar(image, ax=axis, label="Head")

    figure.tight_layout()
    figures_folder = folder / "figures"
    figures_folder.mkdir(exist_ok=True)
    output_path = figures_folder / "first_head.png"
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


#plot_last_head("/home/diego/modflow/master_thesis/test1/model/timestep_1")
#plot_first_head("/home/diego/modflow/master_thesis/test1/model/timestep_1")
