from pathlib import Path

import flopy
import matplotlib.pyplot as plt
import numpy as np


def plot_flow_properties(
    model_ws,
    modelname_mf="flow_model",
    layer=0,
    totim=None,
    output_path=None,
    show=False,
):
    """Plot hydraulic head and horizontal conductivity for one model layer."""
    workspace = Path(model_ws).resolve()
    head_path = workspace / f"{modelname_mf}.hds"
    if not head_path.exists():
        raise FileNotFoundError(f"Head file not found: {head_path}")

    simulation = flopy.mf6.MFSimulation.load(
        sim_ws=str(workspace),
        verbosity_level=0,
    )
    gwf = simulation.get_model(modelname_mf)
    if gwf is None:
        raise ValueError(f"Flow model {modelname_mf!r} was not found in {workspace}.")

    nlay, nrow, ncol = gwf.modelgrid.shape
    if not 0 <= layer < nlay:
        raise ValueError(f"Layer must be between 0 and {nlay - 1}.")

    head_file = flopy.utils.HeadFile(str(head_path), text="HEAD")
    times = head_file.get_times()
    if not times:
        raise ValueError(f"No head records found in {head_path}.")

    if totim is None:
        totim = times[-1]
    else:
        matching_times = [time for time in times if np.isclose(time, totim)]
        if not matching_times:
            raise ValueError(f"No head record found at totim={totim}.")
        totim = matching_times[0]

    heads = head_file.get_data(totim=totim)
    head_layer = np.ma.masked_where(
        ~np.isfinite(heads[layer]) | (heads[layer] < -1.0e20),
        heads[layer],
    )

    conductivity = np.asarray(gwf.npf.k.array)
    if conductivity.ndim == 0:
        conductivity = np.full((nlay, nrow, ncol), conductivity)
    conductivity_layer = conductivity[layer]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    head_map = flopy.plot.PlotMapView(model=gwf, layer=layer, ax=axes[0])
    head_image = head_map.plot_array(head_layer, cmap="viridis")
    head_map.plot_grid(color="0.25", linewidth=0.3, alpha=0.35)
    axes[0].set_title(f"Hydraulic head, layer {layer + 1}, time {totim:g}")
    figure.colorbar(head_image, ax=axes[0], label="Head")

    conductivity_map = flopy.plot.PlotMapView(model=gwf, layer=layer, ax=axes[1])
    conductivity_image = conductivity_map.plot_array(
        conductivity_layer,
        cmap="plasma",
    )
    conductivity_map.plot_grid(color="0.25", linewidth=0.3, alpha=0.35)
    axes[1].set_title(f"Horizontal conductivity, layer {layer + 1}")
    figure.colorbar(conductivity_image, ax=axes[1], label="K")

    if output_path is None:
        output_path = workspace / "figures" / f"flow_properties_layer_{layer + 1}.png"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(figure)
    return output_path

