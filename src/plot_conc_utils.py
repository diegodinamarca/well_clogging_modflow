from utils import read_concentration_file, get_last_concentration, get_concentration_at_time, get_concentration_times
from pathlib import Path

import flopy
import matplotlib.pyplot as plt
import numpy as np


def plot_last_concentration(folder):
	"""Plot all layers of the final concentration timestep from a .ucn file."""
	folder = Path(folder)
	print(f"Looking for .ucn files in {folder}")
	last_concentration = get_last_concentration(folder)
	simulation = flopy.mf6.MFSimulation.load(
		sim_ws=str(folder),
		verbosity_level=0,
	)
	model = simulation.get_model("transport_model")
	if model is None:
		raise ValueError(f"Transport model was not found in {folder}.")

	layer_count = last_concentration.shape[0]
	figure, axes = plt.subplots(
		1, layer_count, figsize=(5 * layer_count, 5), squeeze=False
	)
	for layer, axis in enumerate(axes[0]):
		concentration_map = flopy.plot.PlotMapView(model=model, layer=layer, ax=axis)
		layer_concentration = np.ma.masked_where(
			~np.isfinite(last_concentration[layer])
			| (np.abs(last_concentration[layer]) > 1.0e20),
			last_concentration[layer],
		)
		image = concentration_map.plot_array(layer_concentration, cmap="viridis")
		concentration_map.plot_grid(color="0.25", linewidth=0.3, alpha=0.35)
		axis.set_title(f"Layer {layer + 1}")
		axis.set_xlabel("Column")
		axis.set_ylabel("Row")
		figure.colorbar(image, ax=axis, label="Concentration")

	figure.tight_layout()
	figures_folder = folder / "figures"
	figures_folder.mkdir(exist_ok=True)
	output_path = figures_folder / "last_concentration.png"
	figure.savefig(output_path)
	plt.close(figure)
	return output_path

def plot_first_concentration(folder):
	"""Plot all layers of the final concentration timestep from a .ucn file."""
	folder = Path(folder)
	print(f"Looking for .ucn files in {folder}")
	last_concentration = get_concentration_at_time(folder, timestp=0)
	simulation = flopy.mf6.MFSimulation.load(
		sim_ws=str(folder),
		verbosity_level=0,
	)
	model = simulation.get_model("transport_model")
	if model is None:
		raise ValueError(f"Transport model was not found in {folder}.")

	layer_count = last_concentration.shape[0]
	figure, axes = plt.subplots(
		1, layer_count, figsize=(5 * layer_count, 5), squeeze=False
	)
	for layer, axis in enumerate(axes[0]):
		concentration_map = flopy.plot.PlotMapView(model=model, layer=layer, ax=axis)
		layer_concentration = np.ma.masked_where(
			~np.isfinite(last_concentration[layer])
			| (np.abs(last_concentration[layer]) > 1.0e20),
			last_concentration[layer],
		)
		image = concentration_map.plot_array(layer_concentration, cmap="viridis")
		concentration_map.plot_grid(color="0.25", linewidth=0.3, alpha=0.35)
		axis.set_title(f"Layer {layer + 1}")
		axis.set_xlabel("Column")
		axis.set_ylabel("Row")
		figure.colorbar(image, ax=axis, label="Concentration")

	figure.tight_layout()
	figures_folder = folder / "figures"
	figures_folder.mkdir(exist_ok=True)
	output_path = figures_folder / "first_concentration.png"
	figure.savefig(output_path)
	plt.close(figure)
	return output_path

#plot_last_concentration("/home/diego/modflow/master_thesis/test1/model/timestep_1")
#plot_first_concentration("/home/diego/modflow/master_thesis/test1/model/timestep_1")