from pathlib import Path

from matplotlib import pyplot as plt
from utils import get_last_head, get_head_at_time




def plot_last_head(folder):
    """Plot all layers of the final head timestep from a .hds file."""
    folder = Path(folder)
    print(f"Looking for .hds files in {folder}")
    last_head = get_last_head(folder)

    layer_count = last_head.shape[0]
    figure, axes = plt.subplots(
        1, layer_count, figsize=(5 * layer_count, 5), squeeze=False
    )
    for layer, axis in enumerate(axes[0]):
        image = axis.imshow(last_head[layer], origin="upper")
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
    print(f"Looking for .hds files in {folder}")
    first_head = get_head_at_time(folder, timestp=0)

    layer_count = first_head.shape[0]
    figure, axes = plt.subplots(
        1, layer_count, figsize=(5 * layer_count, 5), squeeze=False
    )
    for layer, axis in enumerate(axes[0]):
        image = axis.imshow(first_head[layer], origin="upper")
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
