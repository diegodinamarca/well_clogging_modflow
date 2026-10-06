

from pathlib import Path
import flopy
import matplotlib.pyplot as plt
import numpy as np


# Read concentration data from a MODFLOW 6 simulation output file (.ucn) and return the concentration values as a numpy array.
def read_concentration_file(model_ws, modelname_mt="transport_model"):
    ucnfile = Path(model_ws) / f"{modelname_mt}.ucn"
    if filenotfound := not ucnfile.exists():
        raise FileNotFoundError(f"No .ucn file found at {ucnfile}")
    print(f"Reading concentration data from {ucnfile}")
    ucnobj = flopy.utils.HeadFile(ucnfile, text="CONCENTRATION")
    conc = ucnobj.get_alldata()
    return conc


def plot_array_at_position(array, x, y, layer=0, times=None, ax=None):
    """Plot array values through time at a zero-based (column, row) position."""
    values = np.asarray(array)
    if values.ndim not in (3, 4):
        raise ValueError("array must have shape (time, row, column) or (time, layer, row, column).")
    if not 0 <= y < values.shape[-2] or not 0 <= x < values.shape[-1]:
        raise IndexError("The requested (x, y) position is outside the array.")

    if values.ndim == 4:
        if not 0 <= layer < values.shape[1]:
            raise IndexError("The requested layer is outside the array.")
        position_values = values[:, layer, y, x]
    else:
        if layer != 0:
            raise IndexError("A 3-D array does not have a layer dimension.")
        position_values = values[:, y, x]

    if times is None:
        times = np.arange(position_values.shape[0])
    elif len(times) != position_values.shape[0]:
        raise ValueError("times must contain one value per array timestep.")

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(times, position_values)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    return ax

def plot_head_at_position(
    model_ws,
    x,
    y,
    layer=0,
    modelname_mf="flow_model",
    output_path=None,
):
    """Save a head plot for one cell, optionally joining simulations."""
    if isinstance(model_ws, (str, Path)):
        model_workspaces = [Path(model_ws)]
    else:
        model_workspaces = [Path(folder) for folder in model_ws]
    if not model_workspaces:
        raise ValueError("At least one model workspace is required.")

    head_arrays = []
    time_arrays = []
    for workspace in model_workspaces:
        head_arrays.append(read_head_file(workspace, modelname_mf))
        time_arrays.append(np.asarray(get_head_times(workspace, modelname_mf)))

    expected_shape = head_arrays[0].shape[1:]
    for workspace, heads in zip(model_workspaces, head_arrays):
        if heads.shape[1:] != expected_shape:
            raise ValueError(
                f"Head grid in {workspace} has shape {heads.shape[1:]}; "
                f"expected {expected_shape}."
            )

    adjusted_times = [time_arrays[0]]
    for times in time_arrays[1:]:
        adjusted_times.append(times - times[0] + adjusted_times[-1][-1] + 1)
    head = np.concatenate(head_arrays, axis=0)
    times = np.concatenate(adjusted_times)

    figure, axis = plt.subplots()
    plot_array_at_position(head, x, y, layer=layer, times=times, ax=axis)
    axis.set_xlabel("Time (days)")
    axis.set_ylabel("Head")
    axis.set_title(f"Head at row {y + 1}, column {x + 1}, layer {layer + 1}")
    figure.tight_layout()

    if output_path is None:
        output_workspace = model_workspaces[0]
        if len(model_workspaces) > 1:
            output_workspace = output_workspace.parent
        output_path = (
            output_workspace
            / "figures"
            / f"head_row_{y + 1}_column_{x + 1}_layer_{layer + 1}.png"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    return output_path

def plot_concentration_at_position(
    model_ws,
    x,
    y,
    layer=0,
    modelname_mt="transport_model",
    output_path=None,
):
    """Save a concentration plot for one cell, optionally joining simulations."""
    if isinstance(model_ws, (str, Path)):
        model_workspaces = [Path(model_ws)]
    else:
        model_workspaces = [Path(folder) for folder in model_ws]
    if not model_workspaces:
        raise ValueError("At least one model workspace is required.")

    concentration_arrays = []
    time_arrays = []
    for workspace in model_workspaces:
        concentration_arrays.append(read_concentration_file(workspace, modelname_mt))
        time_arrays.append(np.asarray(get_concentration_times(workspace, modelname_mt)))

    expected_shape = concentration_arrays[0].shape[1:]
    for workspace, concentrations in zip(model_workspaces, concentration_arrays):
        if concentrations.shape[1:] != expected_shape:
            raise ValueError(
                f"Concentration grid in {workspace} has shape {concentrations.shape[1:]}; "
                f"expected {expected_shape}."
            )

    adjusted_times = [time_arrays[0]]
    for times in time_arrays[1:]:
        adjusted_times.append(times - times[0] + adjusted_times[-1][-1] + 1)
    concentration = np.concatenate(concentration_arrays, axis=0)
    times = np.concatenate(adjusted_times)

    figure, axis = plt.subplots()
    plot_array_at_position(concentration, x, y, layer=layer, times=times, ax=axis)
    axis.set_xlabel("Time (days)")
    axis.set_ylabel("Concentration")
    axis.set_title(f"Concentration at row {y + 1}, column {x + 1}, layer {layer + 1}")
    figure.tight_layout()

    if output_path is None:
        output_workspace = model_workspaces[0]
        if len(model_workspaces) > 1:
            output_workspace = output_workspace.parent
        output_path = (
            output_workspace
            / "figures"
            / f"concentration_row_{y + 1}_column_{x + 1}_layer_{layer + 1}.png"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    return output_path

def plot_head_curve(
    model_ws,
    x,
    y,
    layer=0,
    modelname_mf="flow_model",
    output_path=None,
):
    """Plot one continuous breakthrough curve across all sim_i folders."""
    model_ws = Path(model_ws)
    simulation_folders = [
        folder
        for folder in model_ws.iterdir()
        if folder.is_dir()
        and folder.name.startswith("sim_")
        and folder.name[4:].isdigit()
    ]
    simulation_folders.sort(key=lambda folder: int(folder.name[4:]))
    if not simulation_folders:
        raise FileNotFoundError(f"No sim_i folders found in {model_ws}.")

    if output_path is None:
        output_path = (
            model_ws
            / "figures"
            / f"hydraulic_head_row_{y + 1}_column_{x + 1}_layer_{layer + 1}.png"
        )
    return plot_head_at_position(
        simulation_folders,
        x,
        y,
        layer=layer,
        modelname_mf=modelname_mf,
        output_path=output_path,
    )


def plot_breakthrough_curve(
    model_ws,
    x,
    y,
    layer=0,
    modelname_mt="transport_model",
    output_path=None,
):
    """Plot one continuous breakthrough curve across all sim_i folders."""
    model_ws = Path(model_ws)
    simulation_folders = [
        folder
        for folder in model_ws.iterdir()
        if folder.is_dir()
        and folder.name.startswith("sim_")
        and folder.name[4:].isdigit()
    ]
    simulation_folders.sort(key=lambda folder: int(folder.name[4:]))
    if not simulation_folders:
        raise FileNotFoundError(f"No sim_i folders found in {model_ws}.")

    if output_path is None:
        output_path = (
            model_ws
            / "figures"
            / f"btc_row_{y + 1}_column_{x + 1}_layer_{layer + 1}.png"
        )
    return plot_concentration_at_position(
        simulation_folders,
        x,
        y,
        layer=layer,
        modelname_mt=modelname_mt,
        output_path=output_path,
    )

def get_concentration_at_time(model_ws, timestp, modelname_mt="transport_model"):
    ucnfile = Path(model_ws) / f"{modelname_mt}.ucn"
    if filenotfound := not ucnfile.exists():
        raise FileNotFoundError(f"No .ucn file found at {ucnfile}")
    ucnobj = flopy.utils.HeadFile(ucnfile, text="CONCENTRATION")
    times_mt = ucnobj.get_times()
    conc = ucnobj.get_data(totim=times_mt[timestp])
    return conc

def get_concentration_times(model_ws, modelname_mt="transport_model"):
    ucnfile = Path(model_ws) / f"{modelname_mt}.ucn"
    if filenotfound := not ucnfile.exists():
        raise FileNotFoundError(f"No .ucn file found at {ucnfile}")
    ucnobj = flopy.utils.HeadFile(ucnfile, text="CONCENTRATION")
    # Get available times
    times_mt = ucnobj.get_times()
    return(times_mt)

def get_last_concentration(model_ws, modelname_mt="transport_model"):
    ucnfile = Path(model_ws) / f"{modelname_mt}.ucn"
    if filenotfound := not ucnfile.exists():
        raise FileNotFoundError(f"No .ucn file found at {ucnfile}")
    ucnobj = flopy.utils.HeadFile(ucnfile, text="CONCENTRATION")
    # Get available times
    times_mt = ucnobj.get_times()
    last_time = times_mt[-1]
    conc = ucnobj.get_data(totim=last_time)
    return conc


# Read head data from a MODFLOW 6 simulation output file (.hds) and return the head values as a numpy array.
def read_head_file(model_ws, modelname_mt="flow_model"):
    ucnfile = Path(model_ws) / f"{modelname_mt}.hds"
    if filenotfound := not ucnfile.exists():
        raise FileNotFoundError(f"No .hds file found at {ucnfile}")
    print(f"Reading head data from {ucnfile}")
    ucnobj = flopy.utils.HeadFile(ucnfile, text="HEAD")
    conc = ucnobj.get_alldata()
    return conc

def get_head_at_time(model_ws, timestp, modelname_mt="flow_model"):
    ucnfile = Path(model_ws) / f"{modelname_mt}.hds"
    if filenotfound := not ucnfile.exists():
        raise FileNotFoundError(f"No .hds file found at {ucnfile}")
    ucnobj = flopy.utils.HeadFile(ucnfile, text="HEAD")
    times_mt = ucnobj.get_times()
    head = ucnobj.get_data(totim=times_mt[timestp])
    return head

def get_head_times(model_ws, modelname_mt="flow_model"):
    ucnfile = Path(model_ws) / f"{modelname_mt}.hds"
    if filenotfound := not ucnfile.exists():
        raise FileNotFoundError(f"No .hds file found at {ucnfile}")
    ucnobj = flopy.utils.HeadFile(ucnfile, text="HEAD")
    # Get available times
    times_mt = ucnobj.get_times()
    return(times_mt)

def get_last_head(model_ws, modelname_mt="flow_model"):
    file = Path(model_ws) / f"{modelname_mt}.hds"
    if filenotfound := not file.exists():
        raise FileNotFoundError(f"No .hds file found at {file}")
    ucnobj = flopy.utils.HeadFile(file, text="HEAD")
    # Get available times
    times_mt = ucnobj.get_times()
    last_time = times_mt[-1]
    head = ucnobj.get_data(totim=last_time)
    return head