"""Static and interactive plots for iteration-dependent clogging data."""

from pathlib import Path

import flopy
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    get_concentration_times,
    get_head_times,
    read_concentration_file,
    read_head_file,
    _regular_timestep_duration,
)


def get_simulation_folders(model_ws):
    """Return sim_i folders in iteration order."""
    workspace = Path(model_ws)
    if not workspace.is_dir():
        raise FileNotFoundError(f"Simulation workspace not found: {workspace}")
    folders = [
        folder
        for folder in workspace.iterdir()
        if folder.is_dir()
        and folder.name.startswith("sim_")
        and folder.name[4:].isdigit()
    ]
    folders.sort(key=lambda folder: int(folder.name[4:]))
    if not folders:
        raise FileNotFoundError(f"No sim_i folders found in {workspace}.")
    return folders


def read_deposited_mass_file(model_ws):
    """Load the deposited-mass array saved for one simulation iteration."""
    path = Path(model_ws) / "deposited_mass.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Deposited-mass file not found: {path}")
    deposited_mass = np.load(path, allow_pickle=False)
    if deposited_mass.ndim != 3:
        raise ValueError(
            f"Deposited mass in {path} must have shape (layer, row, column); "
            f"got {deposited_mass.shape}."
        )
    return deposited_mass


def read_hydraulic_conductivity(model_ws, modelname_mf="flow_model"):
    """Read the conductivity array used by the flow model in one iteration."""
    workspace = Path(model_ws)
    name_file = workspace / "mfsim.nam"
    if not name_file.is_file():
        raise FileNotFoundError(f"MODFLOW simulation name file not found: {name_file}")

    simulation = flopy.mf6.MFSimulation.load(
        sim_ws=str(workspace),
        verbosity_level=0,
    )
    gwf = simulation.get_model(modelname_mf)
    if gwf is None:
        raise ValueError(f"Flow model {modelname_mf!r} was not found in {workspace}.")
    if gwf.npf is None or gwf.npf.k.array is None:
        raise ValueError(
            f"Flow model {modelname_mf!r} in {workspace} has no hydraulic "
            "conductivity data."
        )

    shape = tuple(gwf.modelgrid.shape)
    conductivity = np.asarray(gwf.npf.k.array)
    if conductivity.ndim == 0:
        conductivity = np.full(shape, conductivity)
    if conductivity.shape != shape:
        raise ValueError(
            f"Hydraulic conductivity in {workspace} has shape "
            f"{conductivity.shape}; expected {shape}."
        )
    return conductivity


def _workspace_list(model_ws):
    if isinstance(model_ws, (str, Path)):
        workspace = Path(model_ws)
        if not workspace.is_dir():
            raise FileNotFoundError(f"Simulation workspace not found: {workspace}")
        if any(
            child.is_dir()
            and child.name.startswith("sim_")
            and child.name[4:].isdigit()
            for child in workspace.iterdir()
        ):
            return get_simulation_folders(workspace), workspace
        return [workspace], workspace.parent if workspace.name.startswith("sim_") else workspace

    workspaces = [Path(folder) for folder in model_ws]
    if not workspaces:
        raise ValueError("At least one simulation workspace is required.")
    for workspace in workspaces:
        if not workspace.is_dir():
            raise FileNotFoundError(f"Simulation workspace not found: {workspace}")
    if all(
        folder.name.startswith("sim_") and folder.name[4:].isdigit()
        for folder in workspaces
    ):
        workspaces.sort(key=lambda folder: int(folder.name[4:]))
        output_workspace = workspaces[0].parent
    else:
        output_workspace = workspaces[0]
    return workspaces, output_workspace


def _get_iteration_ids(workspaces):
    iteration_ids = []
    for index, workspace in enumerate(workspaces):
        suffix = workspace.name[4:] if workspace.name.startswith("sim_") else ""
        iteration_ids.append(int(suffix) if suffix.isdigit() else index)
    return np.asarray(iteration_ids)


def _position_values(array, x, y, layer):
    if array.ndim != 3:
        raise ValueError(
            "Spatial arrays must have shape (layer, row, column); "
            f"got {array.shape}."
        )
    if not 0 <= layer < array.shape[0]:
        raise IndexError("The requested layer is outside the array.")
    if not 0 <= y < array.shape[1] or not 0 <= x < array.shape[2]:
        raise IndexError("The requested (x, y) position is outside the array.")
    return array[layer, y, x]


def _get_iteration_values(model_ws, x, y, layer, reader):
    workspaces, output_workspace = _workspace_list(model_ws)
    arrays = [np.asarray(reader(workspace)) for workspace in workspaces]
    expected_shape = arrays[0].shape
    for workspace, array in zip(workspaces, arrays):
        if array.shape != expected_shape:
            raise ValueError(
                f"Grid in {workspace} has shape {array.shape}; expected "
                f"{expected_shape}."
            )
    values = np.asarray(
        [_position_values(array, x, y, layer) for array in arrays],
        dtype=float,
    )
    return _get_iteration_ids(workspaces), values, output_workspace


def _save_static_iteration_plot(
    model_ws,
    x,
    y,
    layer,
    reader,
    ylabel,
    filename_prefix,
    output_path=None,
):
    iterations, values, output_workspace = _get_iteration_values(
        model_ws, x, y, layer, reader
    )
    figure, axis = plt.subplots()
    axis.plot(iterations, values, marker="o")
    axis.set_xlabel("Iteration")
    axis.set_ylabel(ylabel)
    axis.set_title(
        f"{ylabel} at row {y + 1}, column {x + 1}, layer {layer + 1}"
    )
    figure.tight_layout()

    if output_path is None:
        output_path = (
            output_workspace
            / "figures"
            / f"{filename_prefix}_row_{y + 1}_column_{x + 1}_layer_{layer + 1}.png"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_deposited_at_position(
    model_ws,
    x,
    y,
    layer=0,
    output_path=None,
):
    """Plot deposited mass at one cell over clogging iterations."""
    return _save_static_iteration_plot(
        model_ws,
        x,
        y,
        layer,
        read_deposited_mass_file,
        "Deposited mass (model units)",
        "deposited_mass",
        output_path,
    )


def plot_hydraulic_conductivity_at_position(
    model_ws,
    x,
    y,
    layer=0,
    modelname_mf="flow_model",
    output_path=None,
):
    """Plot the conductivity used at one cell over clogging iterations."""
    return _save_static_iteration_plot(
        model_ws,
        x,
        y,
        layer,
        lambda workspace: read_hydraulic_conductivity(workspace, modelname_mf),
        "Hydraulic conductivity (model units)",
        "hydraulic_conductivity",
        output_path,
    )


def plot_deposited_mass_curve(model_ws, x, y, layer=0, output_path=None):
    """Save a deposited-mass curve across all sim_i folders."""
    return plot_deposited_at_position(
        get_simulation_folders(model_ws),
        x,
        y,
        layer=layer,
        output_path=output_path,
    )


def plot_hydraulic_conductivity_curve(
    model_ws,
    x,
    y,
    layer=0,
    modelname_mf="flow_model",
    output_path=None,
):
    """Save a conductivity curve across all sim_i model inputs."""
    return plot_hydraulic_conductivity_at_position(
        get_simulation_folders(model_ws),
        x,
        y,
        layer=layer,
        modelname_mf=modelname_mf,
        output_path=output_path,
    )


def _read_temporal_curve_data(
    model_ws,
    x,
    y,
    layer,
    variable,
    modelname,
    timestep_duration=None,
):
    simulation_folders = get_simulation_folders(model_ws)
    if variable == "head":
        array_reader = read_head_file
        times_reader = get_head_times
        ylabel = "Head (model units)"
    else:
        array_reader = read_concentration_file
        times_reader = get_concentration_times
        ylabel = "Concentration"

    curves = []
    shapes = []
    iteration_labels = []
    for folder in simulation_folders:
        arrays = np.asarray(array_reader(folder, modelname))
        times = np.asarray(times_reader(folder, modelname), dtype=float)
        if arrays.ndim != 4 or len(times) != arrays.shape[0]:
            raise ValueError(
                f"{variable.capitalize()} data in {folder} must have one time "
                "value per (time, layer, row, column) record."
            )
        if not len(times):
            raise ValueError(f"No {variable} records found in {folder}.")
        shapes.append(arrays.shape[1:])
        selected = []
        for time_index in range(arrays.shape[0]):
            selected.append(
                _position_values(arrays[time_index], x, y, layer)
            )
        curves.append((times, np.asarray(selected, dtype=float)))
        iteration_labels.append(folder.name[4:])

    for folder, shape in zip(simulation_folders, shapes):
        if shape != shapes[0]:
            raise ValueError(
                f"{variable.capitalize()} grid in {folder} has shape {shape}; "
                f"expected {shapes[0]}."
            )

    time_gap = _regular_timestep_duration(curves[0][0], timestep_duration)
    adjusted_times = [curves[0][0]]
    for times, _ in curves[1:]:
        adjusted_times.append(times - times[0] + adjusted_times[-1][-1] + time_gap)

    all_times = np.concatenate(adjusted_times)
    all_values = np.concatenate([values for _, values in curves])
    iteration_ends = np.cumsum([len(values) for _, values in curves])
    return all_times, all_values, iteration_ends, iteration_labels, ylabel


def _read_iteration_curve_data(
    model_ws,
    x,
    y,
    layer,
    variable,
    modelname_mf,
):
    reader = read_deposited_mass_file if variable == "deposited_mass" else (
        lambda workspace: read_hydraulic_conductivity(workspace, modelname_mf)
    )
    iterations, values, _ = _get_iteration_values(
        model_ws, x, y, layer, reader
    )
    ylabel = (
        "Deposited mass (model units)"
        if variable == "deposited_mass"
        else "Hydraulic conductivity (model units)"
    )
    labels = [str(value) for value in iterations]
    return iterations, values, np.arange(1, len(values) + 1), labels, ylabel


def _range_with_padding(values):
    if not np.any(np.isfinite(values)):
        raise ValueError("Interactive curve data must contain finite values.")
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    padding = (maximum - minimum) * 0.05
    if padding == 0:
        padding = max(abs(minimum) * 0.05, 1.0)
    return [minimum - padding, maximum + padding]


def _write_interactive_curve(
    output_path,
    x_values,
    y_values,
    iteration_ends,
    iteration_labels,
    title,
    xlabel,
    ylabel,
):
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "Interactive HTML plots require Plotly. Install it with "
            "`pip install -r requirements-plotting.txt`."
        ) from error

    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    if x_values.ndim != 1 or y_values.ndim != 1 or len(x_values) != len(y_values):
        raise ValueError("Interactive curve x/y data must be matching 1-D arrays.")
    if not len(x_values):
        raise ValueError("Interactive curve data must not be empty.")

    ends = np.asarray(iteration_ends, dtype=int)
    if (
        ends.ndim != 1
        or not len(ends)
        or np.any(ends <= 0)
        or ends[-1] != len(x_values)
        or len(iteration_labels) != len(ends)
    ):
        raise ValueError("Iteration boundaries must end at the curve data length.")

    figure = go.Figure(
        data=[
            go.Scatter(
                x=x_values[: ends[0]],
                y=y_values[: ends[0]],
                mode="lines+markers",
                name=title,
                hovertemplate=f"{xlabel}: %{{x}}<br>{ylabel}: %{{y}}<extra></extra>",
            )
        ]
    )
    frames = [
        go.Frame(
            name=str(label),
            data=[
                go.Scatter(
                    x=x_values[:end],
                    y=y_values[:end],
                    mode="lines+markers",
                    name=title,
                    hovertemplate=f"{xlabel}: %{{x}}<br>{ylabel}: %{{y}}<extra></extra>",
                )
            ],
        )
        for end, label in zip(ends, iteration_labels)
    ]
    figure.frames = frames
    slider_steps = [
        {
            "args": [
                [frame.name],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": label,
            "method": "animate",
        }
        for frame, label in zip(frames, iteration_labels)
    ]
    figure.update_layout(
        title=title,
        xaxis={"title": xlabel, "range": _range_with_padding(x_values)},
        yaxis={"title": ylabel, "range": _range_with_padding(y_values)},
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Through iteration: "},
                "steps": slider_steps,
            }
        ],
        margin={"l": 70, "r": 30, "t": 70, "b": 90},
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)
    return output_path


def _interactive_output_path(model_ws, filename, output_path):
    if output_path is not None:
        return Path(output_path)
    workspace = Path(model_ws)
    return workspace / "figures" / filename


def plot_breakthrough_curve_html(
    model_ws,
    x,
    y,
    layer=0,
    modelname_mt="transport_model",
    output_path=None,
    timestep_duration=None,
):
    """Write an offline, interactive breakthrough curve HTML file."""
    times, values, ends, labels, ylabel = _read_temporal_curve_data(
        model_ws,
        x,
        y,
        layer,
        "concentration",
        modelname_mt,
        timestep_duration,
    )
    output_path = _interactive_output_path(
        model_ws,
        f"btc_row_{y + 1}_column_{x + 1}_layer_{layer + 1}.html",
        output_path,
    )
    return _write_interactive_curve(
        output_path,
        times,
        values,
        ends,
        labels,
        f"Concentration at row {y + 1}, column {x + 1}, layer {layer + 1}",
        "Time (hours)",
        ylabel,
    )


def plot_head_curve_html(
    model_ws,
    x,
    y,
    layer=0,
    modelname_mf="flow_model",
    output_path=None,
    timestep_duration=None,
):
    """Write an offline, interactive head curve HTML file."""
    times, values, ends, labels, ylabel = _read_temporal_curve_data(
        model_ws,
        x,
        y,
        layer,
        "head",
        modelname_mf,
        timestep_duration,
    )
    output_path = _interactive_output_path(
        model_ws,
        f"hydraulic_head_row_{y + 1}_column_{x + 1}_layer_{layer + 1}.html",
        output_path,
    )
    return _write_interactive_curve(
        output_path,
        times,
        values,
        ends,
        labels,
        f"Head at row {y + 1}, column {x + 1}, layer {layer + 1}",
        "Time (hours)",
        ylabel,
    )


def plot_deposited_mass_curve_html(
    model_ws,
    x,
    y,
    layer=0,
    output_path=None,
):
    """Write an offline, interactive deposited-mass HTML file."""
    iterations, values, ends, labels, ylabel = _read_iteration_curve_data(
        model_ws, x, y, layer, "deposited_mass", "flow_model"
    )
    output_path = _interactive_output_path(
        model_ws,
        f"deposited_mass_row_{y + 1}_column_{x + 1}_layer_{layer + 1}.html",
        output_path,
    )
    return _write_interactive_curve(
        output_path,
        iterations,
        values,
        ends,
        labels,
        f"Deposited mass at row {y + 1}, column {x + 1}, layer {layer + 1}",
        "Iteration",
        ylabel,
    )


def plot_hydraulic_conductivity_curve_html(
    model_ws,
    x,
    y,
    layer=0,
    modelname_mf="flow_model",
    output_path=None,
):
    """Write an offline, interactive conductivity HTML file."""
    iterations, values, ends, labels, ylabel = _read_iteration_curve_data(
        model_ws, x, y, layer, "hydraulic_conductivity", modelname_mf
    )
    output_path = _interactive_output_path(
        model_ws,
        f"hydraulic_conductivity_row_{y + 1}_column_{x + 1}_layer_{layer + 1}.html",
        output_path,
    )
    return _write_interactive_curve(
        output_path,
        iterations,
        values,
        ends,
        labels,
        f"Hydraulic conductivity at row {y + 1}, column {x + 1}, layer {layer + 1}",
        "Iteration",
        ylabel,
    )
