"""Plot vertical cross-sections of MODFLOW 6 simulation results."""

from pathlib import Path

import flopy
import matplotlib.pyplot as plt
import numpy as np

from plot_clogging_utils import read_deposited_mass_file


_VARIABLES = {
    "concentration": ("Concentration (model units)", "concentration"),
    "hydraulic_conductivity": (
        "Hydraulic conductivity (model units)",
        "hydraulic_conductivity",
    ),
    "head": ("Hydraulic head (model units)", "head"),
    "deposited_mass": ("Deposited mass (model units)", "deposited_mass"),
}


def get_cross_section_array(
    gwf,
    sim_i,
    variable,
    totim=None,
    modelname_mf="flow_model",
    modelname_mt="transport_model",
):
    """Read one variable array from a simulation, selecting an output time if needed."""
    workspace = Path(sim_i)
    if variable not in _VARIABLES:
        raise ValueError(
            f"Unsupported cross-section variable {variable!r}; "
            f"choose from {', '.join(_VARIABLES)}."
        )

    if variable == "hydraulic_conductivity":
        if gwf.npf is None or gwf.npf.k.array is None:
            raise ValueError(
                f"Flow model in {workspace} has no hydraulic conductivity data."
            )
        values = np.asarray(gwf.npf.k.array)
        if values.ndim == 0:
            values = np.full(tuple(gwf.modelgrid.shape), values)
    elif variable == "deposited_mass":
        values = read_deposited_mass_file(workspace)
    else:
        if variable == "concentration":
            filename = f"{modelname_mt}.ucn"
            text = "CONCENTRATION"
        else:
            filename = f"{modelname_mf}.hds"
            text = "HEAD"
        output_path = workspace / filename
        if not output_path.is_file():
            raise FileNotFoundError(
                f"{variable.replace('_', ' ').title()} file not found: {output_path}"
            )

        output_file = flopy.utils.HeadFile(str(output_path), text=text)
        try:
            times = output_file.get_times()
            if not times:
                raise ValueError(f"No {variable} output records found in {output_path}.")
            if totim is None:
                selected_time = times[-1]
            else:
                matches = [time for time in times if np.isclose(time, totim)]
                if not matches:
                    raise ValueError(
                        f"No {variable} output at totim={totim}; "
                        f"available times are {times}."
                    )
                selected_time = matches[0]
            values = output_file.get_data(totim=selected_time)
        finally:
            output_file.close()

    values = np.asanyarray(values)
    expected_shape = tuple(gwf.modelgrid.shape)
    if values.shape != expected_shape:
        raise ValueError(
            f"{variable.replace('_', ' ').title()} array in {workspace} has shape "
            f"{values.shape}; expected model grid shape {expected_shape}."
        )
    return values


def plot_cross_section(
    gwf,
    sim_i,
    variable,
    line,
    totim=None,
    modelname_mf="flow_model",
    modelname_mt="transport_model",
    output_path=None,
):
    """Save a FloPy vertical cross-section for a row or column of the flow grid.

    ``line`` accepts ``{"row": zero_based_index}`` or
    ``{"column": zero_based_index}``.
    """
    workspace = Path(sim_i)
    if not isinstance(line, dict) or len(line) != 1:
        raise ValueError("line must contain exactly one 'row' or 'column' index.")
    orientation, index = next(iter(line.items()))
    if orientation not in ("row", "column"):
        raise ValueError("line must specify either 'row' or 'column'.")
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise ValueError(f"The {orientation} index must be an integer.")

    grid_shape = tuple(gwf.modelgrid.shape)
    index_bound = grid_shape[1] if orientation == "row" else grid_shape[2]
    if not 0 <= index < index_bound:
        raise IndexError(
            f"{orientation.title()} index must be between 0 and {index_bound - 1}; "
            f"got {index}."
        )

    values = get_cross_section_array(
        gwf,
        workspace,
        variable,
        totim=totim,
        modelname_mf=modelname_mf,
        modelname_mt=modelname_mt,
    )
    values = np.ma.masked_where(
        ~np.isfinite(values) | (np.abs(values) > 1.0e20),
        values,
    )
    label, variable_slug = _VARIABLES[variable]

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    try:
        cross_section = flopy.plot.PlotCrossSection(
            model=gwf,
            line={orientation: int(index)},
            ax=axis,
        )
        image = cross_section.plot_array(values, cmap="viridis")
        cross_section.plot_grid(color="0.25", linewidth=0.3, alpha=0.35)
        axis.set_xlabel(
            "Distance along columns (model units)"
            if orientation == "row"
            else "Distance along rows (model units)"
        )
        axis.set_ylabel("Elevation (model units)")

        simulation_name = workspace.name
        time_text = f", totim={totim:g}" if totim is not None else ""
        axis.set_title(
            f"{label.split(' (')[0]}, {orientation} {index} "
            f"({simulation_name}{time_text})"
        )
        figure.colorbar(image, ax=axis, label=label)

        if output_path is None:
            time_suffix = f"_totim_{totim:g}" if totim is not None else ""
            output_path = (
                workspace
                / "figures"
                / f"{variable_slug}_{orientation}_{index}{time_suffix}.png"
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output_path
