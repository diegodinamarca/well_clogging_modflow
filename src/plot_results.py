"""Plot results for a range of MODFLOW simulation folders.

Usage from the project workspace::

    python src/plot_results.py 3 --ws test1

The positional argument is the maximum number of ``sim_i`` folders to process.
The optional ``--ws`` argument points to the project folder containing
``simulations/`` and ``config.toml``; it defaults to the current working
directory. Each processed simulation also gets row and column cross-sections
of concentration, hydraulic conductivity, head, and deposited mass at each
configured well. Head and concentration use the latest available output time.

Interactive HTML plots require the optional dependency in
``requirements-plotting.txt``.
"""

from plot_conc_utils import plot_first_concentration, plot_last_concentration
from plot_clogging_utils import (
    get_simulation_folders,
    plot_deposited_mass_curve,
    plot_deposited_mass_curve_html,
    plot_hydraulic_conductivity_curve,
    plot_hydraulic_conductivity_curve_html,
    plot_breakthrough_curve_html,
    plot_head_curve_html,
)
from plot_cross_section import plot_cross_section
from plot_flow_properties import plot_flow_properties
from plot_head_utils import plot_first_head, plot_last_head
from flow_model.load_flow_config import load_flow_config
from utils import (
    plot_breakthrough_curve,
    plot_concentration_at_position,
    plot_head_at_position,
    plot_head_curve,
)
import argparse
import math
from pathlib import Path
import warnings

import flopy

def plot_first_and_last_concentration(folder):
    plot_first_concentration(folder)
    plot_last_concentration(folder)

def plot_first_and_last_head(folder):
    plot_first_head(folder)
    plot_last_head(folder)


def _configured_timestep_duration(parameters, model):
    configured_duration = parameters.get("timestep_duration")
    if configured_duration is not None:
        duration = float(configured_duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("timestep_duration must be a positive number.")
        return duration

    period_lengths = parameters[f"perlen_{model}"]
    steps_per_period = parameters[f"nstp_{model}"]
    if (
        not period_lengths
        or not steps_per_period
        or not math.isfinite(period_lengths[0])
        or period_lengths[0] <= 0
        or steps_per_period[0] <= 0
    ):
        raise ValueError(
            f"Cannot determine the configured {model} timestep duration."
        )
    return period_lengths[0] / steps_per_period[0]


def main():
    # Parse the number of timestep folders and their model workspace.
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="number of timesteps to plot")
    parser.add_argument(
        "--ws",
        dest = "model_ws",
        default=".",
        help="model workspace folder (defaults to the current working directory)",
    )
    args = parser.parse_args()
    model_ws = args.model_ws
    parameters = load_flow_config(Path(model_ws) / "config.toml")
    flow_timestep_duration = _configured_timestep_duration(parameters, "mf")
    transport_timestep_duration = _configured_timestep_duration(parameters, "mt")
    #print(f"Flow timestep duration: {flow_timestep_duration}")
    simulations_workspace = Path(model_ws) / "simulations"
    simulation_folders = get_simulation_folders(simulations_workspace)
    for folder in simulation_folders[: max(args.n, 0)]:
        print(folder)
        plot_first_and_last_concentration(folder)
        plot_first_and_last_head(folder)
        plot_flow_properties(
            folder,
            layer=0,
            mask=parameters["ibound"] != 0,
        )
        simulation = flopy.mf6.MFSimulation.load(
            sim_ws=str(folder),
            verbosity_level=0,
        )
        gwf = simulation.get_model(parameters["modelname_mf"])
        if gwf is None:
            raise ValueError(
                f"Flow model {parameters['modelname_mf']!r} was not found in {folder}."
            )
        for variable in (
            "concentration",
            "hydraulic_conductivity",
            "head",
            "deposited_mass",
        ):
            for (well_row, well_column) in parameters["well_location"]:
                for line in (
                    {"row": well_row - 1},
                    {"column": well_column - 1},
                ):
                    try:
                        plot_cross_section(
                            gwf,
                            folder,
                            variable,
                            line=line,
                            modelname_mf=parameters["modelname_mf"],
                            modelname_mt=parameters["modelname_mt"],
                        )
                    except (FileNotFoundError, ValueError) as error:
                        warnings.warn(
                            f"Skipping {variable} cross-section for {folder} "
                            f"at {line}: {error}",
                            RuntimeWarning,
                        )
        for (well_row, well_column), well_layers in zip(
            parameters["well_location"], parameters["well_layer"]
        ):
            for well_layer in well_layers:
                plot_concentration_at_position(
                    folder,
                    x=well_column - 1,
                    y=well_row - 1,
                    layer=well_layer - 1,
                    timestep_duration=transport_timestep_duration,
                )
                plot_head_at_position(
                    folder,
                    x=well_column - 1,
                    y=well_row - 1,
                    layer=well_layer - 1,
                    timestep_duration=flow_timestep_duration,
                )

    for (well_row, well_column), well_layers in zip(
        parameters["well_location"], parameters["well_layer"]
    ):
        for well_layer in well_layers:
            curve_workspace = Path(model_ws) / "simulations"
            x = well_column - 1
            y = well_row - 1
            layer = well_layer - 1
            plot_breakthrough_curve(
                curve_workspace,
                x=x,
                y=y,
                layer=layer,
                timestep_duration=transport_timestep_duration,
            )
            plot_breakthrough_curve(
                curve_workspace,
                x=well_column - 3,
                y=y,
                layer=layer,
                timestep_duration=transport_timestep_duration,
            )
            plot_head_curve(
                curve_workspace,
                x=x,
                y=y,
                layer=layer,
                timestep_duration=flow_timestep_duration,
            )
            plot_breakthrough_curve_html(
                curve_workspace,
                x,
                y,
                layer=layer,
                timestep_duration=transport_timestep_duration,
            )
            plot_breakthrough_curve_html(
                curve_workspace,
                well_column - 3,
                y,
                layer=layer,
                timestep_duration=transport_timestep_duration,
            )
            plot_head_curve_html(
                curve_workspace,
                x,
                y,
                layer=layer,
                timestep_duration=flow_timestep_duration,
            )
            plot_deposited_mass_curve(curve_workspace, x, y, layer=layer)
            plot_deposited_mass_curve_html(curve_workspace, x, y, layer=layer)
            plot_hydraulic_conductivity_curve(
                curve_workspace, x, y, layer=layer
            )
            plot_hydraulic_conductivity_curve_html(
                curve_workspace, x, y, layer=layer
            )

if __name__ == "__main__":
    main()