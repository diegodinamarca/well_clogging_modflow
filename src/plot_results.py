"""Plot concentration and head results for a range of model timesteps.

Usage from the project workspace::

    python src/plot_results.py 3 --ws test1

The positional argument is the number of timestep folders to process. The
optional ``--ws`` argument points to the folder containing ``timestep_0``,
``timestep_1``, and so on; it defaults to the current working directory.

Interactive HTML plots require the optional dependency in
``requirements-plotting.txt``.
"""

from plot_conc_utils import plot_first_concentration, plot_last_concentration
from plot_clogging_utils import (
    plot_deposited_mass_curve,
    plot_deposited_mass_curve_html,
    plot_hydraulic_conductivity_curve,
    plot_hydraulic_conductivity_curve_html,
    plot_breakthrough_curve_html,
    plot_head_curve_html,
)
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
    for simulation_i in range(args.n):
        # Each timestep folder contains the results to plot for that step.
        folder = Path(model_ws) / "simulations" / f"sim_{simulation_i}"
        print(folder)
        plot_first_and_last_concentration(folder)
        plot_first_and_last_head(folder)
        plot_flow_properties(
            folder,
            layer=0,
            mask=parameters["ibound"] != 0,
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