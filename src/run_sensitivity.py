"""Run a parameter sensitivity analysis for hydraulic conductivity."""

import argparse
import csv
import math
from numbers import Real
from pathlib import Path
import sys
import tomllib

import matplotlib.pyplot as plt
import numpy as np

from flow_model.access_hk import get_hydraulic_conductivity
from flow_model.load_flow_config import load_flow_config
from run_clogging_workflow import run_clogging_workflow
from utils import (
    get_concentration_times,
    get_head_times,
    read_concentration_file,
    read_head_file,
)


SUPPORTED_PARAMETERS = ("n",)
WORKFLOW_ITERATIONS = 3
VARIABLES = ("concentration", "head", "hydraulic_conductivity")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        usage="%(prog)s [folder] <parameter_name>",
        description="Run a clogging sensitivity analysis.",
    )
    parser.add_argument("folder_or_parameter", nargs="?")
    parser.add_argument("parameter_name", nargs="?")
    arguments = parser.parse_args(argv)

    if arguments.folder_or_parameter is None:
        parser.error("parameter_name is required; supported parameters: n")
    if arguments.parameter_name is None:
        possible_folder = Path(arguments.folder_or_parameter)
        if (
            possible_folder.is_dir()
            or arguments.folder_or_parameter in (".", "..")
            or "/" in arguments.folder_or_parameter
            or "\\" in arguments.folder_or_parameter
        ):
            parser.error("parameter_name is required; supported parameters: n")
        folder = Path.cwd()
        parameter_name = arguments.folder_or_parameter
    else:
        folder = Path(arguments.folder_or_parameter)
        parameter_name = arguments.parameter_name

    if parameter_name not in SUPPORTED_PARAMETERS:
        parser.error(
            f"unsupported parameter {parameter_name!r}; supported parameters: "
            f"{', '.join(SUPPORTED_PARAMETERS)}"
        )
    if not folder.is_absolute():
        folder = Path.cwd() / folder
    return folder.resolve(), parameter_name


def _numeric_setting(settings, name, section):
    value = settings.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"[{section}].{name} must be numeric.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"[{section}].{name} must be finite.")
    return value


def load_sensitivity_settings(folder):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Simulation folder does not exist: {folder}")
    config_path = folder / "config.toml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    methods_parameters = config.get("methods_parameters")
    sensitivity = config.get("sensitivity")
    if not isinstance(methods_parameters, dict):
        raise ValueError("config.toml must define a [methods_parameters] table.")
    if not isinstance(sensitivity, dict):
        raise ValueError("config.toml must define a [sensitivity] table.")

    baseline = _numeric_setting(methods_parameters, "n", "methods_parameters")
    intervals_value = sensitivity.get("sensitivity_intervals")
    if (
        isinstance(intervals_value, bool)
        or not isinstance(intervals_value, int)
        or intervals_value <= 0
    ):
        raise ValueError("[sensitivity].sensitivity_intervals must be a positive integer.")
    percentage_minimum = _numeric_setting(
        sensitivity, "porcentage_minimum", "sensitivity"
    )
    percentage_maximum = _numeric_setting(
        sensitivity, "porcentage_maximum", "sensitivity"
    )
    if baseline <= 0:
        raise ValueError("[methods_parameters].n must be positive.")
    if percentage_minimum <= 0:
        raise ValueError("[sensitivity].porcentage_minimum must be positive.")
    if percentage_minimum >= percentage_maximum:
        raise ValueError(
            "[sensitivity].porcentage_minimum must be smaller than "
            "porcentage_maximum."
        )

    methods = config.get("methods")
    if not isinstance(methods, dict) or "k_update_method" not in methods:
        raise ValueError("config.toml must define methods.k_update_method.")
    if methods["k_update_method"] != "lm":
        raise ValueError(
            "Sensitivity analysis for n requires methods.k_update_method = 'lm'."
        )

    return {
        "config_path": config_path,
        "baseline": baseline,
        "intervals": intervals_value,
        "percentage_minimum": percentage_minimum,
        "percentage_maximum": percentage_maximum,
    }


def format_parameter_value(value):
    return repr(float(value))


def generate_parameter_values(
    baseline,
    intervals,
    percentage_minimum,
    percentage_maximum,
):
    minimum = baseline * percentage_minimum / 100
    maximum = baseline * percentage_maximum / 100
    values = np.linspace(minimum, maximum, intervals + 1, dtype=float).tolist()
    labels = [format_parameter_value(value) for value in values]
    if len(set(labels)) != len(labels):
        raise ValueError(
            "Generated parameter values are not uniquely representable; "
            "reduce sensitivity_intervals or adjust the range."
        )
    return values


def simulation_workspace(folder, parameter_name, value):
    return Path(folder) / f"{parameter_name}_{format_parameter_value(value)}"


def artifact_paths(folder, wells, layer_count):
    folder = Path(folder)
    paths = [folder / f"sensitivity_{variable}.csv" for variable in VARIABLES]
    for well_index in range(len(wells)):
        for layer_index in range(layer_count):
            for variable in VARIABLES:
                paths.append(
                    folder
                    / f"sensitivity_{variable}_well_{well_index + 1}_layer_{layer_index + 1}.png"
                )
    return paths


def preflight_outputs(folder, parameter_name, values, wells, layer_count):
    existing_workspaces = [
        simulation_workspace(folder, parameter_name, value)
        for value in values
        if simulation_workspace(folder, parameter_name, value).exists()
    ]
    if existing_workspaces:
        paths = ", ".join(str(path) for path in existing_workspaces)
        raise FileExistsError(f"Simulation results already exist: {paths}")
    existing_artifacts = [
        path for path in artifact_paths(folder, wells, layer_count) if path.exists()
    ]
    if existing_artifacts:
        paths = ", ".join(str(path) for path in existing_artifacts)
        raise FileExistsError(f"Sensitivity output files already exist: {paths}")


def execute_simulations(folder, config_path, parameter_name, values):
    workspaces = {}
    for value in values:
        workspace = simulation_workspace(folder, parameter_name, value).resolve()
        try:
            result = run_clogging_workflow(
                config_path=config_path,
                workspace=workspace,
                parameter_value=value,
                k_update_method="lm",
                iterations=WORKFLOW_ITERATIONS,
            )
        except Exception as error:
            raise RuntimeError(
                f"Simulation for n={format_parameter_value(value)} failed in "
                f"{workspace}: {error}"
            ) from error
        if not result.get("success"):
            failed_workspace = result.get("workspace", workspace)
            failed_iteration = result.get("failed_iteration", "unknown")
            raise RuntimeError(
                f"Simulation for n={format_parameter_value(value)} failed at "
                f"iteration {failed_iteration} in {failed_workspace}."
            )
        completed_iterations = result.get("iterations", [])
        if len(completed_iterations) != WORKFLOW_ITERATIONS:
            raise RuntimeError(
                f"Simulation for n={format_parameter_value(value)} in {workspace} "
                f"reported {len(completed_iterations)} completed iterations; "
                f"expected {WORKFLOW_ITERATIONS}."
            )
        workspaces[value] = workspace
    return workspaces


def extract_parameter_timeseries(value, workspace, model_parameters):
    workspace = Path(workspace)
    period_duration = float(sum(model_parameters["perlen_mf"]))
    if not math.isfinite(period_duration) or period_duration <= 0:
        raise ValueError("Configured total simulation period must be positive and finite.")
    expected_shape = (
        model_parameters["nlay"],
        model_parameters["nrow"],
        model_parameters["ncol"],
    )
    wells = model_parameters["well_location"]
    series = {
        variable: {
            (well_index, layer_index): []
            for well_index in range(len(wells))
            for layer_index in range(model_parameters["nlay"])
        }
        for variable in VARIABLES
    }
    all_times = []

    for iteration in range(WORKFLOW_ITERATIONS):
        iteration_workspace = workspace / f"sim_{iteration}"
        concentration = np.asarray(
            read_concentration_file(
                iteration_workspace, model_parameters["modelname_mt"]
            )
        )
        head = np.asarray(
            read_head_file(iteration_workspace, model_parameters["modelname_mf"])
        )
        concentration_times = np.asarray(
            get_concentration_times(
                iteration_workspace, model_parameters["modelname_mt"]
            ),
            dtype=float,
        )
        head_times = np.asarray(
            get_head_times(iteration_workspace, model_parameters["modelname_mf"]),
            dtype=float,
        )
        if (
            concentration_times.ndim != 1
            or concentration_times.size == 0
            or not np.all(np.isfinite(concentration_times))
            or head_times.shape != concentration_times.shape
            or not np.all(np.isfinite(head_times))
        ):
            raise ValueError(f"Invalid output times in {iteration_workspace}.")
        if not np.allclose(concentration_times, head_times):
            raise ValueError(
                f"Concentration and head output times differ in {iteration_workspace}."
            )
        if concentration.ndim != 4 or concentration.shape[1:] != expected_shape:
            raise ValueError(
                f"Concentration output in {iteration_workspace} has shape "
                f"{concentration.shape}; expected (time, {expected_shape})."
            )
        if head.ndim != 4 or head.shape[1:] != expected_shape:
            raise ValueError(
                f"Head output in {iteration_workspace} has shape {head.shape}; "
                f"expected (time, {expected_shape})."
            )
        if (
            concentration.shape[0] != concentration_times.size
            or head.shape[0] != head_times.size
        ):
            raise ValueError(f"Output time count does not match data in {iteration_workspace}.")

        conductivity = np.asarray(
            get_hydraulic_conductivity(
                iteration_workspace, model_parameters["modelname_mf"]
            )
        )
        if conductivity.shape != expected_shape:
            raise ValueError(
                f"Hydraulic-conductivity array in {iteration_workspace} has shape "
                f"{conductivity.shape}; expected {expected_shape}."
            )

        all_times.append(concentration_times + iteration * period_duration)
        for well_index, (row, column) in enumerate(wells):
            row_index = row - 1
            column_index = column - 1
            for layer_index in range(model_parameters["nlay"]):
                key = (well_index, layer_index)
                series["concentration"][key].append(
                    concentration[:, layer_index, row_index, column_index]
                )
                series["head"][key].append(
                    head[:, layer_index, row_index, column_index]
                )
                series["hydraulic_conductivity"][key].append(
                    np.full(
                        concentration_times.size,
                        conductivity[layer_index, row_index, column_index],
                    )
                )

    times = np.concatenate(all_times)
    if np.any(np.diff(times) < 0):
        raise ValueError(f"Simulation times are not chronological for n={value} in {workspace}.")
    return {
        "times": times,
        **{
            variable: {
                key: np.concatenate(iteration_values)
                for key, iteration_values in variable_series.items()
            }
            for variable, variable_series in series.items()
        },
    }


def validate_aligned_times(values, results):
    reference_times = results[values[0]]["times"]
    for value in values[1:]:
        times = results[value]["times"]
        if times.shape != reference_times.shape or not np.allclose(
            times, reference_times, rtol=1e-10, atol=1e-10
        ):
            raise ValueError(
                f"Time series for n={format_parameter_value(value)} do not align "
                "with the other sensitivity runs."
            )
    return reference_times


def series_label(value, well_index, layer_index):
    return (
        f"n={format_parameter_value(value)}_well={well_index + 1}_"
        f"layer={layer_index + 1}"
    )


def export_time_series(folder, values, results, wells, layer_count):
    folder = Path(folder)
    times = validate_aligned_times(values, results)
    for variable in VARIABLES:
        output_path = folder / f"sensitivity_{variable}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            headers = ["time"]
            for value in values:
                for well_index in range(len(wells)):
                    for layer_index in range(layer_count):
                        headers.append(series_label(value, well_index, layer_index))
            writer.writerow(headers)
            for time_index, time in enumerate(times):
                row = [time]
                for value in values:
                    for well_index in range(len(wells)):
                        for layer_index in range(layer_count):
                            row.append(
                                results[value][variable][(well_index, layer_index)][
                                    time_index
                                ]
                            )
                writer.writerow(row)


def plot_time_series(folder, values, results, wells, layer_count):
    folder = Path(folder)
    times = validate_aligned_times(values, results)
    axis_labels = {
        "concentration": "Concentration (model units)",
        "head": "Hydraulic head (model length units)",
        "hydraulic_conductivity": "Hydraulic conductivity (model length/day)",
    }
    plot_paths = []
    for well_index in range(len(wells)):
        for layer_index in range(layer_count):
            for variable in VARIABLES:
                figure, axis = plt.subplots()
                for value in values:
                    axis.plot(
                        times,
                        results[value][variable][(well_index, layer_index)],
                        label=f"n={format_parameter_value(value)}",
                    )
                axis.set_xlabel("Time (days)")
                axis.set_ylabel(axis_labels[variable])
                axis.set_title(
                    f"{variable.replace('_', ' ').title()} at well "
                    f"{well_index + 1}, layer {layer_index + 1}"
                )
                axis.legend(title="Parameter n")
                figure.tight_layout()
                output_path = folder / (
                    f"sensitivity_{variable}_well_{well_index + 1}_"
                    f"layer_{layer_index + 1}.png"
                )
                figure.savefig(output_path)
                plt.close(figure)
                plot_paths.append(output_path)
    return plot_paths


def run_sensitivity(folder, parameter_name):
    folder = Path(folder).resolve()
    if parameter_name not in SUPPORTED_PARAMETERS:
        raise ValueError(
            f"Unsupported parameter {parameter_name!r}; supported parameters: "
            f"{', '.join(SUPPORTED_PARAMETERS)}"
        )
    settings = load_sensitivity_settings(folder)
    values = generate_parameter_values(
        settings["baseline"],
        settings["intervals"],
        settings["percentage_minimum"],
        settings["percentage_maximum"],
    )
    model_parameters = load_flow_config(settings["config_path"])
    wells = model_parameters["well_location"]
    layer_count = model_parameters["nlay"]
    preflight_outputs(folder, parameter_name, values, wells, layer_count)

    workspaces = execute_simulations(
        folder,
        settings["config_path"],
        parameter_name,
        values,
    )
    results = {}
    for value in values:
        workspace = workspaces[value]
        try:
            results[value] = extract_parameter_timeseries(
                value, workspace, model_parameters
            )
        except Exception as error:
            raise RuntimeError(
                f"Result extraction for n={format_parameter_value(value)} failed "
                f"in {workspace}: {error}"
            ) from error
    validate_aligned_times(values, results)
    plot_time_series(folder, values, results, wells, layer_count)
    export_time_series(folder, values, results, wells, layer_count)
    return {"values": values, "workspaces": workspaces, "results": results}


def main(argv=None):
    try:
        folder, parameter_name = parse_arguments(argv)
        run_sensitivity(folder, parameter_name)
    except (FileNotFoundError, NotADirectoryError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())