"""Parameter-sweep execution and visualization for the clogging workflow."""

import json
import math
from numbers import Integral, Real
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

from flow_model.load_flow_config import load_flow_config
from plot_clogging_utils import (
    get_simulation_folders,
    read_deposited_mass_file,
    read_hydraulic_conductivity,
)
from run_clogging_workflow import initial_conditions, run_clogging_workflow
from utils import (
    _regular_timestep_duration,
    get_concentration_times,
    get_head_times,
    read_concentration_file,
    read_head_file,
)


SENSITIVITY_PARAMETERS = (
    "k_linear_slope",
    "deposition_rate",
    "initial_source_concentration",
    "well_pumping_rate",
)
_METADATA_FILENAME = "sensitivity_metadata.json"


def generate_parameter_values(min_value, max_value, n_intervals):
    """Generate equally spaced values including both range endpoints."""
    if (
        isinstance(min_value, bool)
        or not isinstance(min_value, Real)
        or not math.isfinite(min_value)
    ):
        raise ValueError("min_value must be a finite number.")
    if (
        isinstance(max_value, bool)
        or not isinstance(max_value, Real)
        or not math.isfinite(max_value)
    ):
        raise ValueError("max_value must be a finite number.")
    if min_value > max_value:
        raise ValueError("min_value must be less than or equal to max_value.")
    if (
        isinstance(n_intervals, bool)
        or not isinstance(n_intervals, Integral)
        or n_intervals <= 0
    ):
        raise ValueError("n_intervals must be a positive integer.")
    return np.linspace(
        float(min_value), float(max_value), int(n_intervals) + 1
    ).tolist()


def _format_value(value):
    return format(float(value), ".12g")


def _resolve_config_path(output_folder):
    folder_config = output_folder / "config.toml"
    if folder_config.is_file():
        return folder_config.resolve()
    working_config = Path.cwd() / "config.toml"
    if working_config.is_file():
        return working_config.resolve()
    raise FileNotFoundError(
        f"Configuration file not found at {folder_config} or {working_config}."
    )


def _validate_parameter_name(parameter_name):
    if parameter_name not in SENSITIVITY_PARAMETERS:
        raise ValueError(
            f"Unsupported sensitivity parameter {parameter_name!r}; choose one "
            f"of {', '.join(SENSITIVITY_PARAMETERS)}."
        )


def _base_parameter_value(param_dic, parameter_name):
    if parameter_name == "well_pumping_rate":
        well_locations = param_dic.get("well_location")
        rates = param_dic.get("well_pumping_rate")
        if not isinstance(well_locations, (list, tuple)) or len(well_locations) != 1:
            raise ValueError(
                "Sensitivity analysis for well_pumping_rate supports exactly "
                "one configured well."
            )
        if not isinstance(rates, (list, tuple)) or len(rates) != 1:
            raise ValueError(
                "well_pumping_rate must contain exactly one rate for the "
                "configured well."
            )
        value = rates[0]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("Configured well_pumping_rate must be finite and numeric.")
        return float(value)

    value = param_dic.get(parameter_name)
    if parameter_name == "initial_source_concentration":
        if value is None and "c0" in param_dic:
            value = param_dic["c0"]
        if isinstance(value, Real) and not isinstance(value, bool):
            candidates = [float(value)]
        elif isinstance(value, (list, tuple)) and value:
            if any(
                isinstance(item, bool)
                or not isinstance(item, Real)
                or not math.isfinite(item)
                for item in value
            ):
                raise ValueError(
                    "Configured initial_source_concentration must contain "
                    "finite numeric values."
                )
            candidates = [float(item) for item in value]
        else:
            raise ValueError(
                "Configured initial_source_concentration must be a finite "
                "number or a non-empty sequence of finite numbers."
            )
        if all(
            math.isclose(candidates[0], item, rel_tol=1e-9, abs_tol=1e-12)
            for item in candidates[1:]
        ):
            return candidates if isinstance(value, (list, tuple)) else candidates[0]
        return candidates

    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise TypeError(
            f"Configured parameter {parameter_name!r} must be a finite numeric "
            f"scalar; got {type(value).__name__}."
        )
    return float(value)


def _base_matches_value(base_value, value):
    if isinstance(base_value, Real):
        return math.isclose(
            float(base_value), float(value), rel_tol=1e-9, abs_tol=1e-12
        )
    if isinstance(base_value, (list, tuple)) and base_value:
        return all(
            math.isclose(float(item), float(value), rel_tol=1e-9, abs_tol=1e-12)
            for item in base_value
        )
    return False


def _base_folder_name(parameter_name, base_value):
    if isinstance(base_value, Real):
        value_text = _format_value(base_value)
    else:
        value_text = "__".join(_format_value(value) for value in base_value)
    return f"{parameter_name}_{value_text}"


def _base_case_label(parameter_name, base_value):
    if isinstance(base_value, Real):
        value_text = _format_value(base_value)
    elif all(
        math.isclose(base_value[0], value, rel_tol=1e-9, abs_tol=1e-12)
        for value in base_value[1:]
    ):
        value_text = _format_value(base_value[0])
    else:
        value_text = "[" + ", ".join(_format_value(value) for value in base_value) + "]"
    return f"{parameter_name}={value_text} (base case)"


def _apply_parameter_value(param_dic, parameter_name, value):
    if parameter_name == "well_pumping_rate":
        well_locations = param_dic.get("well_location")
        rates = param_dic.get("well_pumping_rate")
        if not isinstance(well_locations, (list, tuple)) or len(well_locations) != 1:
            raise ValueError(
                "Sensitivity analysis for well_pumping_rate supports exactly "
                "one configured well."
            )
        if not isinstance(rates, (list, tuple)) or len(rates) != 1:
            raise ValueError(
                "well_pumping_rate must contain exactly one rate for the "
                "configured well."
            )
        param_dic["well_pumping_rate"] = [float(value)]
        return

    if parameter_name == "initial_source_concentration":
        nper = param_dic.get("nper_mt")
        if isinstance(nper, bool) or not isinstance(nper, Integral) or nper <= 0:
            raise ValueError(
                "nper_mt must be a positive integer for "
                "initial_source_concentration sensitivity."
            )
        concentration = float(value)
        param_dic["initial_source_concentration"] = [concentration] * int(nper)
        if "c0" in param_dic:
            param_dic["c0"] = concentration
        if "source_active_periods" in param_dic:
            param_dic["source_active_periods"] = [concentration] * int(nper)
        return

    if parameter_name not in param_dic:
        raise ValueError(
            f"Parameter {parameter_name!r} is not present in the loaded "
            "model parameters."
        )
    original = param_dic[parameter_name]
    if isinstance(original, bool) or not isinstance(original, Real):
        raise TypeError(
            f"Parameter {parameter_name!r} must be a numeric scalar; "
            f"got {type(original).__name__}."
        )
    param_dic[parameter_name] = float(value)


def _experiment_names(parameter_name, values):
    counts = {}
    names = []
    labels = []
    for value in values:
        value_text = _format_value(value)
        counts[value_text] = counts.get(value_text, 0) + 1
        run_number = counts[value_text]
        suffix = "" if run_number == 1 else f"__repeat_{run_number}"
        names.append(f"{parameter_name}_{value_text}{suffix}")
        label = f"{parameter_name}={value_text}"
        if run_number > 1:
            label += f" (run {run_number})"
        labels.append(label)
    return names, labels


def _plot_metadata(param_dic):
    def _json_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [_json_value(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _timestep_duration(model):
        configured = param_dic.get("timestep_duration")
        if configured is not None:
            duration = float(configured)
        else:
            period_lengths = param_dic[f"perlen_{model}"]
            steps = param_dic[f"nstp_{model}"]
            if not period_lengths or not steps:
                raise ValueError(
                    f"Cannot determine the configured {model} timestep duration."
                )
            duration = float(period_lengths[0]) / int(steps[0])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(
                f"The configured {model} timestep duration must be positive."
            )
        return duration

    return {
        "modelname_mf": param_dic["modelname_mf"],
        "modelname_mt": param_dic["modelname_mt"],
        "well_location": _json_value(param_dic["well_location"]),
        "well_layer": _json_value(param_dic["well_layer"]),
        "flow_timestep_duration": _timestep_duration("mf"),
        "transport_timestep_duration": _timestep_duration("mt"),
    }


def _write_metadata(path, metadata):
    path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_sensitivity(
    parameter_name,
    min_value=None,
    max_value=None,
    n_intervals=None,
    value_list=None,
    folder=None,
    overwrite=True
):
    """Run parameter sensitivity cases against a single configuration base case."""
    if folder is None and isinstance(value_list, (str, Path)):
        folder = value_list
        value_list = None
    _validate_parameter_name(parameter_name)
    if value_list is not None:
        if not isinstance(value_list, (list, tuple)) or not value_list:
            raise ValueError("value_list must be a non-empty list or tuple.")
        values = []
        for item in value_list:
            if isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(item):
                raise ValueError(f"Invalid value in value_list: {item!r}.")
            values.append(float(item))
    else:
        values = generate_parameter_values(min_value, max_value, n_intervals)
    
    output_folder = Path.cwd() if folder is None else Path(folder)
    output_folder = output_folder.resolve()
    config_path = _resolve_config_path(output_folder)

    initial_parameters = initial_conditions(config_path)
    base_value = _base_parameter_value(initial_parameters, parameter_name)
    base_folder_name = _base_folder_name(parameter_name, base_value)
    base_folder = output_folder / base_folder_name

    experiment_names, experiment_labels = _experiment_names(
        parameter_name, values
    )
    experiment_paths = [
        base_folder if _base_matches_value(base_value, value) else output_folder / name
        for value, name in zip(values, experiment_names)
    ]
    metadata_path = output_folder / _METADATA_FILENAME
    existing = [path for path in [base_folder, *experiment_paths] if path.exists()]

    if existing and (not overwrite):
        raise FileExistsError(
            f"Sensitivity output already exists: {existing[0]}. "
            "Choose a new output folder to avoid overwriting results."
        )
    if metadata_path.exists() and (not overwrite):
        raise FileExistsError(
            f"Sensitivity metadata already exists: {metadata_path}. "
            "Choose a new output folder to avoid overwriting results."
        )

    output_folder.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "parameter_name": parameter_name,
        "values": [float(value) for value in values],
        "base_case": {
            "value": base_value,
            "folder": base_folder_name,
            "status": "pending",
        },
        "config_path": str(config_path),
        **_plot_metadata(initial_parameters),
        "experiments": [
            {
                "value": float(value),
                "folder": (
                    base_folder_name
                    if _base_matches_value(base_value, value)
                    else name
                ),
                "label": label,
                "status": (
                    "skipped_base_value"
                    if _base_matches_value(base_value, value)
                    else "pending"
                ),
            }
            for value, name, label in zip(
                values, experiment_names, experiment_labels
            )
        ],
    }
    _write_metadata(metadata_path, metadata)

    if not base_folder.exists() or overwrite:
        base_parameters = initial_conditions(config_path)
        base_parameters["model_ws"] = base_folder
        base_folder.mkdir(parents=True, exist_ok=True)
        metadata["base_case"]["status"] = "running"
        _write_metadata(metadata_path, metadata)
        result = run_clogging_workflow(config_path, base_parameters)
        if not isinstance(result, dict) or result.get("success") is not True:
            metadata["base_case"]["status"] = "failed"
            metadata["base_case"]["result"] = (
                {
                    key: result[key]
                    for key in ("failed_iteration", "workspace")
                    if key in result
                }
                if isinstance(result, dict)
                else {"result": repr(result)}
            )
            _write_metadata(metadata_path, metadata)
            failed_iteration = metadata["base_case"]["result"].get(
                "failed_iteration"
            )
            iteration_detail = (
                f" failed_iteration={failed_iteration}"
                if failed_iteration is not None
                else ""
            )
            raise RuntimeError(
                f"Sensitivity base-case run{iteration_detail} failed in "
                f"{metadata['base_case']['result'].get('workspace', base_folder)}."
            )
        metadata["base_case"]["status"] = "completed"
        metadata["base_case"]["iteration_count"] = len(
            result.get("iterations", [])
        )
    else:
        metadata["base_case"]["status"] = "reused_existing"
    _write_metadata(metadata_path, metadata)

    completed = []
    for value, experiment_path, experiment in zip(
        values, experiment_paths, metadata["experiments"]
    ):
        if _base_matches_value(base_value, value):
            continue
        param_dic = initial_conditions(config_path)
        _apply_parameter_value(param_dic, parameter_name, value)
        param_dic["model_ws"] = experiment_path
        experiment_path.mkdir(parents=True, exist_ok=True)
        experiment["status"] = "running"
        _write_metadata(metadata_path, metadata)
        result = run_clogging_workflow(config_path, param_dic)

        if not isinstance(result, dict) or result.get("success") is not True:
            experiment["status"] = "failed"
            experiment["result"] = (
                {
                    key: result[key]
                    for key in ("failed_iteration", "workspace")
                    if key in result
                }
                if isinstance(result, dict)
                else {"result": repr(result)}
            )
            _write_metadata(metadata_path, metadata)
            failed_workspace = experiment["result"].get(
                "workspace", experiment_path
            )
            failed_iteration = experiment["result"].get("failed_iteration")
            iteration_detail = (
                f" failed_iteration={failed_iteration}"
                if failed_iteration is not None
                else ""
            )
            raise RuntimeError(
                f"Sensitivity run {experiment['label']}{iteration_detail} failed in "
                f"{failed_workspace}."
            )

        experiment["status"] = "completed"
        experiment["iteration_count"] = len(result.get("iterations", []))
        completed.append(str(experiment_path))
        _write_metadata(metadata_path, metadata)

    return {
        "success": True,
        "parameter_name": parameter_name,
        "values": [float(value) for value in values],
        "experiment_folders": completed,
        "base_case_folder": str(base_folder),
    }


def _discover_experiments(folder):
    metadata_path = folder / _METADATA_FILENAME
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(
                f"Unable to read sensitivity metadata at {metadata_path}: {error}"
            ) from error
        if not isinstance(metadata, dict):
            raise ValueError(f"Sensitivity metadata in {metadata_path} must be an object.")
        if metadata.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported sensitivity metadata version in {metadata_path}."
            )
        if metadata.get("parameter_name") not in SENSITIVITY_PARAMETERS:
            raise ValueError(
                f"Invalid parameter name in sensitivity metadata {metadata_path}."
            )
        recorded_experiments = metadata.get("experiments")
        if not isinstance(recorded_experiments, list):
            raise ValueError(
                f"Experiments in sensitivity metadata {metadata_path} must be a list."
            )
        failed = [
            experiment
            for experiment in recorded_experiments
            if not isinstance(experiment, dict)
            or experiment.get("status")
            not in ("completed", "skipped_base_value")
        ]
        if failed:
            details = ", ".join(
                (
                    f"{item.get('label', item.get('folder'))}: "
                    f"{item.get('status', 'unknown')}"
                )
                if isinstance(item, dict)
                else repr(item)
                for item in failed
            )
            raise RuntimeError(
                f"Cannot plot incomplete sensitivity runs: {details}."
            )
        if not recorded_experiments:
            raise ValueError(f"No experiments recorded in {metadata_path}.")
        base_case = metadata.get("base_case")
        if base_case is not None:
            if not isinstance(base_case, dict):
                raise ValueError(f"Base case in {metadata_path} must be an object.")
            if base_case.get("status") not in ("completed", "reused_existing"):
                raise RuntimeError(
                    f"Cannot plot incomplete sensitivity base case: "
                    f"{base_case.get('status', 'unknown')}."
                )
            base_folder = base_case.get("folder")
            if (
                not isinstance(base_folder, str)
                or Path(base_folder).name != base_folder
                or base_folder in (".", "..")
            ):
                raise ValueError(
                    f"Invalid base-case folder in {metadata_path}: "
                    f"{base_folder!r}."
                )
            base_value = base_case.get("value")
            valid_base_value = isinstance(base_value, Real) and not isinstance(
                base_value, bool
            )
            valid_base_value = valid_base_value and math.isfinite(base_value)
            valid_base_value = valid_base_value or (
                isinstance(base_value, list)
                and bool(base_value)
                and all(
                    isinstance(item, Real)
                    and not isinstance(item, bool)
                    and math.isfinite(item)
                    for item in base_value
                )
            )
            if not valid_base_value:
                raise ValueError(
                    f"Invalid base-case value in {metadata_path}: "
                    f"{base_value!r}."
                )
        for experiment in recorded_experiments:
            experiment_folder = experiment.get("folder")
            value = experiment.get("value")
            if (
                not isinstance(experiment_folder, str)
                or Path(experiment_folder).name != experiment_folder
                or experiment_folder in (".", "..")
            ):
                raise ValueError(
                    f"Invalid experiment folder in sensitivity metadata "
                    f"{metadata_path}: {experiment_folder!r}."
                )
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"Invalid experiment value in sensitivity metadata "
                    f"{metadata_path}: {value!r}."
                )
        return metadata

    candidates = []
    for parameter_name in SENSITIVITY_PARAMETERS:
        prefix = f"{parameter_name}_"
        for child in folder.iterdir():
            if not child.is_dir() or not child.name.startswith(prefix):
                continue
            value_suffix = child.name[len(prefix) :].split("__repeat_", 1)[0]
            try:
                value = float(value_suffix)
            except ValueError:
                continue
            candidates.append(
                {
                    "parameter_name": parameter_name,
                    "value": value,
                    "folder": child.name,
                    "label": f"{parameter_name}={_format_value(value)}",
                    "status": "completed",
                }
            )
    parameter_names = {item["parameter_name"] for item in candidates}
    if len(parameter_names) != 1:
        raise ValueError(
            "The sensitivity folder must contain experiments for exactly one "
            "supported parameter."
        )
    candidates.sort(key=lambda item: (item["value"], item["folder"]))
    config_path = _resolve_config_path(folder)
    parameters = load_flow_config(config_path)
    return {
        "schema_version": 1,
        "parameter_name": candidates[0]["parameter_name"],
        "config_path": str(config_path),
        **_plot_metadata(parameters),
        "experiments": candidates,
    }


def _position_values(array, x, y, layer, path):
    if array.ndim != 3:
        raise ValueError(
            f"Spatial data in {path} must have shape (layer, row, column); "
            f"got {array.shape}."
        )
    if not 0 <= layer < array.shape[0]:
        raise IndexError(f"Layer {layer} is outside the grid in {path}.")
    if not 0 <= y < array.shape[1] or not 0 <= x < array.shape[2]:
        raise IndexError(f"Position (x={x}, y={y}) is outside the grid in {path}.")
    value = float(array[layer, y, x])
    if not math.isfinite(value):
        raise ValueError(f"Spatial value at (x={x}, y={y}, layer={layer}) in {path} is not finite.")
    return value


def _read_experiment(folder, metadata, experiment, x, y, layer):
    workspace = folder / experiment["folder"]
    simulation_folders = get_simulation_folders(workspace)
    flow_duration = metadata["flow_timestep_duration"]
    transport_duration = metadata["transport_timestep_duration"]
    head_parts = []
    concentration_parts = []
    head_times = []
    concentration_times = []
    conductivity = []
    deposited_mass = []
    grid_shape = None

    for simulation_folder in simulation_folders:
        head = np.asarray(
            read_head_file(simulation_folder, metadata["modelname_mf"])
        )
        concentration = np.asarray(
            read_concentration_file(simulation_folder, metadata["modelname_mt"])
        )
        if head.ndim != 4 or concentration.ndim != 4:
            raise ValueError(
                f"Head and concentration outputs in {simulation_folder} must "
                "have shape (time, layer, row, column)."
            )
        times_head = np.asarray(
            get_head_times(simulation_folder, metadata["modelname_mf"]),
            dtype=float,
        )
        times_concentration = np.asarray(
            get_concentration_times(simulation_folder, metadata["modelname_mt"]),
            dtype=float,
        )
        if (
            len(times_head) != head.shape[0]
            or len(times_concentration) != concentration.shape[0]
            or not len(times_head)
            or not len(times_concentration)
            or not np.all(np.isfinite(times_head))
            or not np.all(np.isfinite(times_concentration))
            or np.any(np.diff(times_head) <= 0)
            or np.any(np.diff(times_concentration) <= 0)
        ):
            raise ValueError(
                f"Head/concentration records and times are inconsistent in "
                f"{simulation_folder}."
            )
        if head.shape[1:] != concentration.shape[1:]:
            raise ValueError(
                f"Head and concentration grids differ in {simulation_folder}."
            )
        if not np.allclose(times_head, times_concentration):
            raise ValueError(
                f"Head and concentration time axes differ in {simulation_folder}."
            )
        if grid_shape is None:
            grid_shape = head.shape[1:]
        elif head.shape[1:] != grid_shape:
            raise ValueError(
                f"Simulation grid in {simulation_folder} differs from the "
                "preceding iteration."
            )
        for name, array, path in (
            ("head", head, simulation_folder / f"{metadata['modelname_mf']}.hds"),
            (
                "concentration",
                concentration,
                simulation_folder / f"{metadata['modelname_mt']}.ucn",
            ),
        ):
            if not 0 <= layer < array.shape[1]:
                raise IndexError(f"Layer {layer} is outside the {name} grid in {path}.")
            if not 0 <= y < array.shape[2] or not 0 <= x < array.shape[3]:
                raise IndexError(
                    f"Position (x={x}, y={y}) is outside the {name} grid in {path}."
                )
        head_values = head[:, layer, y, x].astype(float)
        concentration_values = concentration[:, layer, y, x].astype(float)
        if (
            not np.all(np.isfinite(head_values))
            or np.any(head_values < -1.0e20)
            or not np.all(np.isfinite(concentration_values))
        ):
            raise ValueError(
                f"Head or concentration at (x={x}, y={y}, layer={layer}) "
                f"contains invalid values in {simulation_folder}."
            )
        head_parts.append(head_values)
        concentration_parts.append(concentration_values)
        head_times.append(times_head)
        concentration_times.append(times_concentration)

        mass_array = np.asarray(read_deposited_mass_file(simulation_folder))
        conductivity_array = np.asarray(
            read_hydraulic_conductivity(
                simulation_folder, metadata["modelname_mf"]
            )
        )
        if mass_array.shape != grid_shape or conductivity_array.shape != grid_shape:
            raise ValueError(
                f"Deposited-mass or conductivity grid in {simulation_folder} "
                "does not match the flow/transport grid."
            )
        deposited_mass.append(
            _position_values(
                mass_array, x, y, layer, simulation_folder / "deposited_mass.npy"
            )
        )
        conductivity.append(
            _position_values(
                conductivity_array,
                x,
                y,
                layer,
                simulation_folder / "mfsim.nam",
            )
        )

    head_delta = _regular_timestep_duration(head_times[0], flow_duration)
    concentration_delta = _regular_timestep_duration(
        concentration_times[0], transport_duration
    )
    adjusted_head_times = [head_times[0]]
    adjusted_concentration_times = [concentration_times[0]]
    for times in head_times[1:]:
        adjusted_head_times.append(
            times - times[0] + adjusted_head_times[-1][-1] + head_delta
        )
    for times in concentration_times[1:]:
        adjusted_concentration_times.append(
            times
            - times[0]
            + adjusted_concentration_times[-1][-1]
            + concentration_delta
        )
    times = np.concatenate(adjusted_head_times)
    concentration_all_times = np.concatenate(adjusted_concentration_times)
    if not np.allclose(times, concentration_all_times):
        raise ValueError(
            f"Flow and transport time axes differ for {experiment['label']}."
        )
    iteration_ends = np.cumsum([len(part) for part in head_parts])
    iteration_times = times[iteration_ends - 1]
    return {
        "value": float(experiment["value"]),
        "label": experiment.get(
            "label",
            f"{metadata['parameter_name']}={_format_value(experiment['value'])}",
        ),
        "folder": workspace,
        "grid_shape": grid_shape,
        "times": times,
        "head": np.concatenate(head_parts),
        "concentration": np.concatenate(concentration_parts),
        "iteration_times": iteration_times,
        "deposited_mass": np.asarray(deposited_mass),
        "hydraulic_conductivity": np.asarray(conductivity),
    }


def _ensure_matching_times(experiments, key):
    expected = experiments[0][key]
    for experiment in experiments[1:]:
        actual = experiment[key]
        if actual.shape != expected.shape or not np.allclose(actual, expected):
            raise ValueError(
                f"Time axes differ between sensitivity experiments for {key}."
            )


def _calculate_output_ratio(sensitivity, base, variable, label):
    sensitivity = np.asarray(sensitivity, dtype=float)
    base = np.asarray(base, dtype=float)
    if sensitivity.shape != base.shape:
        raise ValueError(
            f"{variable} output shapes differ between {label} and the base case: "
            f"{sensitivity.shape} != {base.shape}."
        )
    if not np.all(np.isfinite(sensitivity)) or not np.all(np.isfinite(base)):
        raise ValueError(
            f"{variable} outputs for {label} and the base case must be finite."
        )

    undefined = base == 0
    ratio = np.full(sensitivity.shape, np.nan, dtype=float)
    np.divide(sensitivity, base, out=ratio, where=~undefined)
    if np.any(undefined):
        warnings.warn(
            f"{variable} ratio for {label} has {int(np.count_nonzero(undefined))} "
            "undefined sample(s) because the base-case output is zero; those "
            "samples are shown as gaps.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not np.all(np.isfinite(ratio[~undefined])):
        raise ValueError(
            f"{variable} ratio for {label} contains non-finite values."
        )
    return ratio


def _save_ratio_plots(
    output_folder, experiments, base_experiment, parameter_name, x, y, layer
):
    figures_folder = output_folder / "figures"
    figures_folder.mkdir(parents=True, exist_ok=True)
    position = f"row_{y + 1}_column_{x + 1}_layer_{layer + 1}"
    variables = (
        ("head", "Hydraulic head ratio", "hydraulic_head_ratio"),
        ("concentration", "Concentration ratio", "concentration_ratio"),
        ("deposited_mass", "Deposited mass ratio", "deposited_mass_ratio"),
        (
            "hydraulic_conductivity",
            "Hydraulic conductivity ratio",
            "hydraulic_conductivity_ratio",
        ),
    )
    for variable, ylabel, filename in variables:
        times_key = (
            "iteration_times"
            if variable in ("deposited_mass", "hydraulic_conductivity")
            else "times"
        )
        figure, axis = plt.subplots()
        ratio_experiments = []
        for experiment in experiments:
            if experiment["grid_shape"] != base_experiment["grid_shape"]:
                raise ValueError(
                    f"Spatial grids differ between {experiment['label']} and "
                    "the base case; ratio outputs cannot be aligned."
                )
            if experiment[times_key].shape != base_experiment[times_key].shape:
                raise ValueError(
                    f"{variable} time axes differ between {experiment['label']} "
                    "and the base case."
                )
            if not np.allclose(
                experiment[times_key], base_experiment[times_key]
            ):
                raise ValueError(
                    f"{variable} time axes differ between {experiment['label']} "
                    "and the base case."
                )
            ratio = _calculate_output_ratio(
                experiment[variable],
                base_experiment[variable],
                variable,
                experiment["label"],
            )
            ratio_experiments.append(
                {
                    "label": experiment["label"],
                    "times": experiment[times_key],
                    "iteration_times": experiment[times_key],
                    variable: ratio,
                }
            )
            axis.plot(
                experiment[times_key],
                ratio,
                label=experiment["label"],
            )
        axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xlabel("Time (hours)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} at {position.replace('_', ' ')}")
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            figures_folder
            / f"{filename}_{parameter_name}_{position}.png",
            dpi=160,
        )
        plt.close(figure)
        stem = f"{filename}_{parameter_name}_{position}"
        _save_interactive_comparison(
            figures_folder / f"{stem}.html",
            ratio_experiments,
            variable,
            f"{ylabel} at {position.replace('_', ' ')}",
            ylabel,
        )


def _save_interactive_comparison(path, experiments, variable, title, ylabel):
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "Interactive sensitivity plots require Plotly. Install it with "
            "`pip install -r requirements-plotting.txt`."
        ) from error

    figure = go.Figure()
    for experiment in experiments:
        times = (
            experiment["iteration_times"]
            if variable in ("deposited_mass", "hydraulic_conductivity")
            else experiment["times"]
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=experiment[variable],
                mode="lines+markers",
                name=experiment["label"],
                hovertemplate=(
                    f"Parameter: {experiment['label']}<br>"
                    "Time: %{x}<br>"
                    f"{ylabel}: %{{y}}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="Time (hours)",
        yaxis_title=ylabel,
        hovermode="closest",
    )
    figure.write_html(path, include_plotlyjs="directory", full_html=True)


def _save_comparison_plots(output_folder, experiments, parameter_name, x, y, layer):
    figures_folder = output_folder / "figures"
    figures_folder.mkdir(parents=True, exist_ok=True)
    position = f"row_{y + 1}_column_{x + 1}_layer_{layer + 1}"
    variables = (
        ("head", "Hydraulic head (model units)", "hydraulic_head"),
        ("concentration", "Concentration (model units)", "concentration"),
        ("deposited_mass", "Deposited mass (model units)", "deposited_mass"),
        (
            "hydraulic_conductivity",
            "Hydraulic conductivity (model units)",
            "hydraulic_conductivity",
        ),
    )
    for variable, ylabel, filename in variables:
        figure, axis = plt.subplots()
        times_key = (
            "iteration_times"
            if variable in ("deposited_mass", "hydraulic_conductivity")
            else "times"
        )
        for experiment in experiments:
            axis.plot(
                experiment[times_key],
                experiment[variable],
                label=experiment["label"],
            )
        axis.set_xlabel("Time (hours)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} at {position.replace('_', ' ')}")
        axis.legend()
        figure.tight_layout()
        stem = f"{filename}_{parameter_name}_{position}"
        figure.savefig(figures_folder / f"{stem}.png", dpi=160)
        plt.close(figure)
        _save_interactive_comparison(
            figures_folder / f"{stem}.html",
            experiments,
            variable,
            f"{ylabel} at {position.replace('_', ' ')}",
            ylabel,
        )


def _save_relationship_plots(
    output_folder, experiment, parameter_name, x, y, layer, base_experiment=None
):
    figures_folder = output_folder / "figures"
    position = f"row_{y + 1}_column_{x + 1}_layer_{layer + 1}"
    comparisons = (
        ("concentration", "Concentration (model units)", "concentration"),
        ("hydraulic_head", "Hydraulic head (model units)", "head"),
        ("deposited_mass", "Deposited mass (model units)", "deposited_mass"),
    )
    related_experiments = (
        [base_experiment, experiment]
        if base_experiment is not None
        else [experiment]
    )
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "Interactive sensitivity plots require Plotly. Install it with "
            "`pip install -r requirements-plotting.txt`."
        ) from error

    for variable, ylabel, series_key in comparisons:
        base = (
            f"hydraulic_conductivity_vs_{variable}_{parameter_name}_"
            f"{_format_value(experiment['value'])}_{position}"
        )
        figure, axis = plt.subplots()
        right_axis = axis.twinx()
        for related in related_experiments:
            conductivity = related["hydraulic_conductivity"]
            iteration_times = related["iteration_times"]
            series = related[series_key]
            if variable == "deposited_mass":
                line_times = iteration_times
                scatter_values = series
            else:
                line_times = related["times"]
                scatter_values = series[
                    np.searchsorted(related["times"], iteration_times)
                ]
            if len(scatter_values) != len(conductivity):
                raise ValueError(
                    f"{variable} and hydraulic-conductivity iteration counts "
                    f"differ for {related['label']}."
                )
            axis.plot(
                iteration_times,
                conductivity,
                marker="o",
                label=f"{related['label']}: hydraulic conductivity",
            )
            right_axis.plot(
                line_times,
                series,
                marker="." if len(line_times) == len(series) else "o",
                label=f"{related['label']}: {ylabel}",
            )
        axis.set_xlabel("Time (hours)")
        axis.set_ylabel("Hydraulic conductivity (model units)")
        right_axis.set_ylabel(ylabel)
        axis.set_title(f"{experiment['label']}: conductivity and {variable}")
        lines, labels = axis.get_legend_handles_labels()
        right_lines, right_labels = right_axis.get_legend_handles_labels()
        axis.legend(lines + right_lines, labels + right_labels, loc="best")
        figure.tight_layout()
        figure.savefig(figures_folder / f"{base}_line.png", dpi=160)
        plt.close(figure)

        line_figure = go.Figure()
        for related in related_experiments:
            conductivity = related["hydraulic_conductivity"]
            iteration_times = related["iteration_times"]
            series = related[series_key]
            line_times = (
                iteration_times if variable == "deposited_mass" else related["times"]
            )
            line_figure.add_trace(
                go.Scatter(
                    x=iteration_times,
                    y=conductivity,
                    mode="lines+markers",
                    name=f"{related['label']}: hydraulic conductivity",
                    hovertemplate=(
                        f"Parameter: {related['label']}<br>Time: %{{x}}<br>"
                        "Hydraulic conductivity: %{y}<extra></extra>"
                    ),
                )
            )
            line_figure.add_trace(
                go.Scatter(
                    x=line_times,
                    y=series,
                    mode="lines+markers",
                    name=f"{related['label']}: {ylabel}",
                    yaxis="y2",
                    hovertemplate=(
                        f"Parameter: {related['label']}<br>Time: %{{x}}<br>"
                        f"{ylabel}: %{{y}}<extra></extra>"
                    ),
                )
            )
        line_figure.update_layout(
            title=f"{experiment['label']}: conductivity and {variable}",
            xaxis_title="Time (hours)",
            yaxis_title="Hydraulic conductivity (model units)",
            yaxis2={
                "title": ylabel,
                "overlaying": "y",
                "side": "right",
            },
            hovermode="closest",
        )
        line_figure.write_html(
            figures_folder / f"{base}_line.html",
            include_plotlyjs="directory",
            full_html=True,
        )

        scatter_figure, scatter_axis = plt.subplots()
        for related in related_experiments:
            conductivity = related["hydraulic_conductivity"]
            iteration_times = related["iteration_times"]
            series = related[series_key]
            scatter_values = (
                series
                if variable == "deposited_mass"
                else series[np.searchsorted(related["times"], iteration_times)]
            )
            scatter_axis.scatter(
                conductivity, scatter_values, label=related["label"]
            )
        scatter_axis.set_xlabel("Hydraulic conductivity (model units)")
        scatter_axis.set_ylabel(ylabel)
        scatter_axis.set_title(f"{experiment['label']}: conductivity vs {variable}")
        scatter_axis.legend()
        scatter_figure.tight_layout()
        scatter_figure.savefig(figures_folder / f"{base}_scatter.png", dpi=160)
        plt.close(scatter_figure)

        interactive_scatter = go.Figure()
        for related in related_experiments:
            conductivity = related["hydraulic_conductivity"]
            iteration_times = related["iteration_times"]
            series = related[series_key]
            scatter_values = (
                series
                if variable == "deposited_mass"
                else series[np.searchsorted(related["times"], iteration_times)]
            )
            interactive_scatter.add_trace(
                go.Scatter(
                    x=conductivity,
                    y=scatter_values,
                    mode="markers",
                    name=related["label"],
                    customdata=iteration_times,
                    hovertemplate=(
                        f"Parameter: {related['label']}<br>"
                        "Time: %{customdata}<br>"
                        "Hydraulic conductivity: %{x}<br>"
                        f"{ylabel}: %{{y}}<extra></extra>"
                    ),
                )
            )
        interactive_scatter.update_layout(
            title=f"{experiment['label']}: conductivity vs {variable}",
            xaxis_title="Hydraulic conductivity (model units)",
            yaxis_title=ylabel,
        )
        interactive_scatter.write_html(
            figures_folder / f"{base}_scatter.html",
            include_plotlyjs="directory",
            full_html=True,
        )


def plot_sensitivity(folder):
    """Create static and interactive plots from completed sensitivity results."""
    output_folder = Path(folder).resolve()
    if not output_folder.is_dir():
        raise FileNotFoundError(
            f"Sensitivity output folder not found: {output_folder}"
        )
    metadata = _discover_experiments(output_folder)
    well_locations = metadata.get("well_location")
    well_layers = metadata.get("well_layer")
    if (
        not isinstance(well_locations, list)
        or not well_locations
        or not isinstance(well_layers, list)
        or not well_layers
    ):
        raise ValueError(
            "Sensitivity metadata/configuration must contain a well location "
            "and layer to select the default plot position."
        )
    if len(well_locations[0]) != 2 or not well_layers[0]:
        raise ValueError("Configured injection well position/layers are invalid.")
    y = int(well_locations[0][0]) - 1
    x = int(well_locations[0][1]) - 1
    layer = int(well_layers[0][0]) - 1
    sensitivity_metadata = [
        experiment
        for experiment in metadata["experiments"]
        if experiment.get("status", "completed") == "completed"
    ]
    if not sensitivity_metadata:
        raise ValueError(
            f"No non-base sensitivity experiments are available in {output_folder}."
        )
    experiments = [
        _read_experiment(output_folder, metadata, experiment, x, y, layer)
        for experiment in sensitivity_metadata
    ]
    if not experiments:
        raise ValueError(f"No completed experiments found in {output_folder}.")

    _ensure_matching_times(experiments, "times")
    _ensure_matching_times(experiments, "iteration_times")
    base_experiment = None
    if "base_case" in metadata:
        base_case = metadata["base_case"]
        base_experiment = _read_experiment(
            output_folder,
            metadata,
            {
                "value": (
                    base_case["value"][0]
                    if isinstance(base_case["value"], list)
                    else base_case["value"]
                ),
                "folder": base_case["folder"],
                "label": _base_case_label(
                    metadata["parameter_name"], base_case["value"]
                ),
            },
            x,
            y,
            layer,
        )
        _ensure_matching_times([base_experiment, *experiments], "times")
        _ensure_matching_times([base_experiment, *experiments], "iteration_times")
    plotted_experiments = (
        [base_experiment, *experiments]
        if base_experiment is not None
        else experiments
    )
    _save_comparison_plots(
        output_folder,
        plotted_experiments,
        metadata["parameter_name"],
        x,
        y,
        layer,
    )
    if base_experiment is not None:
        _save_ratio_plots(
            output_folder,
            experiments,
            base_experiment,
            metadata["parameter_name"],
            x,
            y,
            layer,
        )
    for experiment in experiments:
        _save_relationship_plots(
            output_folder,
            experiment,
            metadata["parameter_name"],
            x,
            y,
            layer,
            base_experiment,
        )
    return output_folder / "figures"


if __name__ == "__main__":
    """Run sensitivity analysis from the command line."""
    """Usage with parameter range: python sensitivity_workflow.py <parameter_name> <min_value> <max_value> <n_intervals> [--folder <output_folder>] [--overwrite]"""
    """Usage with list of values: python sensitivity_workflow.py <parameter_name> --value_list <value1,value2,...> [--folder <output_folder>] [--overwrite]"""
    import argparse
    print("Starting sensitivity analysis workflow...")
    parser = argparse.ArgumentParser(
        description="Run sensitivity analysis for clogging simulations."
    )
    parser.add_argument(
        "parameter_name",
        choices=SENSITIVITY_PARAMETERS,
        help="The parameter to vary in the sensitivity analysis.",
    )
    parser.add_argument(
        "--min_value",
        type=float,
        help="The minimum value of the parameter range.",
    )
    parser.add_argument(
        "--max_value",
        type=float,
        help="The maximum value of the parameter range.",
    )
    parser.add_argument(
        "--n_intervals",
        type=int,
        help="The number of intervals to divide the range into (number of runs = n_intervals + 1).",
    )
    parser.add_argument(
        "--value_list",
        type=str,
        default=None,
        help="Comma-separated list of values for the parameter (alternative to range).",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Output folder for sensitivity results (default: current working directory).",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=True,
        help="Overwrite existing results if they exist.",
    )
    parser.add_argument(
        "--plot",
        type=bool,
        default=True,
        help="Generate plots for the sensitivity analysis results.",
    )

    args = parser.parse_args()
    print(f"Running sensitivity analysis for parameter: {args.parameter_name}")
    if args.value_list:
        args.value_list = [float(v) for v in args.value_list.split(",")]
        print(f"Using value list: {args.value_list}")
    else:
        print(f"Using parameter range: {args.min_value} to {args.max_value} with {args.n_intervals} intervals.")
    
    run_sensitivity(
        args.parameter_name,
        args.min_value,
        args.max_value,
        args.n_intervals,
        args.value_list,
        folder=args.folder,
        overwrite=args.overwrite,
    )
    print("Sensitivity analysis completed successfully.")
    print(f"Plot argument is set to: {args.plot}")
    if args.plot:
        print("Generating plots for sensitivity analysis results...")
        figures_folder = plot_sensitivity(args.folder or Path.cwd())
        print(f"Plots saved in: {figures_folder}")
