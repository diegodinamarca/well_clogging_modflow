# This file will define a workflow for implementing
# MODFLOW inside a loop to update key features between iterations
# to simulate clogging and its effects on the flow and transport model.


from pathlib import Path

import flopy
import numpy as np
from utils import get_last_concentration, get_last_head
from flow_model.load_flow_config import load_flow_config
from flow_model.run_flow_model import run_flow_model
from transport_model.run_transport_model import run_transport_model

def calculate_porosity(porosity, deposited_mass):
    # Calculate new porosity based on deposited mass
    new_porosity = porosity - porosity * np.mean(deposited_mass)  # Example: 10% of deposited mass
    return new_porosity

def calculate_hydraulic_conductivity(
    porosity, reference_porosity, reference_conductivity, method="kozeny_carman",
    time = None, lamb = None, linear_slope=None, affected_cells=None
):
    if not method in ["kozeny_carman", "exp_decay", "lm"]:
        raise ValueError("Unsupported method for calculating hydraulic conductivity.")
    if not 0 < reference_porosity < 1:
        raise ValueError("Reference porosity must be between 0 and 1.")
    if np.any(porosity <= 0) or np.any(porosity >= 1):
        raise ValueError("Porosity values must be between 0 and 1.")
    if method == "kozeny_carman":
        return reference_conductivity * (porosity / reference_porosity) ** 3 * (
            (1 - reference_porosity) / (1 - porosity)
        ) ** 2
    if method == "exp_decay":
        if time is None or lamb is None:
            raise ValueError("Time and lambda must be provided for exponential decay method.")
        return reference_conductivity * np.exp(-lamb * time)
    if method == "lm":
        if linear_slope is None:
            raise ValueError("Parameter linear_slope must be provided for linear method.")
        reference_conductivity[affected_cells] *= linear_slope
        return reference_conductivity


def run_modflow(param_dic):
    mf, success, buff = run_flow_model(param_dic)
    print(f"MODFLOW 6 simulation completed. Success: {success}")
    if not success:
        failed_run_modflow(
            buff,
            source="flow",
            model_ws=param_dic["model_ws"],
        )
        return mf, None, False
    mt = mf.get_model(param_dic["modelname_mt"])
    return mf, mt, success

# If model failed write the buffer to a file for debugging add the source of the failure to the log
# buff is a list of strings that contains the error messages from the model run. The source parameter indicates whether the failure occurred in the flow or transport model.
def failed_run_modflow(buff, source="flow", model_ws="."):
    model_ws = Path(model_ws)
    if source == "flow":
        log_path = model_ws / "flow_model_error.log"
    elif source == "transport":
        log_path = model_ws / "transport_model_error.log"
    else:
        raise ValueError("Buffer source must be either 'flow' or 'transport'. Which model failed to run?")
    log_path.write_text("\n".join(buff))

def calculate_deposition(concentration, dep_rate=0.01):
    # Placeholder for deposition calculation logic
    deposited_mass = concentration * dep_rate
    return deposited_mass

def initial_conditions(config_path):
    # Placeholder for initial conditions setup
    param_dic = load_flow_config(config_path = config_path)
    return param_dic

def update_concentration(concentration, deposited_mass):
    new_concentration = concentration - deposited_mass
    return new_concentration

def read_hyraulic_conductivity_file():
    # Placeholder for reading hydraulic conductivity file logic
    pass

def read_porosity_file():
    # Placeholder for reading porosity file logic
    pass


def update_parameters_list(param_list, new_K, porosity, head, concentration):
    # Update the parameter list with new hydraulic conductivity
    param_list["hk"] = new_K
    param_list["porosity"] = porosity
    param_list["strt"] = head
    param_list["sconc"] = concentration
    return param_list


def set_model_workspace(param_dic, timestep, base_workspace=None):
    base_workspace = Path(base_workspace or param_dic["model_ws"])
    param_dic["model_ws"] = str(base_workspace / f"sim_{timestep}")
    print ("Model workspace set to:", param_dic["model_ws"]) 
    return param_dic


def run_clogging_workflow(
    config_path,
    workspace,
    parameter_value,
    k_update_method=None,
    iterations=3,
):
    """Run the complete clogging workflow in an independent workspace."""
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer.")

    param_dic = initial_conditions(config_path)
    base_workspace = Path(workspace).resolve()
    base_workspace.mkdir(parents=True, exist_ok=True)
    param_dic["model_ws"] = str(base_workspace)
    if k_update_method is None:
        k_update_method = param_dic.get("k_update_method", "exp_decay")
    param_dic["k_linear_slope"] = parameter_value
    iteration_results = []

    for iteration in range(iterations):
        param_dic = set_model_workspace(param_dic, iteration, base_workspace)
        _, _, success = run_modflow(param_dic)
        if not success:
            return {
                "success": False,
                "iterations": iteration_results,
                "failed_iteration": iteration,
                "workspace": param_dic["model_ws"],
            }
        concentration = get_last_concentration(
            param_dic["model_ws"], param_dic["modelname_mt"]
        )
        head = get_last_head(param_dic["model_ws"], param_dic["modelname_mf"])
        mean_k = float(np.mean(param_dic["hk"]))

        print (f"##################### Iteration {iteration} Summary ################")
        print(f"Porosity at iteration {iteration}: {round(param_dic['porosity'], 2)}")
        print(f"Mean hydraulic conductivity at iteration {iteration}: {round(mean_k, 2)}")
        print(f"Mean concentration at iteration {iteration}: {round(np.mean(concentration), 2)}")
        print (f"#####################################################################")

        deposited_mass = calculate_deposition(concentration, dep_rate=param_dic.get("deposition_rate", 0.01))
        concentration = update_concentration(concentration, deposited_mass)
        #porosity = calculate_porosity(param_dic["porosity"], deposited_mass)
        porosity = param_dic["porosity"]

        # get cells where deposited mass is greater than a threshold
        threshold = param_dic.get("deposition_threshold", 0.001)
        affected_cells = deposited_mass > threshold
        print(f"shape of affected_cells: {affected_cells.shape}, number of affected cells: {np.sum(affected_cells)}")
        # Reduce hydraulic conductivity in affected cells
        K = calculate_hydraulic_conductivity(
            porosity,
            param_dic["porosity"],
            param_dic["hk"],
            method=k_update_method,
            time=iteration,
            lamb=param_dic["lambdak"],
            linear_slope=parameter_value,
            affected_cells=affected_cells
        )
        param_dic = update_parameters_list(param_dic, K, porosity, head, concentration)
        iteration_results.append(
            {
                "iteration": iteration,
                "workspace": str(base_workspace / f"sim_{iteration}"),
                "mean_hydraulic_conductivity": mean_k,
                "mean_porosity": float(np.mean(porosity)),
            }
        )

    return {"success": True, "iterations": iteration_results}


def main():
    run_dir = Path.cwd().resolve()
    config_path = run_dir / "config.toml"
    param_dic = initial_conditions(config_path)
    return run_clogging_workflow(
        config_path,
        param_dic["model_ws"],
        param_dic.get("k_linear_slope"),
        k_update_method=param_dic.get("k_update_method", "lm"),
    )


if __name__ == "__main__":
    main()