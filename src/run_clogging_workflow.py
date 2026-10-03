# This file will define a workflow for implementing
# MODFLOW inside a loop to update key features between iterations
# to simulate clogging and its effects on the flow and transport model.


from pathlib import Path

import flopy
import numpy as np
from utils import get_last_concentration
from flow_model.load_flow_config import load_flow_config
from flow_model.run_flow_model import run_flow_model
from transport_model.run_transport_model import run_transport_model

def calculate_porosity(porosity, deposited_mass):
    # Calculate new porosity based on deposited mass
    new_porosity = porosity -porosity * np.mean(deposited_mass)  # Example: 10% of deposited mass
    return new_porosity

def calculate_hydraulic_conductivity(
    porosity, reference_porosity, reference_conductivity, method="kozeny_carman"
):
    if method != "kozeny_carman":
        raise ValueError("Unsupported method for calculating hydraulic conductivity.")
    if not 0 < reference_porosity < 1:
        raise ValueError("Reference porosity must be between 0 and 1.")
    if np.any(porosity <= 0) or np.any(porosity >= 1):
        raise ValueError("Porosity values must be between 0 and 1.")
    return reference_conductivity * (porosity / reference_porosity) ** 3 * (
        (1 - reference_porosity) / (1 - porosity)
    ) ** 2


def run_modflow(param_dic):
    mf, success, buff = run_flow_model(param_dic)
    print(f"MODFLOW 6 simulation completed. Success: {success}")
    if not success:
        failed_run_modflow(buff, source="flow")
        return mf, None, buff, None, success, False
    mt = mf.get_model(param_dic["modelname_mt"])
    return mf, mt, success

# If model failed write the buffer to a file for debugging add the source of the failure to the log
# buff is a list of strings that contains the error messages from the model run. The source parameter indicates whether the failure occurred in the flow or transport model.
def failed_run_modflow(buff, source = "flow"):
    if source == "flow":
        with open("flow_model_error.log", "w") as f:
            # Write the buffer to the log file
            f.write("\n".join(buff))
    elif source == "transport":
        with open("transport_model_error.log", "w") as f:
            f.write("\n".join(buff))
    else:
        raise ValueError("Buffer source must be either 'flow' or 'transport'. Which model failed to run?")

def calculate_deposition(concentration):
    # Placeholder for deposition calculation logic
    deposited_mass = concentration * 0.1  # Example: 10% of concentration
    return deposited_mass

def initial_conditions(config_path):
    # Placeholder for initial conditions setup
    param_dic = load_flow_config(config_path = config_path)
    return param_dic

def update_concentration(param_dic, new_concentration):
    param_dic["sconc"] = new_concentration
    return param_dic

def read_hyraulic_conductivity_file():
    # Placeholder for reading hydraulic conductivity file logic
    pass

def read_porosity_file():
    # Placeholder for reading porosity file logic
    pass

def calculate_deposition(concentration):
    # Placeholder for calculating deposition logic
    deposited_mass = concentration * 0.01  # Example: 10% of concentration
    return deposited_mass

def update_parameters_list(param_list, new_K, porosity, concentration):
    # Update the parameter list with new hydraulic conductivity
    param_list["hk"] = new_K
    param_list["porosity"] = porosity
    param_list["concentration"] = concentration
    return param_list


def set_model_workspace(param_dic, timestep, base_workspace=None):
    base_workspace = Path(base_workspace or param_dic["model_ws"])
    param_dic["model_ws"] = str(base_workspace / f"timestep_{timestep}")
    print ("Model workspace set to:", param_dic["model_ws"]) 
    return param_dic


def main():
    run_dir = Path.cwd().resolve()
    config_path = run_dir / "config.toml"
    param_dic = initial_conditions(config_path)
    base_workspace = param_dic["model_ws"]

    # Number of iterations for each simulation. Each simulations runs for every timestep defined in the config.file
    simulation = range(3)

    for iter, _ in enumerate(simulation):
        
        param_dic = set_model_workspace(param_dic, iter, base_workspace)
        mf, mt, success = run_modflow(param_dic)
        if not success:
            print(f"Model failed at iteration {iter}. Check logs for details.")
            break
        concentration = get_last_concentration(
            param_dic["model_ws"], param_dic["modelname_mt"]
        )
        mean_k = np.mean(param_dic["hk"])

        print (f"##################### Iteration {iter} Summary ################")
        print(f"Porosity at iteration {iter}: {round(param_dic['porosity'], 2)}")
        print(f"Mean hydraulic conductivity at iteration {iter}: {round(mean_k, 2)}")
        print(f"Mean concentration at iteration {iter}: {round(np.mean(concentration), 2)}")
        print (f"#####################################################################")

        #print(f"Concentration at timestep {timestep}: {concentration}")
        param_dic = update_concentration(param_dic, concentration)
        # Here you can implement the logic to update parameters based on concentration
        # For example:
        deposited_mass = calculate_deposition(concentration)


        porosity = calculate_porosity(param_dic["porosity"], deposited_mass)
        K = calculate_hydraulic_conductivity(
             porosity, param_dic["porosity"], param_dic["hk"]
        )
        param_dic = update_parameters_list(param_dic, K, porosity, concentration)


if __name__ == "__main__":
    main()