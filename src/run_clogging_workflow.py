# This file will define a workflow for implementing
# MODFLOW inside a loop to update key features between iterations
# to simulate clogging and its effects on the flow and transport model.


from pathlib import Path

from flow_model.load_flow_config import load_flow_config
from run_modflow import run_modflow

def calculate_porosity(deposited_mass):
    # Calculate new porosity based on deposited mass
    new_porosity = deposited_mass * 0.1  # Example: 10% of deposited mass
    return new_porosity

def calculate_hydraulic_conductivity(porosity, method = "kozeny_carman"):
    # Calculate new hydraulic conductivity based on updated porosity
    if method == "kozeny_carman":
        new_K = kozeny_carman(porosity)
    else:
        raise ValueError("Unsupported method for calculating hydraulic conductivity.")
    return new_K

def calculate_deposition(tss):
    # Placeholder for deposition calculation logic
    deposited_mass = tss.sum()  # Example: sum of TSS values
    return deposited_mass

def initial_conditions(config_path):
    # Placeholder for initial conditions setup
    param_dic = load_flow_config(config_path = config_path)

    return param_dic

def read_concentration_file():
    # Placeholder for reading concentration file logic
    pass

def read_hyraulic_conductivity_file():
    # Placeholder for reading hydraulic conductivity file logic
    pass

def read_porosity_file():
    # Placeholder for reading porosity file logic
    pass

def calculate_deposition(concentration):
    # Placeholder for calculating deposition logic
    deposited_mass = concentration * 0.1  # Example: 10% of concentration
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
    config_path = Path(__file__).resolve().parent.parent / "test1" / "config.toml"
    param_dic = initial_conditions(config_path)
    base_workspace = param_dic["model_ws"]
    simulation = range(3)
    for timestep, _ in enumerate(simulation):
        param_dic = set_model_workspace(param_dic, timestep, base_workspace)

        # mf, mt, buff, buff_mt = run_modflow(param_dic)
        # concentration = read_concentration_file()
        # deposited_mass = calculate_deposition(concentration)
        # porosity = calculate_porosity(deposited_mass)
        # K = calculate_hydraulic_conductivity(porosity)
        # param_dic = update_parameters_list(param_dic, K, porosity, concentration)


if __name__ == "__main__":
    main()