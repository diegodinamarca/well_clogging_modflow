from .create_mf6_simulation import create_mf6_simulation


def create_flow_model(modelname_mf, model_ws, param_dic):
    """Write and run the coupled MODFLOW 6 flow and transport simulation."""
    if modelname_mf != param_dic["modelname_mf"]:
        raise ValueError("modelname_mf must match the name in param_dic.")
    sim, _, _ = create_mf6_simulation(param_dic, model_ws=model_ws)
    print(f"Running MODFLOW 6 simulation in {model_ws}...")
    success, buff = sim.run_simulation(
        silent=param_dic.get("silent", True),
        report=param_dic.get("report", True),
    )
    return sim, success, buff