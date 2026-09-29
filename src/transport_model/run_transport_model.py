from flow_model.load_flow_config import load_flow_config
from flow_model.run_flow_model import run_flow_model


def run_transport_model(param_dic=None, mf=None):
    """Return the GWT model from a coupled MODFLOW 6 simulation.

    When no simulation is supplied, both GWF and GWT are built and run because
    MODFLOW 6 solves the coupled models together.
    """
    if param_dic is None:
        raise ValueError("param_dic must be provided to run the transport model.")

    if mf is None:
        mf, success, buff = run_flow_model(param_dic)
        if not success:
            return mf, None, success, buff
    else:
        success, buff = True, []

    gwt = mf.get_model(param_dic["modelname_mt"])
    return mf, gwt, success, buff


if __name__ == "__main__":
    config_path = "./config.toml"
    param_dic = load_flow_config(config_path=config_path)
    simulation, gwt, success, output = run_transport_model(param_dic)
    print(f"Coupled MODFLOW 6 simulation completed. Success: {success}")