from pathlib import Path

from .load_flow_config import load_flow_config
from .create_flow_model import create_flow_model


def run_flow_model(param_dic = None):
    """Run the flow model using the provided configuration."""
    if param_dic is None:
        raise ValueError("param_dic must be provided to run the flow model.")
    mf, success, buff = create_flow_model(
        modelname_mf=param_dic["modelname_mf"],
        model_ws=param_dic["model_ws"],
        param_dic=param_dic
    )
    return mf, success, buff

if __name__ == "__main__":
    config_path = Path.cwd() / "config.toml"
    param_dic = load_flow_config(config_path=config_path)
    print("Config loaded successfully")
    mf, success, buff = run_flow_model(param_dic)
    print(f"Flow model creation completed. Success: {success}")
