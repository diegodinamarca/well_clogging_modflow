from .load_flow_config import load_flow_config
from .create_flow_model import create_flow_model


def run_flow_model(config_path):
    """Run the flow model using the provided configuration."""
    param_dic = load_flow_config(config_path = config_path)
    print("Config loaded successfully")
    mf, success, buff = create_flow_model(
        modelname_mf=param_dic["modelname_mf"],
        model_ws=param_dic["model_ws"],
        param_dic=param_dic
    )
    print(f"Flow model creation completed. Success: {success}")
    return mf, success, buff

if __name__ == "__main__":
    config_path = "./test1/config.toml"
    run_flow_model(config_path)