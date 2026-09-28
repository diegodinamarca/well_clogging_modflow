from flow_model.load_flow_config import load_flow_config
from transport_model.create_transport_model import create_transport_model

def run_transport_model(config_path, mf = None):
    """Run the transport model using the provided configuration."""
    param_dic = load_flow_config(config_path=config_path)
    print("Config loaded successfully")
    # read model_ws, modelname_mf, modelname_mt from param_dic
    model_ws = param_dic.get("model_ws")
    modelname_mf = param_dic.get("modelname_mf")
    modelname_mt = param_dic.get("modelname_mt")

    #if mf is None run flow model first
    if mf is None:
        from flow_model.run_flow_model import run_flow_model
        mf, success, buff = run_flow_model(config_path)
        if not success:
            print("Flow model failed to run. Cannot proceed with transport model.")
            return None, False, None
    
    print("Running Transport model...")
    mt, success_mt, buff_mt = create_transport_model(
        modelname_mt=modelname_mt,
        model_ws=model_ws,
        mf=mf,
        param_dic=param_dic,
        kd=0.001, lambda1=0.0, isothm=0, ireact=0, igetsc=0
    )
    print(f"Transport model creation completed. Success: {success_mt}")
    return mf,mt, success_mt, buff_mt