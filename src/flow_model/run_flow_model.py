param_dic = load_flow_config(conc_values = conc_values, randomFields = True, pumpRate = 10,
                             randomConcFields = True, heterogeneousClay = True, addWells = True)
print("Config loaded successfully")

model_ws ="./model_no_sorption_NAPL_1m"
modelname_mf= "flow_model"
modelname_mt = 'transport_no_sorption'

mf, success, buff = create_flow_model(
    modelname_mf=modelname_mf,
    model_ws=model_ws,
    param_dic=param_dic
)
print(f"Flow model creation completed. Success: {success}")


