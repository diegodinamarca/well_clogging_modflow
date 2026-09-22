



param_dic = load_flow_config(conc_values = conc_values, randomFields = True, pumpRate = 10,
                             randomConcFields = True, heterogeneousClay = True, addWells = True)
print("Config loaded successfully")


model_ws ="./model_no_sorption_NAPL_1m"
modelname_mf= "flow_model"
modelname_mt = 'transport_no_sorption'


print("Running Transport model...")
# ===== MODEL NAME =====
mt, success_mt, buff_mt = create_transport_model(
    modelname_mt=modelname_mt,
    model_ws=model_ws,
    mf=mf,
    param_dic=param_dic,
    kd = 0.001, lambda1 = 0.0, isothm = 0, ireact = 0, igetsc = 0
)
print(f"Transport model creation completed. Success: {success_mt}")
