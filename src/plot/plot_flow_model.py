param_dic = load_flow_config(conc_values = conc_values, randomFields = True, pumpRate = 10,
                             randomConcFields = True, heterogeneousClay = True, addWells = True)
print("Config loaded successfully")

model_ws ="./model_no_sorption_NAPL_1m"
modelname_mf= "flow_model"
modelname_mt = 'transport_no_sorption'

Lx = param_dic["Lx"]
Ly = param_dic["Ly"]
c0 = param_dic["c0"]
source_row = param_dic["source_row"]
source_col = param_dic["source_col"]
obs_row, obs_col = param_dic["obs_row"], param_dic["obs_col"]
delr = param_dic["delr"]
ks = param_dic["hk"]
Lx = param_dic["Lx"]
prsity = param_dic["prsity"]


plot_head(model_ws, modelname_mf, Lx, Ly, nlayer=0)
