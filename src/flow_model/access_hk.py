import flopy

def get_hydraulic_conductivity(model_ws, modelname="flow_model"):
    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws, verbosity_level=0)
    gwf = sim.get_model(modelname)
    k = gwf.npf.k.array
    return k



if __name__ == "__main__":
    model_ws = "/home/diego/modflow/master_thesis/test1/simulations/sim_0"
    k = get_hydraulic_conductivity(model_ws)
    print(k)