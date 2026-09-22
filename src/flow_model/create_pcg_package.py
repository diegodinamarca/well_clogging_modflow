import flopy


def create_pcg_package(mf):
    """Create the MODFLOW PCG solver package."""
    pcg = flopy.modflow.ModflowPcg(mf)
    print("PCG package created")
    return pcg