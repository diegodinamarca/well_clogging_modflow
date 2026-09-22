import flopy


def create_bas_package(mf, ibound, strt):
    """Create the MODFLOW basic package."""
    bas = flopy.modflow.ModflowBas(
        mf,
        ibound=ibound,
        strt=strt
    )
    print("BAS package created")
    return bas