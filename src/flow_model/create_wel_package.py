import flopy


def create_wel_package(mf, wel_spd, ipakcb):
    """Create the MODFLOW well package when stress-period data is provided."""
    if wel_spd is None:
        return None

    print("Well_spd value found in param_dic.", wel_spd)
    print("Adding well package with specified stress period data.")
    return flopy.modflow.ModflowWel(
        mf,
        stress_period_data=wel_spd,
        ipakcb=ipakcb
    )