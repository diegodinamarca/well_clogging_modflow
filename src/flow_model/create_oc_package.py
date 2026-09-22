import flopy


def create_oc_package(mf, nper, nstp_list, compact_oc):
    """Create the MODFLOW output-control package."""
    stress_period_data = {}
    for kper in range(nper):
        for kstp in range(nstp_list[kper]):
            stress_period_data[(kper, kstp)] = ["save head", "save budget"]

    oc = flopy.modflow.ModflowOc(
        mf,
        stress_period_data=stress_period_data,
        compact=compact_oc
    )
    print("OC package created")
    return oc