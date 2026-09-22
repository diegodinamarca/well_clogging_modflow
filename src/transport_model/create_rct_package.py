import flopy


def create_rct_package(mt, isothm, ireact, igetsc, rhob, kd, alpha, lambda1):
    """Create the MT3DMS reaction and sorption package."""
    rct = flopy.mt3d.Mt3dRct(
        mt,
        isothm=isothm,
        ireact=ireact,
        igetsc=igetsc,
        rhob=rhob,
        sp1=kd,
        sp2=alpha,
        rc1=lambda1,
        rc2=lambda1
    )
    print("RCT package created")
    return rct