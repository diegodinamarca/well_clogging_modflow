import flopy


def create_adv_package(mt, mixelm, percel):
    """Create the MT3DMS advection package."""
    adv = flopy.mt3d.Mt3dAdv(
        mt,
        mixelm=mixelm,
        percel=percel
    )
    print("ADV package created")
    return adv