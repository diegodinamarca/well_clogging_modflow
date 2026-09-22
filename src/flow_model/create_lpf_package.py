import flopy


def create_lpf_package(mf, laytyp, hk, vka, ipakcb, ss):
    """Create the MODFLOW layer-property flow package."""
    lpf = flopy.modflow.ModflowLpf(
        mf,
        laytyp=laytyp,
        hk=hk,
        vka=vka,
        ipakcb=ipakcb,
        ss=ss
    )
    print("LPF package created")
    return lpf