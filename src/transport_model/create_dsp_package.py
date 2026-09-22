import flopy


def create_dsp_package(mt, al, trpt, trpv):
    """Create the MT3DMS dispersion package."""
    dsp = flopy.mt3d.Mt3dDsp(
        mt,
        al=al,
        trpt=trpt,
        trpv=trpv
    )
    print("DSP package created")
    return dsp