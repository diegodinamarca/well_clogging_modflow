import flopy


def create_btn_package(mt, icbund, prsity, sconc_array, ncomp, obs_row,
                       obs_col, perlen_list, nper, timprs):
    """Create the MT3DMS basic transport package."""
    btn = flopy.mt3d.Mt3dBtn(
        mt,
        icbund=icbund,
        prsity=prsity,
        sconc=sconc_array,
        ncomp=ncomp,
        obs=[(0, obs_row, obs_col)],
        perlen=perlen_list,
        nper=nper,
        timprs=timprs,
        nprs=len(timprs)
    )
    print("BTN package created")
    print(f"Observation point at: row={obs_row}, col={obs_col}")
    return btn