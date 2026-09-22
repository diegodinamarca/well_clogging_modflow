import flopy


def create_dis_package(mf, nlay, nrow, ncol, delr, delc, ztop, zbot,
                       nper, perlen_list, nstp_list, steady_list):
    """Create the MODFLOW discretization package."""
    dis = flopy.modflow.ModflowDis(
        mf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=ztop,
        botm=zbot,
        nper=nper,
        perlen=perlen_list,
        nstp=nstp_list,
        steady=steady_list
    )
    print("DIS package created")
    print(f"Model grid: {nlay} layers, {nrow} rows, {ncol} columns")
    return dis