import flopy


def create_ssm_package(mt, nper, nrow, source_active_periods, source_itype,
                       source_row, source_col, c0):
    """Create the MT3DMS source/sink mixing package."""
    stress_period_data = {}
    for kper in range(nper):
        stress_period_data[kper] = [
            (0, row, 0, source_active_periods[kper], source_itype)
            for row in range(nrow)
        ]

    ssm = flopy.mt3d.Mt3dSsm(
        mt,
        stress_period_data=stress_period_data
    )
    print("SSM package created")
    print(f"Contaminant source at: row={source_row}, col={source_col}")
    print(f"Initial source concentration: {c0}")
    return ssm