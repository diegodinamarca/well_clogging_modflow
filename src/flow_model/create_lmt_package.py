import flopy


def create_lmt_package(mf, output_file_name):
    """Create the MODFLOW link-to-MT3DMS package."""
    lmt = flopy.modflow.ModflowLmt(
        mf,
        output_file_name=output_file_name
    )
    print("LMT package created - flow model is now linked to transport model")
    return lmt