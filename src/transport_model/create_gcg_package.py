import flopy


def create_gcg_package(mt):
    """Create the MT3DMS GCG solver package."""
    gcg = flopy.mt3d.Mt3dGcg(mt)
    print("GCG package created")
    return gcg