import numpy as np


def create_hyd_params(grid, nlay, nrow, ncol, k_values=None, method = "avgK"):
    """Create hydraulic parameters for the MODFLOW layer-property package.

    Parameters
    ----------
    grid : dict
        Grid configuration containing hydraulic settings.
    nlay, nrow, ncol : int
        Number of model layers, rows, and columns.
    k_values : array-like, optional
        Hydraulic conductivity values, one per layer. When omitted, values
        are read from ``grid["k_values"]``.

    Returns
    -------
    tuple
        Horizontal conductivity, vertical conductivity, specific storage,
        and layer types.
    """
    if method == "avgK":
        print("Using average hydraulic conductivity method.")
        if k_values is None:
            k_values = grid["k_values"]
            if k_values is None:
                raise ValueError("Hydraulic conductivity values must be provided in the grid configuration or as argument k_values.")

        k_values = np.asarray(k_values, dtype=float)
        if k_values.ndim != 1 or len(k_values) != nlay:
            raise ValueError(
                f"Number of hydraulic conductivity values ({len(k_values)}) "
                f"must equal nlay ({nlay})."
            )

        hk = np.ones((nlay, nrow, ncol), dtype=float) * k_values[:, None, None]
        vka = hk * grid["vertical_conductivity_percent"] / 100
        ss = grid["specific_storage"]
        laytyp = grid["laytyp"]
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented for hydraulic parameter creation.")

    return hk, vka, ss, laytyp