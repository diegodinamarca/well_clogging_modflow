  # -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:05:39 2026

@author: rappe
"""

import tomllib
import warnings
from pathlib import Path

import numpy as np

from .create_hyd_params import create_hyd_params


def _create_refined_grid(nrow, ncol, delr, delc, well_locations, factor, radius):
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("Well refinement factor must be a positive integer.")
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise ValueError("Well refinement radius must be a positive number.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Well refinement radius must be a positive number.")
    if not np.isfinite(delr) or not np.isfinite(delc) or delr <= 0 or delc <= 0:
        raise ValueError("Grid cell widths delr and delc must be positive numbers.")

    refine_rows = np.zeros(nrow, dtype=bool)
    refine_columns = np.zeros(ncol, dtype=bool)
    row_centers = (np.arange(nrow) + 0.5) * delc
    column_centers = (np.arange(ncol) + 0.5) * delr

    for location in well_locations:
        if len(location) != 2:
            raise ValueError("Each well location must contain a row and column index.")
        row = int(location[0]) - 1
        column = int(location[1]) - 1
        if row < 0 or row >= nrow:
            raise ValueError(f"Configured well row index {location[0]} is outside 1..{nrow}.")
        if column < 0 or column >= ncol:
            raise ValueError(f"Configured well column index {location[1]} is outside 1..{ncol}.")
        refine_rows |= np.abs(row_centers - row_centers[row]) <= radius
        refine_columns |= np.abs(column_centers - column_centers[column]) <= radius

    row_counts = np.where(refine_rows, factor, 1)
    column_counts = np.where(refine_columns, factor, 1)

    row_parent = np.repeat(np.arange(nrow), row_counts)
    column_parent = np.repeat(np.arange(ncol), column_counts)

    row_starts = np.cumsum(np.r_[0, row_counts[:-1]])
    column_starts = np.cumsum(np.r_[0, column_counts[:-1]])
    
    row_index_map = row_starts + (row_counts - 1) // 2
    column_index_map = column_starts + (column_counts - 1) // 2

    refined_delc = np.repeat(delc / row_counts, row_counts)
    refined_delr = np.repeat(delr / column_counts, column_counts)
    if not np.any(refine_rows):
        refined_delc = delc
    if not np.any(refine_columns):
        refined_delr = delr

    return (
        refined_delr,
        refined_delc,
        row_parent,
        column_parent,
        row_index_map,
        column_index_map,
    )


def _map_grid_index(value, size, index_map, name):
    index = int(value) - 1
    if index < 0 or index >= size:
        raise ValueError(f"Configured {name} index {value} is outside 1..{size}.")
    return int(index_map[index]) + 1


def _create_circular_domain(nrow, ncol, well_location):
    """Create a discrete circular ibound mask centered on one coarse-grid well."""
    if len(well_location) != 2:
        raise ValueError("A circular domain requires one well row and column.")
    well_row, well_column = (int(value) - 1 for value in well_location)
    if not (0 <= well_row < nrow and 0 <= well_column < ncol):
        raise ValueError("Circular-domain well location is outside the coarse grid.")

    rows, columns = np.ogrid[:nrow, :ncol]
    radius = nrow / 2
    inside_circle = (
        (rows - well_row) ** 2 + (columns - well_column) ** 2 <= radius**2
    )

    padded_circle = np.pad(inside_circle, 1, constant_values=False)
    perimeter = np.zeros_like(inside_circle)
    for row_offset in range(3):
        for column_offset in range(3):
            if row_offset == 1 and column_offset == 1:
                continue
            perimeter |= inside_circle & ~padded_circle[
                row_offset : row_offset + nrow,
                column_offset : column_offset + ncol,
            ]

    domain = np.where(inside_circle, 1, 0).astype(np.int32)
    domain[perimeter] = -1
    return domain

def plot_circular_domain(mask):
            import matplotlib.pyplot as plt

            plt.imshow(mask[0], cmap="coolwarm", origin="upper")
            plt.colorbar(label="Ibound Value")
            plt.title("Circular Domain Ibound Mask")
            plt.xlabel("Column Index")
            plt.ylabel("Row Index")
            plt.show()

def load_flow_config(config_path=None):
    """Load model inputs from TOML and build the derived flow configuration."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config.toml"
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    executables = config["executables"]
    grid = config["grid"]
    time = config["time"]
    hydrology = config["hydrology"]
    transport = config["transport"]
    runtime = config["runtime"]
    folders = config["folders"]
    methods = config["methods"]
    methods_parameters = config["methods_parameters"]

    # ===== PATHS =====
    exe_name_mf = executables["modflow"]
    model_ws = str(config_path.parent / "simulations")
    modelname_mf = folders["modelname_mf"]
    modelname_mt = folders["modelname_mt"]

    # ===== SPATIAL DISCRETIZATION =====

    ztop = grid["ztop"]
    zbot = grid["zbot"]
    nlay = grid["nlay"]
    coarse_nrow = grid["nrow"]
    coarse_ncol = grid["ncol"]
    coarse_delr = grid["delr"]
    coarse_delc = grid["delc"]
    circular_domain = grid.get("circular_domain", False)
    if not isinstance(circular_domain, bool):
        raise ValueError("grid.circular_domain must be a TOML boolean.")

    # ===== WELL INPUT ===== #
    coarse_well_location = hydrology["well_location"]
    well_pumping_rate = hydrology["well_pumping_rate"]
    well_layer = hydrology["well_layer"]
    refinement_factor = grid.get("well_refinement_factor", 1)
    refinement_radius = grid.get(
        "well_refinement_radius", min(coarse_delr, coarse_delc)
    )
    (
        delr,
        delc,
        row_parent,
        column_parent,
        row_index_map,
        column_index_map,
    ) = _create_refined_grid(
        coarse_nrow,
        coarse_ncol,
        coarse_delr,
        coarse_delc,
        coarse_well_location,
        refinement_factor,
        refinement_radius,
    )
    nrow = len(row_parent)
    ncol = len(column_parent)
    well_location = [
        [
            _map_grid_index(row, coarse_nrow, row_index_map, "well row"),
            _map_grid_index(column, coarse_ncol, column_index_map, "well column"),
        ]
        for row, column in coarse_well_location
    ]

    # ===== BOUNDARY CONDITIONS =====
    if circular_domain and len(coarse_well_location) == 1:
        circular_mask = _create_circular_domain(
            coarse_nrow, coarse_ncol, coarse_well_location[0]
        )
        coarse_ibound = np.broadcast_to(
            circular_mask, (nlay, coarse_nrow, coarse_ncol)
        ).copy()
        # Plot circular domain for verification
        # plot_circular_domain(circular_mask)
    else:
        if circular_domain:
            warnings.warn(
                "Circular domain generation requires exactly one configured well; "
                "using the existing domain.",
                UserWarning,
                stacklevel=2,
            )
        coarse_ibound = np.ones((nlay, coarse_nrow, coarse_ncol), dtype=np.int32)
        coarse_ibound[:, :, 0] = -1
        coarse_ibound[:, :, -1] = -1
        coarse_ibound[:, 0, :] = 1
        coarse_ibound[:, -1, :] = 1
    ibound = np.take(
        np.take(coarse_ibound, row_parent, axis=1), column_parent, axis=2
    )


    # ===== INITIAL HEAD =====
    coarse_strt = np.ones((nlay, coarse_nrow, coarse_ncol), dtype=np.float32) * 25.0
    #coarse_strt[:, :, 0] = 25.0
    #coarse_strt[:, :, -1] = 24.5
    strt = np.take(np.take(coarse_strt, row_parent, axis=1), column_parent, axis=2)


    # ===== HYDRAULIC PARAMETERS =====
    hk, vka, ss, laytyp = create_hyd_params(grid, nlay, nrow, ncol, method=methods["conductivity"])

    # ===== TRANSPORT PARAMETERS =====
    prsity = transport["porosity"]
    
    # ===== RECHARGE ====== #
    recharge = hydrology["recharge"]

    # ===== TIME DISCRETIZATION =====
    nper_mf = time["nper_mf"]
    nper_mt = time["nper_mt"]
    perlen_mf = time["perlen_mf"]
    perlen_mt = time["perlen_mt"]
    nstp_mf = time["nstp_mf"]
    nstp_mt = time["nstp_mt"]
    steady_mf = time["steady_mf"]
    timestep_duration = time.get("timestep_duration")

    
    # ===== DISPERSION =====
    al = transport["longitudinal_dispersivity"]
    trpt = transport["horizontal_transverse_dispersivity"]
    trpv = transport["vertical_transverse_dispersivity"]

    # ===== CONCENTRATION =====
    ncomp = transport["ncomp"]
    gw_c0 = transport.get("initial_aquifer_concentration", 0.0)
    c0 = transport.get("initial_source_concentration", 0.0)
    active_periods = transport.get(
        "source_active_periods", [0.0] * time["nper_mt"]
    )

    # ===== ARRAYS =====
    icbund = np.ones((nlay, nrow, ncol), dtype=np.int32)
    sconc = np.zeros((nlay, nrow, ncol), dtype=np.float32)

    # load initial concentrations to array
    if isinstance(gw_c0, (int, float)):
        sconc[:, :, :] = gw_c0  # Set all cells to the specified concentration value
    elif isinstance(gw_c0, (list)):
        if len(gw_c0) != nlay:
            raise ValueError("Length of initial aquifer concentration list must match the number of layers (nlay).")
        for layer in range(nlay):
            # initial concentration array structure is [layer, row, col]
            sconc[layer, :, :] = gw_c0[layer]  # Set each layer to the specified concentration value
    elif isinstance(gw_c0, np.ndarray):
        if gw_c0.shape != (nlay, coarse_nrow, coarse_ncol):
            raise ValueError(
                "Shape of initial aquifer concentration array must match "
                "the configured coarse grid dimensions."
            )
        sconc = np.take(
            np.take(gw_c0.astype(np.float32), row_parent, axis=1),
            column_parent,
            axis=2,
        )
    else:
        raise TypeError("initial aquifer concentration must be an int, float, list, or numpy.ndarray. Check config file.")

    # ===== OBSERVATION POINT =====
    obs_row = _map_grid_index(
        transport["observation_row"], coarse_nrow, row_index_map, "observation row"
    )
    obs_col = _map_grid_index(
        transport["observation_column"],
        coarse_ncol,
        column_index_map,
        "observation column",
    )

    # ===== SOURCE LOCATION =====
    source_row = _map_grid_index(
        transport["source_row"], coarse_nrow, row_index_map, "source row"
    )
    source_col = _map_grid_index(
        transport["source_column"],
        coarse_ncol,
        column_index_map,
        "source column",
    )

    # ===== SORPTION / REACTION =====
    bulk_density = transport["bulk_density"]
    kd = transport["distribution_coefficient"]
    decay_rate = transport["decay_rate"]
    isothm = transport["isothm"]
    ireact = transport["ireact"]
    igetsc = transport["igetsc"]

    # ===== FINAL DICTIONARY =====
    param_dic = {

        # --- executables ---
        "exe_name_mf": exe_name_mf,

        # --- folders ---
        "model_ws": model_ws,
        "modelname_mf": modelname_mf,
        "modelname_mt": modelname_mt,

        # --- methods ---
        "k_update_method": methods["k_update_method"],

        # --- methods parameters ---
        "lambdak": methods_parameters["lambdak"],
        "k_linear_slope": methods_parameters.get("k_linear_slope"),

        # --- grid ---
        "nlay": nlay,
        "nrow": nrow,
        "ncol": ncol,
        "ztop": ztop,
        "zbot": zbot,
        "delr": delr,
        "delc": delc,

        # --- well ---
        "well_refinement_factor": refinement_factor,
        "well_refinement_radius": refinement_radius,
        "well_location": well_location,
        "well_pumping_rate": well_pumping_rate,
        "well_layer": well_layer,

        # --- time ---
        "nper_mf": nper_mf,
        "nper_mt": nper_mt,
        "perlen_mf": perlen_mf,
        "perlen_mt": perlen_mt,
        "nstp_mf": nstp_mf,
        "nstp_mt": nstp_mt,
        "steady_mf": steady_mf,
        "timestep_duration": timestep_duration,

        # --- boundary & initial ---
        "ibound": ibound,
        "strt": strt,

        # --- hydraulic ---
        "laytyp": laytyp,
        "hk": hk,
        "vka": vka,
        "ss": ss,

        # --- transport ---
        "porosity": prsity,
        "deposition_rate": transport.get("deposition_rate", 0.01),
        "deposition_threshold": transport.get("deposition_threshold", 0.001),
        
        # --- recharge ---
        "recharge": recharge,

        # --- optional ---
        "ipakcb": runtime["ipakcb"],
        "output_file_name": runtime["output_file_name"],
        "compact_oc": runtime["compact_oc"],
        "silent": runtime["silent"],
        "report": runtime["report"],

        # --- basic transport ---
        "icbund": icbund,
        "sconc": sconc,

        # --- dispersion ---
        "al": al,
        "trpt": trpt,
        "trpv": trpv,

        # --- source ---
        "c0": c0,
        "source_row": source_row,
        "source_col": source_col,
        "source_layer": transport.get("source_layer", 1),
        "source_active_periods": active_periods,

        # --- observation ---
        "obs_row": obs_row,
        "obs_col": obs_col,

        # --- reaction / sorption ---
        "bulk_density": bulk_density,
        "kd": kd,
        "decay_rate": decay_rate,
        "isothm": isothm,
        "ireact": ireact,
        "igetsc": igetsc,

        # --- optional / defaults ---
        "ncomp": ncomp,
        "mixelm": transport.get("mixelm", 0),
        "percel": transport.get("percel", 0.75),
        "source_itype": transport.get("source_itype", 1),        

    }

    return param_dic