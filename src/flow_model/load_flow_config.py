  # -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:05:39 2026

@author: rappe
"""

import os
import tomllib
from pathlib import Path

import numpy as np

from .create_hyd_params import create_hyd_params


def load_flow_config(config_path=None):
    """Load model inputs from TOML and build the derived flow configuration."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.toml"
    config_path = Path(config_path)
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    config_dir = config_path.parent

    def configured_path(section, key):
        path = Path(config[section][key])
        return str(path if path.is_absolute() else config_dir / path)

    executables = config["executables"]
    grid = config["grid"]
    time = config["time"]
    hydrology = config["hydrology"]
    transport = config["transport"]
    runtime = config["runtime"]
    folders = config["folders"]
    methods = config["methods"]

    # ===== PATHS =====
    exe_name_mf = executables["modflow"]
    exe_name_mt = executables["mt3dms"]
    model_ws = folders["model_workspace"]
    modelname_mf = folders["modelname_mf"]
    modelname_mt = folders["modelname_mt"]

    # ===== SPATIAL DISCRETIZATION =====

    ztop = grid["ztop"]
    zbot = grid["zbot"]
    nlay = grid["nlay"]
    nrow = grid["nrow"]
    ncol = grid["ncol"]
    delr = grid["delr"]
    delc = grid["delc"]
    
    # ===== INITIAL HEAD =====
    strt = np.ones((nlay, nrow, ncol), dtype=np.float32) * 25.0
    strt[:, :, 0] = 25.0
    strt[:, :, -1] = 24.5
    
    # ===== BOUNDARY CONDITIONS =====
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, :, 0] = -1
    ibound[:, :, -1] = -1
    ibound[:, 0, :] = -1
    ibound[:, -1, :] = -1

    # ===== WELL INPUT ===== #

    # ===== HYDRAULIC PARAMETERS =====
    hk, vka, ss, laytyp = create_hyd_params(grid, nlay, nrow, ncol, method=methods["conductivity"])

    # ===== TRANSPORT PARAMS (MT3DMS) =====
    prsity = transport["porosity"]
    # ===== RECHARGE ====== #
    rech = hydrology["recharge_infiltration_factor"]

    # ===== TIME DISCRETIZATION =====
    nper_mf = time["nper_mf"]
    nper_mt = time["nper_mt"]
    perlen_mf = time["perlen_mf"]
    perlen_mt = time["perlen_mt"]
    nstp_mf = time["nstp_mf"]
    nstp_mt = time["nstp_mt"]
    steady_mf = time["steady_mf"]

    
    # ===== DISPERSION =====
    al = transport["longitudinal_dispersivity"]
    trpt = transport["horizontal_transverse_dispersivity"]
    trpv = transport["vertical_transverse_dispersivity"]

    # ===== CONCENTRATION =====
    ncomp = transport["ncomp"]
    c0 = transport["initial_concentration"]
    sconc = transport["initial_sorbed_concentration"]
    

    # periods for injection of contaminants (not used)
    active_periods = [c0, 0.0]
    # active_periods = [c0]*100+[0.0]*100

    # ===== ARRAYS =====
    icbund = np.ones((nlay, nrow, ncol), dtype=np.int32)
    sconc = np.zeros((nlay, nrow, ncol), dtype=np.float32)
    
    # load initial concentrations to array
    sconc[:, :, :] = c0  # Set all cells to the specified concentration value

    # ===== OBSERVATION POINT =====
    obs_row = transport["observation_row"]
    obs_col = transport["observation_column"]

    # ===== SOURCE LOCATION =====
    source_row = transport["source_row"]
    source_col = transport["source_column"]

    # ===== SORPTION / REACTION =====
    rhob = transport["bulk_density"]
    kd = transport["distribution_coefficient"]
    lambda1 = transport["decay_rate"]
    isothm = transport["isothm"]
    ireact = transport["ireact"]
    igetsc = transport["igetsc"]

    # ===== FINAL DICTIONARY =====
    param_dic = {

        # --- executables ---
        "exe_name_mf": exe_name_mf,
        "exe_name_mt": exe_name_mt,

        # --- folders ---
        "model_ws": model_ws,
        "modelname_mf": modelname_mf,
        "modelname_mt": modelname_mt,
        # --- input files ---

        # --- grid ---
        "nlay": nlay,
        "nrow": nrow,
        "ncol": ncol,
        "ztop": ztop,
        "zbot": zbot,
        "delr": delr,
        "delc": delc,
        # --- well ---
        # "wel_spd": wel_spd,

        # --- time ---
        "nper_mf": nper_mf,
        "nper_mt": nper_mt,
        "perlen_mf": perlen_mf,
        "perlen_mt": perlen_mt,
        "nstp_mf": nstp_mf,
        "nstp_mt": nstp_mt,
        "steady_mf": steady_mf,

        # --- boundary & initial ---
        "ibound": ibound,
        "strt": strt,
        # --- hydraulic ---
        "laytyp": laytyp,
        "hk": hk,
        "vka": vka,
        "ss": ss,

        # --- transport ---
        "prsity": prsity,
        
        # --- recharge ---
        "rech" : rech,

        # --- optional ---
        "ipakcb": runtime["ipakcb"],
        "output_file_name": runtime["output_file_name"],
        "compact_oc": runtime["compact_oc"],
        "silent": runtime["silent"],
        "report": runtime["report"],

        # # --- metadata (no usado directamente por flopy pero útil) ---
        # "Lx": Lx,
        # "Ly": Ly,

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

        # --- observation ---
        "obs_row": obs_row,
        "obs_col": obs_col,

        # --- reaction / sorption ---
        "rhob": rhob,
        "kd": kd,
        "lambda1": lambda1,
        "isothm": isothm,
        "ireact": ireact,
        "igetsc": igetsc,

        # --- optional / defaults ---
        "ncomp": ncomp,
        "mixelm": transport.get("mixelm", 0),
        "percel": transport.get("percel", 0.75),
        "source_itype": transport.get("source_itype", 1),

        # source active first period only (default behavior)
        "source_active_periods": active_periods

    }

    return param_dic