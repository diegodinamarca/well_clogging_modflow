# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:17:58 2026

@author: rappe
"""

import flopy
import numpy as np
from transport_model.create_adv_package import create_adv_package
from transport_model.create_btn_package import create_btn_package
from transport_model.create_dsp_package import create_dsp_package
from transport_model.create_gcg_package import create_gcg_package
from transport_model.create_rct_package import create_rct_package
from transport_model.create_ssm_package import create_ssm_package

def create_transport_model(modelname_mt, model_ws, mf, param_dic,
                           kd = 0.001, alpha = 1, lambda1 = 0.0, isothm = 0, ireact = 0, igetsc = 0):
    """
    Create an MT3DMS transport model linked to an existing MODFLOW model.

    Parameters
    ----------
    modelname_mt : str
        Name of the MT3DMS model.
    model_ws : str
        Workspace folder where MT3DMS files will be written.
    mf : flopy.modflow.Modflow
        Previously created MODFLOW model object.
    param_dic : dict
        Dictionary containing all required MT3DMS transport parameters.

    Required keys in param_dic
    --------------------------
    exe_name_mt : str
    icbund : array-like
    prsity : float or array-like
    sconc : array-like
    obs_row : int
    obs_col : int
    perlen_list : list
    nper : int
    al : float
    trpt : float
    trpv : float
    c0 : float
    source_row : int
    source_col : int
    isothm : int
    ireact : int
    igetsc : int
    rhob : float
    kd : float
    lambda1 : float

    Optional keys
    -------------
    ncomp : int, default 1
    mixelm : int, default 0
    percel : float, default 0.75
    source_itype : int, default 1
    source_active_periods : list, default first period on, rest off

    Returns
    -------
    mt : flopy.mt3d.Mt3dms
        Created MT3DMS model object.
    """

    # ---------- Required parameters ----------
    exe_name_mt = param_dic["exe_name_mt"]

    icbund = param_dic["icbund"]
    prsity = param_dic["prsity"]
    sconc = param_dic["sconc"]

    obs_row = param_dic["obs_row"]
    obs_col = param_dic["obs_col"]

    perlen_list = param_dic["perlen_mt"]
    nper = param_dic["nper_mt"]

    al = param_dic["al"]
    trpt = param_dic["trpt"]
    trpv = param_dic["trpv"]

    c0 = param_dic["c0"]
    source_row = param_dic["source_row"]
    source_col = param_dic["source_col"]
    rhob = param_dic["rhob"]

    # ---------- Optional parameters ----------
    ncomp = param_dic.get("ncomp", 1)
    mixelm = param_dic.get("mixelm", 0)
    percel = param_dic.get("percel", 0.75)
    source_itype = param_dic.get("source_itype", 1)
    silent = param_dic.get("silent", True)
    report = param_dic.get("report", True)
    # default behavior:
    # source active in first stress period, zero afterwards
    source_active_periods = param_dic.get(
        "source_active_periods",
        [c0] + [0.0] * (nper - 1)
    )

    # ---------- Basic validation ----------
    if len(perlen_list) != nper:
        raise ValueError(f"Length of perlen_list ({len(perlen_list)}) must equal nper ({nper}).")

    if len(source_active_periods) != nper:
        raise ValueError(
            f"Length of source_active_periods ({len(source_active_periods)}) must equal nper ({nper})."
        )

    # ===== CREATE MT3DMS MODEL =====
    mt = flopy.mt3d.Mt3dms(
        modelname=modelname_mt,
        model_ws=model_ws,
        exe_name=exe_name_mt,
        modflowmodel=mf
    )

    print(f"Created MT3DMS model: {modelname_mt}")
    print("Transport model is linked to flow model")
    timprs = np.linspace(0, 46000, 52)
    # ===== BASIC TRANSPORT PACKAGE =====
    btn = create_btn_package(
        mt, icbund, prsity, sconc, ncomp, obs_row, obs_col,
        perlen_list, nper, timprs
    )

    # ===== ADVECTION PACKAGE =====
    adv = create_adv_package(mt, mixelm, percel)

    # ===== DISPERSION PACKAGE =====
    dsp = create_dsp_package(mt, al, trpt, trpv)

    # ===== SOURCE/SINK MIXING PACKAGE =====
    ssm = create_ssm_package(
        mt, nper, param_dic["nrow"], source_active_periods, source_itype,
        source_row, source_col, c0
    )

    # ===== SOLVER PACKAGE =====
    gcg = create_gcg_package(mt)

    # ===== REACTION / SORPTION PACKAGE =====
    rct = create_rct_package(
        mt, isothm, ireact, igetsc, rhob, kd, alpha, lambda1
    )
    
    mt.write_input()
    print("MT3DMS input files written")
    
    # ===== RUN MT3DMS =====
    print("Running MT3DMS...")
    success_mt, buff_mt = mt.run_model(silent=silent, report=report)
    return mt, success_mt, buff_mt