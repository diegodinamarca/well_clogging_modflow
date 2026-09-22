# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:00:29 2026

@author: rappe
"""

import flopy
from flow_model.create_bas_package import create_bas_package
from flow_model.create_dis_package import create_dis_package
from flow_model.create_lmt_package import create_lmt_package
from flow_model.create_lpf_package import create_lpf_package
from flow_model.create_oc_package import create_oc_package
from flow_model.create_pcg_package import create_pcg_package
from flow_model.create_wel_package import create_wel_package


def create_flow_model(modelname_mf, model_ws, param_dic):
    """
    Create, write, and run a MODFLOW flow model using parameters
    provided in a dictionary loaded from an external config file.

    Parameters
    ----------
    modelname_mf : str
        Name of the MODFLOW model.
    model_ws : str
        Workspace folder where MODFLOW files will be written.
    param_dic : dict
        Dictionary containing all required model parameters.

    Required keys in param_dic
    --------------------------
    exe_name_mf : str
    nlay : int
    nrow : int
    ncol : int
    delr : float or array-like
    delc : float or array-like
    ztop : float or array-like
    zbot : float or array-like
    nper : int
    perlen_list : list
    nstp_list : list
    steady_list : list
    ibound : array-like
    strt : array-like
    laytyp : int or list
    hk : array-like
    vka : array-like
    ss : float or array-like

    Optional keys
    -------------
    ipakcb : int, default 53
    output_file_name : str, default 'mt3d_link.ftl'
    compact_oc : bool, default True
    silent : bool, default True
    report : bool, default True

    Returns
    -------
    mf : flopy.modflow.Modflow
        The created MODFLOW model object.
    success : bool
        Whether the model ran successfully.
    buff : list
        MODFLOW run output.
    """

    # ---------- Required parameters ----------
    exe_name_mf = param_dic["exe_name_mf"]

    nlay = param_dic["nlay"]
    nrow = param_dic["nrow"]
    ncol = param_dic["ncol"]

    delr = param_dic["delr"]
    delc = param_dic["delc"]
    ztop = param_dic["ztop"]
    zbot = param_dic["zbot"]

    nper = param_dic["nper_mf"]
    perlen_list = param_dic["perlen_mf"]
    nstp_list = param_dic["nstp_mf"]
    steady_list = param_dic["steady_mf"]

    ibound = param_dic["ibound"]
    strt = param_dic["strt"]
    #strt = [i+3 for i in param_dic["strt"]]

    laytyp = param_dic["laytyp"]
    hk = param_dic["hk"]
    vka = param_dic["vka"]
    ss = param_dic["ss"]
    rech = param_dic["rech"]

    # ---------- Optional parameters ----------
    ipakcb = param_dic.get("ipakcb", 53)
    output_file_name = param_dic.get("output_file_name", "mt3d_link.ftl")
    compact_oc = param_dic.get("compact_oc", True)
    silent = param_dic.get("silent", True)
    report = param_dic.get("report", True)

    # ---------- Basic validation ----------
    if nper > 1:
        if len(perlen_list) != nper:
            raise ValueError(f"Length of perlen_list ({len(perlen_list)}) must equal nper ({nper}).")

        if len(nstp_list) != nper:
            raise ValueError(f"Length of nstp_list ({len(nstp_list)}) must equal nper ({nper}).")
    
        if len(steady_list) != nper:
            raise ValueError(f"Length of steady_list ({len(steady_list)}) must equal nper ({nper}).")

    # ===== CREATE MODFLOW MODEL =====
    mf = flopy.modflow.Modflow(
        modelname=modelname_mf,
        exe_name=exe_name_mf,
        model_ws=model_ws
    )
    print(f"Created MODFLOW model: {modelname_mf}")
    # ===== DISCRETIZATION PACKAGE =====
    dis = create_dis_package(
        mf, nlay, nrow, ncol, delr, delc, ztop, zbot,
        nper, perlen_list, nstp_list, steady_list
    )
    print(f"ibound shape: {ibound.shape}, strt shape: {strt[0].shape}")
    # ===== BASIC PACKAGE =====
    bas = create_bas_package(mf, ibound, strt)

    # ===== LAYER PROPERTY FLOW PACKAGE =====
    lpf = create_lpf_package(mf, laytyp, hk, vka, ipakcb, ss)
    
    # ===== RECHARGE PACKAGE ====== #
    # rch = flopy.modflow.ModflowRch(mf, rech=rech)
    
    # ===== WELL PACKAGE =====
    wel_spd = param_dic.get("wel_spd", None)
    wel = create_wel_package(mf, wel_spd, ipakcb)

    # ===== OUTPUT CONTROL PACKAGE =====
    oc = create_oc_package(mf, nper, nstp_list, compact_oc)

    # ===== SOLVER PACKAGE =====
    pcg = create_pcg_package(mf)

    # ===== LINK-MT3DMS PACKAGE =====
    lmt = create_lmt_package(mf, output_file_name)

    # ===== WRITE INPUT FILES =====
    mf.write_input()
    print("MODFLOW input files written to:", model_ws)

    # ===== RUN MODFLOW =====
    print("Running MODFLOW...")
    success, buff = mf.run_model(silent=silent, report=report)

    if success:
        print("\n" + "=" * 50)
        print("MODFLOW completed successfully!")
        print("=" * 50)
    else:
        print("\nMODFLOW failed! Check the listing file for errors.")

    return mf, success, buff