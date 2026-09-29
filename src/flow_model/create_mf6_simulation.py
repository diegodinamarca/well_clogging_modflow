from pathlib import Path

import flopy
import numpy as np


def _cell_index(value, size, name):
    index = int(value) - 1
    if index < 0 or index >= size:
        raise ValueError(f"Configured {name} index {value} is outside 1..{size}.")
    return index


def create_mf6_simulation(param_dic, model_ws=None):
    """Build and write the coupled MODFLOW 6 flow and transport simulation."""
    workspace = Path(model_ws or param_dic["model_ws"]).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    nlay, nrow, ncol = param_dic["nlay"], param_dic["nrow"], param_dic["ncol"]
    if param_dic.get("ncomp", 1) != 1:
        raise ValueError("This MODFLOW 6 setup supports one GWT species per simulation.")
    nper = param_dic["nper_mf"]
    period_lengths = param_dic["perlen_mf"]
    steps_per_period = param_dic["nstp_mf"]
    steady_periods = param_dic["steady_mf"]
    if not (len(period_lengths) == len(steps_per_period) == len(steady_periods) == nper):
        raise ValueError("Flow stress-period values must each have nper entries.")
    if param_dic["nper_mt"] != nper or param_dic["perlen_mt"] != period_lengths:
        raise ValueError("MODFLOW 6 requires matching flow and transport stress periods.")
    if param_dic["nstp_mt"] != steps_per_period:
        raise ValueError("MODFLOW 6 requires matching flow and transport time steps.")
    source_active_periods = param_dic["source_active_periods"]
    if len(source_active_periods) != nper:
        raise ValueError("Source concentrations must have one value per stress period.")

    sim_name = "modflow6_simulation"
    sim = flopy.mf6.MFSimulation(
        sim_name=sim_name,
        exe_name=param_dic["exe_name_mf"],
        sim_ws=str(workspace),
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=nper,
        perioddata=[
            (period_lengths[period], steps_per_period[period], 1.0)
            for period in range(nper)
        ],
    )

    modelname_gwf = param_dic["modelname_mf"]
    modelname_gwt = param_dic["modelname_mt"]
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=modelname_gwf,
        save_flows=True,
    )
    gwt = flopy.mf6.ModflowGwt(
        sim,
        modelname=modelname_gwt,
        save_flows=True,
    )

    idomain = (np.asarray(param_dic["ibound"]) != 0).astype(int)
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=param_dic["delr"],
        delc=param_dic["delc"],
        top=param_dic["ztop"],
        botm=param_dic["zbot"],
        idomain=idomain,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=param_dic["strt"])
    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=param_dic["laytyp"],
        k=param_dic["hk"],
        k22=param_dic["hk"],
        k33=param_dic["vka"],
        save_specific_discharge=True,
    )

    steady_state = {period: True for period, is_steady in enumerate(steady_periods) if is_steady}
    transient = {period: True for period, is_steady in enumerate(steady_periods) if not is_steady}
    specific_yield = param_dic.get("specific_yield")
    if transient and specific_yield is None and np.any(np.asarray(param_dic["laytyp"]) != 0):
        raise ValueError("Transient convertible layers require a configured specific_yield.")
    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=param_dic["laytyp"],
        ss=param_dic["ss"],
        sy=0.0 if specific_yield is None else specific_yield,
        steady_state=steady_state or None,
        transient=transient or None,
    )

    fixed_head_cells = np.argwhere(np.asarray(param_dic["ibound"]) < 0)
    chd_data = [
        (
            (int(layer), int(row), int(column)),
            float(param_dic["strt"][layer, row, column]),
            0.0,
        )
        for layer, row, column in fixed_head_cells
    ]
    if chd_data:
        flopy.mf6.ModflowGwfchd(
            gwf,
            pname="CHD",
            auxiliary=["CONCENTRATION"],
            stress_period_data={period: chd_data for period in range(nper)},
        )

    well_data = {}
    well_locations = param_dic["well_location"]
    well_rates = param_dic["well_pumping_rate"]
    well_layers = param_dic["well_layer"]
    if not (len(well_locations) == len(well_rates) == len(well_layers)):
        raise ValueError("Each configured well requires a location, rate, and layer list.")
    for period in range(nper):
        concentration = source_active_periods[period]
        records = []
        for location, rate, layers in zip(well_locations, well_rates, well_layers):
            row = _cell_index(location[0], nrow, "well row")
            column = _cell_index(location[1], ncol, "well column")
            for layer_value in layers:
                layer = _cell_index(layer_value, nlay, "well layer")
                records.append(((layer, row, column), float(rate), float(concentration)))
        well_data[period] = records
    active_wells = any(well_data.values())
    if active_wells:
        flopy.mf6.ModflowGwfwel(
            gwf,
            pname="WEL",
            auxiliary=["CONCENTRATION"],
            stress_period_data=well_data,
            save_flows=True,
        )

    recharge = float(param_dic["recharge"])
    active_recharge = recharge != 0.0
    if active_recharge:
        flopy.mf6.ModflowGwfrcha(
            gwf,
            pname="RCHA",
            auxiliary=["CONCENTRATION"],
            recharge=recharge,
            aux=0.0,
            save_flows=True,
        )

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{modelname_gwf}.hds",
        budget_filerecord=f"{modelname_gwf}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    flopy.mf6.ModflowGwtdis(
        gwt,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=param_dic["delr"],
        delc=param_dic["delc"],
        top=param_dic["ztop"],
        botm=param_dic["zbot"],
        idomain=idomain,
    )
    flopy.mf6.ModflowGwtic(gwt, strt=param_dic["sconc"])
    flopy.mf6.ModflowGwtadv(gwt, scheme=param_dic.get("advection_scheme", "TVD"))
    al = float(param_dic["al"])
    flopy.mf6.ModflowGwtdsp(
        gwt,
        alh=al,
        ath1=al * float(param_dic["trpt"]),
        ath2=al * float(param_dic["trpt"]),
        atv=al * float(param_dic["trpv"]),
    )

    mst_options = {
        "porosity": param_dic["prsity"],
        "sorption": "LINEAR",
        "bulk_density": param_dic["rhob"],
        "distcoef": param_dic["kd"],
    }
    decay = float(param_dic["lambda1"])
    if decay > 0.0:
        mst_options.update(first_order_decay=True, decay=decay)
    flopy.mf6.ModflowGwtmst(gwt, **mst_options)

    source_packages = []
    if chd_data:
        source_packages.append(("CHD", "AUX", "CONCENTRATION"))
    if active_wells:
        source_packages.append(("WEL", "AUX", "CONCENTRATION"))
    if active_recharge:
        source_packages.append(("RCHA", "AUX", "CONCENTRATION"))
    if source_packages:
        flopy.mf6.ModflowGwtssm(
            gwt,
            sources=source_packages,
        )

    source_row = _cell_index(param_dic["source_row"], nrow, "source row")
    source_column = _cell_index(param_dic["source_col"], ncol, "source column")
    source_layer = _cell_index(param_dic.get("source_layer", 1), nlay, "source layer")
    source_cell = (source_layer, source_row, source_column)
    cnc_data = {
        period: [(source_cell, float(concentration))]
        for period, concentration in enumerate(source_active_periods)
    }
    flopy.mf6.ModflowGwtcnc(gwt, stress_period_data=cnc_data)
    flopy.mf6.ModflowGwtfmi(
        gwt,
        packagedata=[
            ("GWFHEAD", f"{modelname_gwf}.hds"),
            ("GWFBUDGET", f"{modelname_gwf}.cbc"),
        ],
    )
    flopy.mf6.ModflowGwtoc(
        gwt,
        concentration_filerecord=f"{modelname_gwt}.ucn",
        budget_filerecord=f"{modelname_gwt}.cbc",
        saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")],
    )

    flopy.mf6.ModflowGwfgwt(
        sim,
        exgmnamea=modelname_gwf,
        exgmnameb=modelname_gwt,
    )
    ims_gwf = flopy.mf6.ModflowIms(
        sim,
        filename="gwf.ims",
        pname="ims_gwf",
        print_option="SUMMARY",
        complexity="MODERATE",
        outer_dvclose=1.0e-6,
        outer_maximum=100,
        inner_maximum=300,
        inner_dvclose=1.0e-6,
        rcloserecord="1.0e-6 strict",
        linear_acceleration="BICGSTAB",
    )
    ims_gwt = flopy.mf6.ModflowIms(
        sim,
        filename="gwt.ims",
        pname="ims_gwt",
        print_option="SUMMARY",
        complexity="MODERATE",
        outer_dvclose=1.0e-6,
        outer_maximum=100,
        inner_maximum=300,
        inner_dvclose=1.0e-6,
        rcloserecord="1.0e-6 strict",
        linear_acceleration="BICGSTAB",
    )
    sim.register_ims_package(ims_gwf, [modelname_gwf])
    sim.register_ims_package(ims_gwt, [modelname_gwt])
    sim.write_simulation()
    return sim, gwf, gwt