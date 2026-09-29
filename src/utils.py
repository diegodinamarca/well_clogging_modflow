

from pathlib import Path
import flopy


# Read concentration data from a MODFLOW 6 simulation output file (.ucn) and return the concentration values as a numpy array.
def read_concentration_file(model_ws, modelname_mt="transport_model"):
    ucnfile = Path(model_ws) / f"{modelname_mt}.ucn"
    if filenotfound := not ucnfile.exists():
        raise FileNotFoundError(f"No .ucn file found at {ucnfile}")
    print(f"Reading concentration data from {ucnfile}")
    ucnobj = flopy.utils.HeadFile(ucnfile, text="CONCENTRATION")
    conc = ucnobj.get_alldata()
    return conc

def get_concentration_at_time(model_ws, timestp, modelname_mt="transport_model"):
    ucnfile = Path(model_ws) / f"{modelname_mt}.ucn"
    if filenotfound := not ucnfile.exists():
        raise FileNotFoundError(f"No .ucn file found at {ucnfile}")
    ucnobj = flopy.utils.HeadFile(ucnfile, text="CONCENTRATION")
    times_mt = ucnobj.get_times()
    conc = ucnobj.get_data(totim=times_mt[timestp])
    return conc

def get_concentration_times(model_ws, modelname_mt="transport_model"):
    ucnfile = Path(model_ws) / f"{modelname_mt}.ucn"
    if filenotfound := not ucnfile.exists():
        raise FileNotFoundError(f"No .ucn file found at {ucnfile}")
    ucnobj = flopy.utils.HeadFile(ucnfile, text="CONCENTRATION")
    # Get available times
    times_mt = ucnobj.get_times()
    return(times_mt)

def get_last_concentration(model_ws, modelname_mt="transport_model"):
    ucnfile = Path(model_ws) / f"{modelname_mt}.ucn"
    if filenotfound := not ucnfile.exists():
        raise FileNotFoundError(f"No .ucn file found at {ucnfile}")
    ucnobj = flopy.utils.HeadFile(ucnfile, text="CONCENTRATION")
    # Get available times
    times_mt = ucnobj.get_times()
    last_time = times_mt[-1]
    conc = ucnobj.get_data(totim=last_time)
    return conc


# Read head data from a MODFLOW 6 simulation output file (.hds) and return the head values as a numpy array.
def read_head_file(model_ws, modelname_mt="flow_model"):
    ucnfile = Path(model_ws) / f"{modelname_mt}.hds"
    if filenotfound := not ucnfile.exists():
        raise FileNotFoundError(f"No .hds file found at {ucnfile}")
    print(f"Reading head data from {ucnfile}")
    ucnobj = flopy.utils.HeadFile(ucnfile, text="HEAD")
    conc = ucnobj.get_alldata()
    return conc

def get_head_at_time(model_ws, timestp, modelname_mt="flow_model"):
    ucnfile = Path(model_ws) / f"{modelname_mt}.hds"
    if filenotfound := not ucnfile.exists():
        raise FileNotFoundError(f"No .hds file found at {ucnfile}")
    ucnobj = flopy.utils.HeadFile(ucnfile, text="HEAD")
    times_mt = ucnobj.get_times()
    head = ucnobj.get_data(totim=times_mt[timestp])
    return head

def get_head_times(model_ws, modelname_mt="flow_model"):
    ucnfile = Path(model_ws) / f"{modelname_mt}.hds"
    if filenotfound := not ucnfile.exists():
        raise FileNotFoundError(f"No .hds file found at {ucnfile}")
    ucnobj = flopy.utils.HeadFile(ucnfile, text="HEAD")
    # Get available times
    times_mt = ucnobj.get_times()
    return(times_mt)

def get_last_head(model_ws, modelname_mt="flow_model"):
    file = Path(model_ws) / f"{modelname_mt}.hds"
    if filenotfound := not file.exists():
        raise FileNotFoundError(f"No .hds file found at {file}")
    ucnobj = flopy.utils.HeadFile(file, text="HEAD")
    # Get available times
    times_mt = ucnobj.get_times()
    last_time = times_mt[-1]
    head = ucnobj.get_data(totim=last_time)
    return head