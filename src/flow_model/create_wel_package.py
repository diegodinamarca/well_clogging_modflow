import flopy

def setup_well_data(well_locations, well_pumping_rate, well_layer):
    # Build stress period data for stress period 0
    spd = []

    for location, pumping_rate in zip(well_locations, well_pumping_rate):
        row, col = location
        # Assuming well_layers is a list of layers corresponding to each well location  
        for well in well_layer:
            for lay in well:
                spd.append([int(lay), int(row), int(col), pumping_rate])
    print("Well locations [layer, row, col, pumping_rate]: ",spd)
    return spd


def create_wel_package(mf, wel_spd, ipakcb):
    print("Adding well package with specified stress period data.")
    return flopy.modflow.ModflowWel(
        mf,
        stress_period_data=wel_spd,
        ipakcb=ipakcb
    )