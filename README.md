# Well clogging MODFLOW model

This project runs coupled groundwater-flow and solute-transport simulations to
study how particle deposition affects hydraulic conductivity.

## Model workflow

The workflow runs the configured number of iterations from `[runtime]` in
`config.toml`. Each iteration runs a coupled MODFLOW 6 flow and transport
simulation in its own directory: `simulations/sim_0`, `simulations/sim_1`, and
so on. The model uses each run's concentration and flow results to calculate
deposition, then carries the updated model state into the next iteration.

Each iteration is a separate simulation with the time discretization configured
in `config.toml`; iterations are not individual model time steps. The
`[time]` section controls the stress periods and time steps within each
simulation.

## Hydraulic-conductivity updates

Set `methods.k_update_method` in `config.toml` to choose how conductivity is
updated between iterations. Supported values are:

- `"lm"`: for cells where deposited mass exceeds
  `transport.deposition_threshold`, multiply conductivity by
  `methods_parameters.k_linear_slope`. For example, `0.99` reduces conductivity
  by 1% in affected cells per iteration.
- `"exp_decay"`: apply exponential decay over the iteration count, using
  `methods_parameters.k_exp`. only works if the time-step between iterations is equal to one.
- `"power_law"`: calculate conductivity from the porosity ratio raised to
  `methods_parameters.k_power`. Requires updating porosity
- `"kozeny_carman"`: use the Kozeny-Carman porosity relationship. Requires updating porosity
- `"ravshanov"`: combine a porosity power law with exponential attenuation
  based on deposited mass, using `k_power` and `k_exp`.
- `"none"`: leave conductivity unchanged.

The method names are case-sensitive.
`k_linear_slope`, `k_exp`, and `k_power` are set in the
`[methods_parameters]` section. The deposition threshold is configured as
`transport.deposition_threshold`.

Currently, the workflow calculates a porosity candidate but then uses the
existing porosity for the conductivity update. As a result, the
`"power_law"` and `"kozeny_carman"` methods do not currently incorporate
deposition-driven porosity changes; `"ravshanov"` still depends on deposited
mass as well as porosity.

## Time discretization

Edit the `[time]` section of `config.toml` to change the time discretization
used by the iterations:

- `nper_mf` and `nper_mt` set the number of flow and transport stress periods.
- `perlen_mf` and `perlen_mt` set the length of each period.
- `nstp_mf` and `nstp_mt` set the number of time steps in each period.
- `steady_mf` sets whether each flow stress period is steady state.

Flow and transport must use matching stress-period counts, period lengths, and
time-step counts. The period-length, time-step, and steady-state lists must
each contain one entry per stress period. Source concentrations in
`transport.initial_source_concentration` must also contain one entry per
stress period.

Weird behavior is encountered when running each iteration using steady state and more than one time-step.
Recommended usage involves setting nper = 1, perlen = 1, nstp = 1 and steady_mf = false

## Usage

Run commands from a project/workspace folder containing `config.toml`. The
workflow writes simulation results to that folder's `simulations/` directory. 

### Run the model

```bash
python <script_folder>/run_clogging_workflow.py
```

For example, when running from a workspace folder one level below the project
root:

```bash
python ../src/run_clogging_workflow.py
```

### Plot simulation results

```bash
python src/plot_results.py <n> --ws <folder>
```

- `<n>`: maximum number of `sim_i` iterations to plot.
- `--ws`: workspace folder containing `config.toml` and `simulations/`.

For example:

```bash
python src/plot_results.py 3 --ws test1
```

### Run a sensitivity analysis

Run sensitivity analysis from a folder containing the config.toml or specify the folder in the `--folder` parameter.

With a range of values:

```bash
python src/sensitivity_workflow.py <parameter_name> <min_value> <max_value> <n_intervals> [--folder <output_folder>] [--overwrite]
```

- `<parameter_name>`: parameter name from `config.toml`.
- `<min_value>` and `<max_value>`: bounds of the parameter range.
- `<n_intervals>`: number of intervals in the range.
- `--folder`: optional output folder.
- `--overwrite`: optional flag to overwrite existing output.

For example:

```bash
python src/sensitivity_workflow.py k_linear_slope 0.1 1 10
```

With a list of values:

```bash
python src/sensitivity_workflow.py <parameter_name> --value_list <value1,value2,...> [--folder <output_folder>] [--overwrite]
```

For example:

```bash
python src/sensitivity_workflow.py k_linear_slope --value_list 0.9,0.95,0.99
```
