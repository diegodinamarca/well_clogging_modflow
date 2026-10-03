"""Plot concentration and head results for a range of model timesteps.

Usage from the project workspace::

    python src/plot_results.py 3 --ws test1

The positional argument is the number of timestep folders to process. The
optional ``--ws`` argument points to the folder containing ``timestep_0``,
``timestep_1``, and so on; it defaults to the current working directory.
"""

from plot_conc_utils import plot_first_concentration, plot_last_concentration
from plot_flow_properties import plot_flow_properties
from plot_head_utils import plot_first_head, plot_last_head
import argparse

def plot_first_and_last_concentration(folder):
    plot_first_concentration(folder)
    plot_last_concentration(folder)

def plot_first_and_last_head(folder):
    plot_first_head(folder)
    plot_last_head(folder)

def main():
    # Parse the number of timestep folders and their model workspace.
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="number of timesteps to plot")
    parser.add_argument(
        "--ws",
        dest = "model_ws",
        default=".",
        help="model workspace folder (defaults to the current working directory)",
    )
    args = parser.parse_args()
    model_ws = args.model_ws
    for simulation_i in range(args.n):
        # Each timestep folder contains the results to plot for that step.
        folder = (
            f"{model_ws}/simulations/sim_{simulation_i}"
        )
        print(folder)
        plot_first_and_last_concentration(folder)
        plot_first_and_last_head(folder)
        plot_flow_properties(folder, layer=0)

if __name__ == "__main__":
    main()