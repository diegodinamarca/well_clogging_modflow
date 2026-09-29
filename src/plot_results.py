from plot_conc_utils import plot_first_concentration, plot_last_concentration
from plot_head_utils import plot_first_head, plot_last_head

def plot_first_and_last_concentration(folder):
    plot_first_concentration(folder)
    plot_last_concentration(folder)

def plot_first_and_last_head(folder):
    plot_first_head(folder)
    plot_last_head(folder)

plot_first_and_last_concentration("/home/diego/modflow/master_thesis/test1/model/timestep_0")
plot_first_and_last_concentration("/home/diego/modflow/master_thesis/test1/model/timestep_1")
plot_first_and_last_head("/home/diego/modflow/master_thesis/test1/model/timestep_0")
plot_first_and_last_head("/home/diego/modflow/master_thesis/test1/model/timestep_1")
