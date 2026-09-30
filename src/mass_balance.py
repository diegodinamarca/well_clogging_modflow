from flopy.utils import CellBudgetFile

file = "/home/diego/modflow/master_thesis/test1/model/timestep_0/flow_model.cbc"
cbc = CellBudgetFile(file)

print(cbc.get_unique_record_names())

wel = cbc.get_data(
    kstpkper=(0, 0),
    text="WEL"
)[0]
print(wel)
print(wel.dtype)
print(f"Well flows: {wel['q']}")

total_wel_flow = wel["q"].sum()
print(f"Total WEL flow: {total_wel_flow}")