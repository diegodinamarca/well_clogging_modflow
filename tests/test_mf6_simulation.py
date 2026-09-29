import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flow_model.create_mf6_simulation import create_mf6_simulation
from flow_model.load_flow_config import load_flow_config


class Modflow6SimulationTests(unittest.TestCase):
    def setUp(self):
        self.config_path = PROJECT_ROOT / "test1" / "config.toml"
        self.parameters = load_flow_config(self.config_path)

    def test_config_uses_config_directory_for_model_workspace(self):
        self.assertEqual(
            Path(self.parameters["model_ws"]),
            self.config_path.parent / "model",
        )

    def test_writes_coupled_gwf_gwt_inputs(self):
        self.parameters["source_active_periods"] = [1.25, 0.0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, _, _ = create_mf6_simulation(
                self.parameters,
                model_ws=temporary_directory,
            )
            workspace = Path(temporary_directory)
            expected_files = {
                "mfsim.nam",
                "flow_model.nam",
                "transport_model.nam",
                "flow_model.chd",
                "flow_model.wel",
                "flow_model.rcha",
                "transport_model.fmi",
                "transport_model.cnc",
                "transport_model.mst",
                "modflow6_simulation.gwfgwt",
                "gwf.ims",
                "gwt.ims",
            }
            self.assertTrue(expected_files.issubset({path.name for path in workspace.iterdir()}))

            well_input = (workspace / "flow_model.wel").read_text()
            self.assertIn("1 10 10 5.00000000E+00", well_input)
            self.assertIn("2 10 10 5.00000000E+00", well_input)
            self.assertIn("CONSTANT       0.00250000", (workspace / "flow_model.rcha").read_text())
            cnc_input = (workspace / "transport_model.cnc").read_text()
            self.assertIn("1 10 10 1.25000000E+00", cnc_input)
            self.assertIn("GWFHEAD  FILEIN  flow_model.hds", (workspace / "transport_model.fmi").read_text())
            self.assertIn("distcoef", (workspace / "transport_model.mst").read_text())
            ssm_input = (workspace / "transport_model.ssm").read_text()
            self.assertIn("CHD  AUX  CONCENTRATION", ssm_input)
            self.assertIn("WEL  AUX  CONCENTRATION", ssm_input)
            self.assertIn("RCHA  AUX  CONCENTRATION", ssm_input)
            simulation_namefile = (workspace / "mfsim.nam").read_text()
            self.assertIn("gwf.ims", simulation_namefile)
            self.assertIn("gwt.ims", simulation_namefile)

    def test_rejects_well_layer_outside_grid(self):
        self.parameters["well_layer"] = [[1, self.parameters["nlay"] + 1]]
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "well layer index"):
                create_mf6_simulation(self.parameters, model_ws=temporary_directory)

    def test_rejects_mismatched_flow_and_transport_time_steps(self):
        self.parameters["nstp_mt"] = [1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "matching flow and transport time steps"):
                create_mf6_simulation(self.parameters, model_ws=temporary_directory)

    def test_rejects_multiple_transport_species(self):
        self.parameters["ncomp"] = 2
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "supports one GWT species"):
                create_mf6_simulation(self.parameters, model_ws=temporary_directory)


if __name__ == "__main__":
    unittest.main()