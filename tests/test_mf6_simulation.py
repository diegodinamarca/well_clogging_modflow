import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flow_model.create_mf6_simulation import create_mf6_simulation
from flow_model.load_flow_config import _create_refined_grid, load_flow_config


class Modflow6SimulationTests(unittest.TestCase):
    def setUp(self):
        self.config_path = PROJECT_ROOT / "test1" / "config.toml"
        self.parameters = load_flow_config(self.config_path)

    def test_config_uses_config_directory_for_model_workspace(self):
        self.assertEqual(
            Path(self.parameters["model_ws"]),
            self.config_path.parent / "model",
        )

    def test_well_refinement_preserves_extent_and_maps_locations(self):
        self.assertEqual(self.parameters["nrow"], 23)
        self.assertEqual(self.parameters["ncol"], 23)
        self.assertAlmostEqual(float(np.sum(self.parameters["delr"])), 200.0)
        self.assertAlmostEqual(float(np.sum(self.parameters["delc"])), 200.0)
        self.assertEqual(self.parameters["well_location"], [[11, 11]])
        self.assertEqual(self.parameters["obs_row"], 11)
        self.assertEqual(self.parameters["obs_col"], 11)
        self.assertEqual(self.parameters["source_row"], 11)
        self.assertEqual(self.parameters["source_col"], 11)
        self.assertEqual(self.parameters["hk"].shape, (2, 23, 23))
        self.assertEqual(self.parameters["sconc"].shape, (2, 23, 23))

    def test_root_config_loads_with_well_refinement(self):
        parameters = load_flow_config(PROJECT_ROOT / "config.toml")

        self.assertEqual(parameters["nrow"], 23)
        self.assertEqual(parameters["ncol"], 23)
        self.assertEqual(parameters["source_active_periods"], [0.0, 0.0])

    def test_well_refinement_is_shared_by_flow_and_transport(self):
        self.parameters["source_active_periods"] = [1.0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, gwf, gwt = create_mf6_simulation(
                self.parameters,
                model_ws=temporary_directory,
            )
            well_input = (Path(temporary_directory) / "flow_model.wel").read_text()

        self.assertEqual(gwf.dis.nrow.get_data(), gwt.dis.nrow.get_data())
        self.assertEqual(gwf.dis.ncol.get_data(), gwt.dis.ncol.get_data())
        np.testing.assert_array_equal(gwf.dis.delr.array, gwt.dis.delr.array)
        np.testing.assert_array_equal(gwf.dis.delc.array, gwt.dis.delc.array)
        self.assertEqual(well_input.count("11 11 -2.00000000E+01"), 2)

    def test_refinement_factor_one_keeps_the_original_grid(self):
        result = _create_refined_grid(20, 20, 10, 10, [[10, 10]], 1, 10)
        delr, delc, row_parent, column_parent, row_map, column_map = result

        np.testing.assert_array_equal(np.asarray(delr), np.full(20, 10))
        np.testing.assert_array_equal(np.asarray(delc), np.full(20, 10))
        np.testing.assert_array_equal(row_parent, np.arange(20))
        np.testing.assert_array_equal(column_parent, np.arange(20))
        self.assertEqual(row_map[9] + 1, 10)
        self.assertEqual(column_map[9] + 1, 10)

    def test_refinement_rejects_invalid_factor_and_radius(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            _create_refined_grid(20, 20, 10, 10, [[10, 10]], 0, 10)
        with self.assertRaisesRegex(ValueError, "positive number"):
            _create_refined_grid(20, 20, 10, 10, [[10, 10]], 2, 0)

    def test_refined_array_values_follow_their_parent_cells(self):
        _, _, row_parent, column_parent, _, _ = _create_refined_grid(
            20, 20, 10, 10, [[10, 10]], 2, 10
        )
        original = np.arange(2 * 20 * 20).reshape((2, 20, 20))
        refined = np.take(np.take(original, row_parent, axis=1), column_parent, axis=2)

        for row, parent_row in enumerate(row_parent):
            for column, parent_column in enumerate(column_parent):
                np.testing.assert_array_equal(
                    refined[:, row, column], original[:, parent_row, parent_column]
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
            self.assertIn("1 11 11 5.00000000E+00", well_input)
            self.assertIn("2 11 11 5.00000000E+00", well_input)
            self.assertIn("CONSTANT       0.00250000", (workspace / "flow_model.rcha").read_text())
            cnc_input = (workspace / "transport_model.cnc").read_text()
            self.assertIn("1 11 11 1.25000000E+00", cnc_input)
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