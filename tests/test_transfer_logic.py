import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("generate_transfer_logic", ROOT / "experiments/generate_transfer_logic.py")
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class TransferLogicTests(unittest.TestCase):
    def test_topology_holdouts_and_cycle_do_not_break_certificates(self):
        lineages = {}
        for split in ("train", "validation", "test"):
            rows = [transfer.make_record(i, split) for i in range(40)]
            lineages[split] = {r["lineage"] for r in rows}
            for row in rows:
                transfer.check_example(row)
                self.assertEqual(row["verification"]["lean"], "not_run")
                self.assertGreater(len(row["long_path"]), len(row["short_path"]))
        self.assertFalse(lineages["train"] & lineages["validation"] | lineages["train"] & lineages["test"] | lineages["validation"] & lineages["test"])

    def test_reproducible_and_both_test_grammars_present(self):
        self.assertEqual(transfer.make_record(0, "test"), transfer.make_record(0, "test"))
        self.assertNotEqual(transfer.make_record(0, "test")["family"], transfer.make_record(1, "test")["family"])


if __name__ == "__main__":
    unittest.main()
