"""Backend dispatch retains the default affine experiment and optional imports."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from lemmaferry.rosetta import PairedSplit, RosettaPair
from lemmaferry.translators import fit_translator, load_translator


class TranslatorDispatchTests(unittest.TestCase):
    def test_default_and_explicit_affine_match_and_all_controls_dispatch(self):
        x = np.arange(24, dtype=float).reshape(8, 3)
        y = x[:, :2] * [2, -3] + [7, 8]
        train = PairedSplit(x, y, [f"t{i}" for i in range(8)], ["train"] * 8, "train")
        val = PairedSplit(x[:3] + .5, (x[:3, :2] + .5) * [2, -3] + [7, 8],
                          [f"v{i}" for i in range(3)], ["val"] * 3, "val")
        original = RosettaPair.fit(train, val)
        dispatched = fit_translator(train, val)
        np.testing.assert_array_equal(original.to_language(x), dispatched.to_language(x))
        np.testing.assert_array_equal(dispatched.to_language(x, control="shuffled"), dispatched.maps["T_shuffled"].predict(x))
        np.testing.assert_array_equal(dispatched.to_math(y, control="shuffled"), dispatched.maps["S_shuffled"].predict(y))
        np.testing.assert_allclose(dispatched.to_language(x, control="mean"), np.broadcast_to(y.mean(0), y.shape))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            dispatched.save(path)
            self.assertIsInstance(load_translator(path), RosettaPair)
        with self.assertRaises(ValueError):
            fit_translator(train, val, translator="unknown")
        with self.assertRaises(ValueError):
            dispatched.to_language(x, control="wrong")

    def test_import_and_affine_dispatch_do_not_load_torch(self):
        script = "from lemmaferry.translators import fit_translator,load_translator; import sys; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    def test_unknown_or_malformed_bundle_is_rejected_without_executing_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for content in ([], {"schema": "python.pickle"}, {"schema": "lemmaferry.swap.bundle.v999"}):
                (path / "manifest.json").write_text(json.dumps(content))
                with self.assertRaises(ValueError):
                    load_translator(path)


if __name__ == "__main__":
    unittest.main()
