"""Crash-window recovery preserves hash-bound old/new numeric checkpoints."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from lemmaferry.scratch import io


class ScratchOptionalImportTests(unittest.TestCase):
    def test_logic_and_pure_state_hashing_work_when_torch_is_unavailable(self):
        script = """
import sys
sys.modules['torch'] = None
from lemmaferry.scratch import io,logic
assert io.torch is None
assert len(io.digest({'ordinary': True})) == 64
assert len(io.state_digest({'ordinary': [1,2]})) == 64
assert logic.infer(['hyp',0],[0]) == 0
try:
    io.save('unused.pt',{})
except RuntimeError:
    pass
else:
    raise AssertionError('Torch-free code unexpectedly saved a neural checkpoint')
"""
        subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)


@unittest.skipUnless(HAS_TORCH, 'optional Torch runtime required')
class ScratchAtomicCheckpointTests(unittest.TestCase):
    def test_journal_finishes_new_data_promotion_only_when_both_hashes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp)/'state.pt', Path(tmp)/'new.pt'
            io.save(old, {'value': torch.tensor([1.])})
            previous = io.read(old.with_suffix('.sha256.json'))['sha256']
            io.save(new, {'value': torch.tensor([2.])})
            new_hash = io.digest(new.read_bytes())
            old.write_bytes(new.read_bytes())
            io.write(old.with_suffix('.pending.json'), {'new_sha256':new_hash,'previous_sha256':previous})
            restored = io.load(old)
            self.assertEqual(restored['value'].tolist(), [2.])
            self.assertEqual(io.read(old.with_suffix('.sha256.json'))['sha256'], new_hash)
            self.assertFalse(old.with_suffix('.pending.json').exists())

    def test_pre_promotion_crash_retains_the_valid_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.pt'
            io.save(path, {'value': torch.tensor([1.])})
            previous = io.read(path.with_suffix('.sha256.json'))['sha256']
            original_bytes = path.read_bytes()
            io.write(path.with_suffix('.pending.json'), {'new_sha256':'0'*64,'previous_sha256':previous})
            restored = io.load(path)
            self.assertEqual(restored['value'].tolist(), [1.])
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(io.read(path.with_suffix('.sha256.json'))['sha256'], previous)
            self.assertFalse(path.with_suffix('.pending.json').exists())

    def test_arbitrary_modified_bytes_do_not_gain_validity_from_a_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.pt'
            io.save(path, {'value': torch.tensor([1.])})
            previous = io.read(path.with_suffix('.sha256.json'))['sha256']
            path.write_bytes(path.read_bytes()+b'corrupted')
            io.write(path.with_suffix('.pending.json'), {'new_sha256':'0'*64,'previous_sha256':previous})
            with self.assertRaisesRegex(ValueError, 'integrity'):
                io.load(path)
            self.assertTrue(path.with_suffix('.pending.json').exists())


if __name__ == '__main__':
    unittest.main()
