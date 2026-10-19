"""Atomic artifacts and integrity checks; no executable checkpoint objects."""
import hashlib
import json
import os
from pathlib import Path
try:
    import torch
except ImportError:  # Pure proof generation and the default NumPy workflow stay usable.
    torch = None


def state_digest(value):
    """Canonical content hash for numeric state trees, independent of torch ZIPs."""
    h = hashlib.sha256()
    def visit(item):
        if torch is not None and isinstance(item, torch.Tensor):
            t = item.detach().cpu().contiguous()
            h.update(b'tensor:')
            h.update(str(t.dtype).encode())
            h.update(json.dumps(list(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b'dict:')
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode() + b':')
            for x in item: visit(x)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            h.update(type(item).__name__.encode() + b':')
            h.update(json.dumps(item, allow_nan=False).encode())
        else:
            raise TypeError(f'unsupported state type: {type(item).__name__}')
        h.update(b';')
    visit(value)
    return h.hexdigest()


def digest(data):
    if not isinstance(data, bytes):
        data = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    if torch is None: raise RuntimeError('numeric neural checkpoints require the optional swap dependency')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    checksum = path.with_suffix('.sha256.json')
    journal = path.with_suffix('.pending.json')
    previous = read(checksum)['sha256'] if checksum.exists() else None
    new_hash = digest(temp.read_bytes())
    write(journal, {'new_sha256': new_hash, 'previous_sha256': previous})
    temp.replace(path)
    write(checksum, {'sha256': new_hash})
    journal.unlink(missing_ok=True)


def load(path, device='cpu'):
    if torch is None: raise RuntimeError('numeric neural checkpoints require the optional swap dependency')
    path = Path(path)
    checksum, journal = path.with_suffix('.sha256.json'), path.with_suffix('.pending.json')
    actual = digest(path.read_bytes())
    recorded = read(checksum)['sha256'] if checksum.exists() else None
    if journal.exists():
        pending = read(journal)
        # Complete only a fully written, journal-bound promotion. An arbitrary
        # modification still fails the ordinary checksum contract.
        if actual == pending['new_sha256'] and recorded in (pending['previous_sha256'], pending['new_sha256']):
            write(checksum, {'sha256': actual})
            recorded = actual
        if actual == recorded: journal.unlink()
    if actual != recorded:
        raise ValueError('checkpoint integrity failure')
    return torch.load(path, map_location=device, weights_only=True)
