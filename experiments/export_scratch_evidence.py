#!/usr/bin/env python3
"""Export compact, verified from-scratch evidence without copying raw weights.

Final order: study -> optional continuation -> independent verification -> report
-> optional plots/diagnostics -> this command. Only stdlib is required; this
reader never imports the training package, loads tensors, or runs a model.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = 'evidence-manifest.json'
SMALL_LIMIT = 1_000_000
REQUIRED = (
    'report.html', 'results.csv', 'summary.json', 'plan.json',
    'selected-config.json', 'pilot-registration.json', 'pilot.json',
    'development.json', 'generation-receipt.json', 'hardware.json',
    'report-path-amendment.json', 'independent-verification.json',
)
OPTIONAL = (
    'collapse-diagnostics.json', 'baseline-independent-verification.json',
    'continuations.json', 'experiment-clock.json', 'predecessor.json',
    'sample.json', 'overall-budget-receipt.json', 'answer-diagnostics.json',
    'validation/cpu-integration.json', 'validation/mps-integration.json',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    """Stream large checkpoints and refuse a file changed during the read."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'expected regular file: {path}')
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), f'file changed while hashing: {path}')
    return digest.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f'nonfinite JSON number: {value}')
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, f'duplicate JSON field: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=reject, object_pairs_hook=pairs)


def file_entry(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def verify_completion(root):
    plan = read_json(root / 'plan.json')
    verification = read_json(root / 'independent-verification.json')
    config = plan['config']
    expected = set(itertools.product(config['seeds'], config['sizes'], plan['arms']))
    require(len(expected) == 36, 'this export requires the registered 36-condition study')
    require(verification.get('status') == 'passed' and not verification.get('incomplete_paths'),
            'independent verification must pass for the complete main study')
    conditions = verification['conditions']
    actual = [(r['seed'], r['size'], r['arm']) for r in conditions]
    require(len(actual) == len(set(actual)) == verification['conditions_checked'] == 36
            and set(actual) == expected, 'verification conditions differ from the registered Cartesian study')
    require(all(len(r['stages']) == config['stages'] for r in conditions),
            'verification omitted a registered recipient stage')
    require(verification['plan_sha256'] == sha(root / 'plan.json'), 'verification refers to another plan')
    require(read_json(root / 'selected-config.json') == config, 'selected config differs from the frozen plan')
    dataset = read_json(root / 'dataset.json')
    fingerprint = hashlib.sha256(canonical(dataset['rows'])).hexdigest()
    require(fingerprint == dataset['fingerprint'] == plan['dataset_sha256']
            == verification['dataset']['fingerprint'], 'dataset differs from the independently verified plan')
    splits = {split: [row['id'] for row in dataset['rows'] if row['split'] == split]
              for split in ('train', 'validation', 'test', 'harder')}
    expected_counts = {'train': max(config['sizes']), 'validation': config['validation'],
                       'test': config['test'], 'harder': config['harder']}
    require(splits == plan['split_ids'] and {key: len(value) for key, value in splits.items()} == expected_counts
            and len({row['id'] for row in dataset['rows']}) == sum(expected_counts.values()),
            'dataset split counts or identities differ from the registered study')
    require(read_json(root / 'generation-receipt.json')['fingerprint'] == fingerprint,
            'generation receipt refers to another dataset')
    corpus = root / 'corpus/alice.txt'
    require(sha(corpus) == plan['ordinary_corpus']['sha256']
            and corpus.stat().st_size == plan['ordinary_corpus']['bytes'],
            'ordinary corpus differs from the pinned source edition')
    require(plan.get('source_sha256'), 'frozen source hashes are missing')
    for relative, expected_hash in plan['source_sha256'].items():
        path = REPOSITORY / relative
        require(relative.startswith('src/lemmaferry/') and relative.endswith('.py')
                and path.resolve().is_relative_to(REPOSITORY), 'unsupported registered source path')
        require(sha(path) == expected_hash, f'frozen source changed: {relative}')
    summary = read_json(root / 'summary.json')
    require(summary.get('status') == 'completed' and summary.get('expected_conditions') == 36
            and summary.get('completed_conditions') == 36 and not summary.get('missing_artifacts'),
            'report is incomplete; regenerate it after verification')
    require(summary.get('independent_verification') == verification,
            'summary contains stale verification; run report after the final verify command')
    require(summary.get('plan_sha256') == sha(root / 'plan.json'), 'summary refers to another plan')
    report = (root / 'report.html').read_text(encoding='utf-8')
    require('COMPLETED' in report and '36/36 recipient conditions evaluated' in report,
            'HTML report is stale or incomplete')
    expected_rows = {(*condition, split) for condition in expected for split in ('test', 'harder')}
    with (root / 'results.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    actual_rows = [(int(r['seed']), int(r['size']), r['arm'], r['split'])
                   for r in rows if r['arm'] in plan['arms']]
    require(len(actual_rows) == len(set(actual_rows)) == 72 and set(actual_rows) == expected_rows,
            'CSV lacks the complete registered condition/split rows')
    summary_rows = {(r['seed'], r['size'], r['arm'], r['split']): r for r in summary['evaluations']}
    for row in rows:
        key = (int(row['seed']), int(row['size']), row['arm'], row['split'])
        require(key in summary_rows, 'CSV and summary row identities differ')
        for metric in ('correct', 'total', 'accuracy', 'generation_correct', 'generation_accuracy', 'answer_loss'):
            require(math.isfinite(float(row[metric])) and float(row[metric]) == summary_rows[key][metric],
                    f'CSV and summary metric differ: {key}, {metric}')
    history_path = root / 'continuations.json'
    if history_path.exists():
        history = read_json(history_path)
        require(all(entry.get('complete') is True for entry in history['entries']),
                'a continuation is unfinished; finish it and rerun final verification')
        require(len(verification.get('continuations', [])) == len(history['entries']),
                'verification predates the completed continuation history')
    return plan, verification, summary


def raw_inventory(root):
    """Hash critical raw evidence; validate sidecars and map envelopes first."""
    paths = []
    for folder, directories, files in os.walk(root, followlinks=False):
        # Atomic bridge construction can retain interrupted, never-promoted
        # bundles for local inspection. They are not accepted study artifacts;
        # critical_paths still requires every canonical completed bundle.
        directories[:] = [name for name in directories if not (Path(folder) / name).is_symlink()
                          and not (name.startswith('.') and ('.staging-' in name or '.interrupted-' in name))]
        for name in files:
            path = Path(folder) / name
            if path.suffix in {'.json', '.pt', '.npz', '.lean'} or (
                    path.suffix == '.log' and 'lean-cache' in path.parts) or path == root / 'tests.log' or (
                    path.name == 'alice.txt' and path.parent.name == 'corpus'):
                paths.append(path)
    entries = {str(path.relative_to(root)): file_entry(path) for path in sorted(paths)}
    for path in paths:
        relative = str(path.relative_to(root))
        if path.suffix == '.pt':
            sidecar = path.with_suffix('.sha256.json')
            pointer = read_json(sidecar)
            require(set(pointer) == {'sha256'} and pointer['sha256'] == entries[relative]['sha256'],
                    f'checkpoint sidecar mismatch: {relative}')
            entries[relative].update(kind='numeric_checkpoint', verified_sidecar=str(sidecar.relative_to(root)))
        elif path.name.endswith('.sha256.json'):
            require(path.with_name(path.name.removesuffix('.sha256.json') + '.pt').is_file(),
                    f'orphan checkpoint sidecar: {relative}')
        elif path.name == 'manifest.json' and path.parent.name == 'maps':
            manifest = read_json(path)
            require(manifest['schema'] in ('lemmaferry.swap.bundle.v1', 'lemmaferry.rosetta.bundle.v1')
                    and manifest['archive'] == 'maps.npz', f'unsupported map manifest: {relative}')
            content = {key: value for key, value in manifest.items() if key != 'manifest_content_sha256'}
            require(hashlib.sha256(canonical(content)).hexdigest() == manifest['manifest_content_sha256'],
                    f'map manifest content hash mismatch: {relative}')
            archive = str((path.parent / 'maps.npz').relative_to(root))
            require(archive in entries and entries[archive]['sha256'] == manifest['archive_sha256'],
                    f'map archive checksum mismatch: {relative}')
            entries[relative].update(kind='translator_manifest', verified_archive=archive)
    return entries


def critical_paths(plan):
    """Do not silently omit required artifacts deleted after verification."""
    paths = {'dataset.json', 'tests.log', 'corpus/alice.txt'}
    def checkpoint(relative):
        paths.add(relative)
        paths.add(str(Path(relative).with_suffix('.sha256.json')))
    def bridge(relative):
        paths.update({relative + '/maps/manifest.json', relative + '/maps/maps.npz', relative + '/cost.json'})
        checkpoint(relative + '/vectors.pt')
    for seed in plan['config']['seeds']:
        checkpoint(f'starting/{seed}.pt')
        paths.update({f'starting/{seed}-pretraining.json', f'starting/{seed}-evaluation.json'})
        for size in plan['config']['sizes']:
            prefix = f'study/{seed}-{size}'
            checkpoint(prefix + '/encoder.pt')
            paths.add(prefix + '/encoder-metrics.json')
            bridge(prefix + '/shared-stage-0')
            for arm in plan['arms']:
                paths.add(prefix + '/' + arm + '/evaluation.json')
                for stage in range(plan['config']['stages']):
                    relative = prefix + '/' + arm + f'/stage-{stage}'
                    checkpoint(relative + '/recipient.pt')
                    paths.update({relative + '/budget.json', relative + '/geometry.json'})
                    if stage: bridge(relative)
    return paths


def test_summary(path):
    text = path.read_text(encoding='utf-8')
    runs = list(re.finditer(r'^Ran (\d+) tests? in ([0-9.]+)s\s*$', text, re.M))
    require(runs, 'tests.log lacks a unittest summary')
    last = runs[-1]
    passed = re.search(r'^OK(?: \(skipped=(\d+)\))?\s*$', text[last.end():], re.M)
    require(passed is not None and not re.search(r'^FAILED', text[last.end():], re.M),
            'last recorded test suite did not pass')
    tests, skipped = int(last.group(1)), int(passed.group(1) or 0)
    return {'schema': 'lemmaferry.scratch.tests-summary.v1', 'status': 'passed', 'tests': tests,
            'executed': tests - skipped, 'skipped': skipped, 'seconds': float(last.group(2)),
            'raw_log': 'tests.log', 'raw_log_sha256': sha(path), 'raw_log_bytes': path.stat().st_size}


def check_hardware(value):
    permitted = {'cpu', 'physical_cpu_cores', 'logical_cpu_cores', 'memory_bytes', 'gpu',
                 'system', 'architecture', 'machine', 'os_release', 'device', 'serial_numbers_or_uuids_collected'}
    require(isinstance(value, dict) and set(value) <= permitted, 'unsupported hardware fields; no private machine metadata is exported')
    require(value.get('serial_numbers_or_uuids_collected') is False, 'hardware receipt must exclude serial numbers and UUIDs')
    def walk(node):
        if isinstance(node, dict):
            for key, child in node.items():
                require(key == 'serial_numbers_or_uuids_collected' or not re.search(r'serial|uuid|udid|owner|username|hostname', key, re.I),
                        'private machine identifier in hardware receipt')
                walk(child)
        elif isinstance(node, list):
            for child in node: walk(child)
    walk(value)


def continuation_summary(root):
    history = read_json(root / 'continuations.json')
    rows = []
    for entry in history['entries']:
        relative = Path(entry['directory']) / 'evaluation.json'
        path = root / relative
        require(path.resolve().is_relative_to(root), 'continuation report path leaves run')
        evaluation = read_json(path)
        rows.append({'continuation': entry,
                     'evaluation': {split: {key: value for key, value in result.items() if key != 'predictions'}
                                    for split, result in evaluation.items()},
                     'raw_evaluation': str(relative), 'raw_evaluation_sha256': sha(path),
                     'per_row_predictions': {split: {'count': len(result['predictions']),
                         'canonical_sha256': hashlib.sha256(canonical(result['predictions'])).hexdigest()}
                         for split, result in evaluation.items()}})
    return {'schema': 'lemmaferry.scratch.continuation-summary.v1', 'continuations': rows,
            'scope': 'Exploratory continuations, separate from the frozen main comparison. Every aggregate metric is retained; raw question/prediction rows remain local and hash-bound.'}


def compact_collapse_diagnostics(value, raw_sha256):
    """Keep aggregate diagnostics while hash-binding the two bulky array fields."""
    fields = {'per_coordinate_variance', 'singular_value_energy'}
    counts = {field: 0 for field in sorted(fields)}
    values = {field: 0 for field in sorted(fields)}

    def compact(node):
        if isinstance(node, dict):
            result = {}
            for key, item in node.items():
                if key in fields and isinstance(item, list):
                    require(all(type(number) in (int, float) and math.isfinite(number) for number in item),
                            f'collapse diagnostic {key} is not a finite numeric array')
                    counts[key] += 1
                    values[key] += len(item)
                    result[key] = {'publication_omitted_numeric_array': True, 'count': len(item),
                                   'canonical_sha256': hashlib.sha256(canonical(item)).hexdigest(),
                                   'min': min(item) if item else None,
                                   'max': max(item) if item else None,
                                   'mean': math.fsum(item) / len(item) if item else None}
                else:
                    result[key] = compact(item)
            return result
        if isinstance(node, list):
            return [compact(item) for item in node]
        return node

    require(isinstance(value, dict) and 'publication_array_summary' not in value,
            'unexpected collapse diagnostic publication envelope')
    result = compact(value)
    result['publication_array_summary'] = {
        'scope': 'Only per_coordinate_variance and singular_value_energy numeric arrays are replaced by count, canonical SHA-256, minimum, maximum and mean. All scalar variance/rank/participation metrics, provenance, budgets and traces are retained. Full arrays remain in the raw receipt.',
        'raw_receipt': 'collapse-diagnostics.json', 'raw_receipt_sha256': raw_sha256,
        'array_canonicalization': 'UTF-8 JSON with sorted keys, compact separators and no nonfinite numbers.',
        'arrays_replaced': counts, 'numeric_values_omitted': values,
    }
    return result


def export(run, output):
    run, output = Path(run).expanduser().resolve(), Path(output).expanduser().resolve()
    require(not output.is_relative_to(run), 'publication output must be outside the immutable raw run')
    for relative in REQUIRED:
        require((run / relative).is_file(), f'missing completed-study artifact: {relative}')
    plan, verification, summary = verify_completion(run)
    tests = test_summary(run / 'tests.log')
    check_hardware(read_json(run / 'hardware.json'))
    if (run / 'baseline-independent-verification.json').exists():
        require(read_json(run / 'baseline-independent-verification.json').get('status') == 'passed',
                'optional baseline independent verification did not pass')
    for relative in ('validation/cpu-integration.json', 'validation/mps-integration.json'):
        if (run / relative).exists():
            require(read_json(run / relative).get('status') == 'passed',
                    f'optional integration receipt did not pass: {relative}')
    raw = raw_inventory(run)
    missing = sorted(critical_paths(plan) - set(raw))
    require(not missing, 'critical raw evidence missing: ' + ', '.join(missing[:5]))
    require(raw['corpus/alice.txt']['sha256'] == plan['ordinary_corpus']['sha256'],
            'ordinary pretraining corpus differs from its frozen source receipt')
    checkpoint_receipts = verification['recipient_checkpoint_hashes']
    require(len(checkpoint_receipts) == 36 * plan['config']['stages'],
            'verification omitted a main recipient checkpoint hash')
    for relative, expected_hash in checkpoint_receipts.items():
        require(relative in raw and raw[relative]['sha256'] == expected_hash,
                f'checkpoint changed after independent verification: {relative}')
    if (run / 'continuations.json').exists():
        for entry in read_json(run / 'continuations.json')['entries']:
            require(entry['checkpoint'] in raw and raw[entry['checkpoint']]['sha256'] == entry['checkpoint_sha256'],
                    'continuation checkpoint differs from its completion receipt')
    names = list(REQUIRED) + [name for name in OPTIONAL if (run / name).is_file()]
    generated = {'tests-summary.json': tests}
    if (run / 'continuations.json').exists(): generated['continuation-summary.json'] = continuation_summary(run)
    audit = output / 'example-audit.json'
    audit_before = sha(audit) if audit.exists() else None
    if audit.exists():
        require(read_json(audit)['dataset_fingerprint'] == plan['dataset_sha256'], 'preserved example audit belongs to another dataset')
    warnings, transformations = [], {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.scratch-export-', dir=output.parent) as temporary:
        staging = Path(temporary)
        for relative in names:
            source, target = run / relative, staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == '.json': read_json(source)
            shutil.copy2(source, target)
            require(sha(target) == sha(source), f'copy changed evidence bytes: {relative}')
            if source.suffix == '.csv':
                original = source.read_bytes().decode('utf-8')
                normalized = original.replace('\r\n', '\n')
                require(list(csv.reader(io.StringIO(original))) == list(csv.reader(io.StringIO(normalized))),
                        'CSV newline normalization changed rows')
                target.write_text(normalized, encoding='utf-8', newline='\n')
                transformations[relative] = 'CSV CRLF record endings normalized to LF; all parsed rows and fields retained.'
            elif relative == 'collapse-diagnostics.json':
                compact = compact_collapse_diagnostics(read_json(source), raw[relative]['sha256'])
                target.write_text(json.dumps(compact, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
                transformations[relative] = 'Only per_coordinate_variance and singular_value_energy arrays summarized by count/canonical SHA-256/min/max/mean; all other fields retained. Full raw receipt SHA-256 is recorded in publication_array_summary.'
            elif source.suffix == '.json' and source.stat().st_size > SMALL_LIMIT:
                target.write_text(json.dumps(read_json(source), separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
                transformations[relative] = 'Lossless JSON formatting compaction; every field and metric retained.'
            if target.stat().st_size > SMALL_LIMIT:
                warnings.append(f'{relative}: {target.stat().st_size:,} bytes after publication compaction')
        for relative, value in generated.items():
            (staging / relative).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        for path in staging.rglob('*'):
            if path.is_file():
                relative = path.relative_to(staging)
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination)
    if audit_before is not None: require(sha(audit) == audit_before, 'example audit was changed during export')
    published = {}
    permitted_suffixes = {'.json', '.txt', '.md', '.html', '.csv', '.svg', '.png', '.pdf'}
    for path in sorted(output.rglob('*')):
        if path.is_file() and path != output / MANIFEST:
            require(path.suffix in permitted_suffixes and not path.is_symlink(), f'unexpected publication file: {path}')
            published[str(path.relative_to(output))] = file_entry(path)
            if path.stat().st_size > SMALL_LIMIT and str(path.relative_to(output)) not in transformations:
                note = f'{path.relative_to(output)}: {path.stat().st_size:,} published bytes'
                if note not in warnings: warnings.append(note)
    for relative, receipt in raw.items():
        require((run / relative).stat().st_size == receipt['bytes'] and sha(run / relative) == receipt['sha256'],
                f'raw evidence changed during export: {relative}')
    html_rows = (output / 'report.html').read_text(encoding='utf-8').count('<tr>')
    if html_rows > 1200: warnings.append(f'report.html has {html_rows} table rows; use the CSV for detailed comparisons')
    git = lambda *args: subprocess.check_output(['git', '-C', str(REPOSITORY), *args], text=True).strip()
    manifest = {'schema': 'lemmaferry.scratch.publication-evidence.v1',
        'exported_at_utc': datetime.now(timezone.utc).isoformat(), 'raw_root': str(run),
        'raw_root_note': 'User-authorized local evidence path; datasets, numeric checkpoints, arrays and logs remain here.',
        'starting_commit': plan['starting_commit'], 'branch': git('branch', '--show-current'), 'export_head': git('rev-parse', 'HEAD'),
        'exporter_sha256': sha(Path(__file__)), 'source_sha256': plan['source_sha256'],
        'dataset_fingerprint': plan['dataset_sha256'], 'conditions': 36,
        'independent_verification_status': verification['status'],
        'raw_files': raw, 'publication_files': published, 'transformations': transformations,
        'tests': tests, 'large_publication_files_or_density_notes': warnings,
        'preserved_example_audit_sha256': audit_before,
        'scope': 'Compact metrics, methods and independent verification, with raw evidence hashes. No raw weights, activation arrays, datasets, logs or private conversations copied. The manifest excludes itself from publication hashes.'}
    # Keep in-progress manifest files outside the publication inventory, so a
    # terminated exporter can be retried without publishing a partial receipt.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='.scratch-manifest-',
                                     suffix='.json', dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    try:
        os.replace(temporary, output / MANIFEST)
    finally:
        temporary.unlink(missing_ok=True)
    return {'status': 'exported', 'output': str(output), 'manifest_sha256': sha(output / MANIFEST),
            'raw_files_hashed': len(raw), 'publication_files': len(published), 'warnings': warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=REPOSITORY / 'artifacts/from-scratch')
    args = parser.parse_args(argv)
    try:
        result = export(args.run, args.output)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f'Export refused: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
