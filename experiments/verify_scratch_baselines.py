#!/usr/bin/env python3
"""Independently replay starting baselines after the complete main verification."""
import argparse
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lemmaferry.scratch.io import digest, read, write  # noqa: E402
from lemmaferry.scratch.verify import verify_evaluation, _recipient_state  # noqa: E402


def verify_baselines(run, device='cpu', samples=8):
    started = time.monotonic()
    run = Path(run).resolve()
    canonical_path = run / 'independent-verification.json'
    if not canonical_path.is_file() or read(canonical_path).get('status') != 'passed':
        raise ValueError('complete, passed main independent verification is required first')
    canonical, plan, dataset = read(canonical_path), read(run / 'plan.json'), read(run / 'dataset.json')
    plan_sha = digest((run / 'plan.json').read_bytes())
    if canonical['plan_sha256'] != plan_sha:
        raise ValueError('main verification refers to another study plan')
    if dataset['fingerprint'] != digest(dataset['rows']) or canonical['dataset']['fingerprint'] != dataset['fingerprint'] or plan['dataset_sha256'] != dataset['fingerprint']:
        raise ValueError('main verification dataset binding changed')
    repository = Path(__file__).resolve().parents[1]
    package = repository / 'src/lemmaferry'
    expected = {str(p.relative_to(repository)) for p in (package / 'scratch').rglob('*.py')} | {'src/lemmaferry/swap.py'}
    if set(plan['source_sha256']) != expected or canonical['source_hashes_checked'] != len(expected):
        raise ValueError('registered source inventory changed')
    for relative, checksum in plan['source_sha256'].items():
        path = repository / relative
        if not path.resolve().is_relative_to(repository) or digest(path.read_bytes()) != checksum:
            raise ValueError('registered source changed: ' + relative)
    config, seeds = plan['config'], []
    if canonical['conditions_checked'] != len(config['seeds']) * len(config['sizes']) * len(plan['arms']):
        raise ValueError('main independent verification is incomplete')
    for seed in config['seeds']:
        checkpoint = run / 'starting' / f'{seed}.pt'
        score_path = run / 'starting' / f'{seed}-evaluation.json'
        state = _recipient_state(checkpoint, config, config['pretrain_steps'])
        model_sha = state['model_sha256']; del state
        scores, splits = read(score_path), {}
        if set(scores) != {'test', 'harder'}:
            raise ValueError('starting baseline evaluation splits changed')
        for split in ('test', 'harder'):
            rows = [row for row in dataset['rows'] if row['split'] == split]
            replay = verify_evaluation(checkpoint, scores[split], rows, config,
                                       device=device, samples=samples, tolerance=1e-4)
            splits[split] = {'verification': replay, **{key: scores[split][key] for key in ('overall', 'by_family', 'by_depth')}}
        seeds.append({'seed': seed, 'checkpoint_sha256': digest(checkpoint.read_bytes()),
                      'model_state_sha256': model_sha, 'evaluation_sha256': digest(score_path.read_bytes()), 'splits': splits})
    return {'schema': 'lemmaferry.scratch.baseline-verification.v1', 'status': 'passed',
            'run': str(run), 'plan_sha256': plan_sha, 'dataset_fingerprint': dataset['fingerprint'],
            'main_independent_verification_sha256': digest(canonical_path.read_bytes()),
            'source_sha256': plan['source_sha256'], 'script_sha256': digest(Path(__file__).read_bytes()),
            'device': device, 'tolerance': 1e-4, 'samples_per_split_requested': samples,
            'seeds_checked': len(seeds), 'seeds': seeds, 'elapsed_seconds': time.monotonic() - started,
            'scope': 'Every recorded metric row checked; deterministic question-only likelihood/NLL and greedy-generation replay on the declared subset. No training or tuning.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.samples < 0: parser.error('--samples must be nonnegative')
    torch.set_num_threads(1)
    result = verify_baselines(args.run, args.device, args.samples)
    output = args.output or args.run / 'baseline-independent-verification.json'
    write(output, result)
    print({'status': result['status'], 'seeds_checked': result['seeds_checked'], 'output': str(output)})


if __name__ == '__main__':
    main()
