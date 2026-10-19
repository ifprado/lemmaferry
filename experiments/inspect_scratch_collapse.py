#!/usr/bin/env python3
"""Post-fit variance diagnostics for frozen scratch bridges and recipients.

Only registered training/validation representations and optional validation
questions are read. No test scores, thresholds, updates or model selection are
performed. Effective rank describes centered covariance energy, not semantics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lemmaferry.scratch import io, model as net  # noqa: E402
from lemmaferry.scratch.verify import checked_checkpoint, _recipient_state  # noqa: E402
from lemmaferry.rosetta import PairedSplit  # noqa: E402
from lemmaferry.translators import load_translator  # noqa: E402

ARMS = ('text_only', 'swap_paired', 'swap_shuffled', 'swap_mean')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def variance_statistics(values, *, mean=None, scale=None, effective_rank=True):
    """Population variance after optional frozen train-only standardization.

    Subtracting the first row before centering preserves an exactly constant
    output as exact zero, including a nonzero training-mean control. Rank uses
    squared singular-value probabilities and is zero for constant matrices.
    """
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 2 and min(array.shape) > 0 and np.isfinite(array).all(),
            'variance input must be a nonempty finite matrix')
    if mean is not None or scale is not None:
        mean, scale = np.asarray(mean, dtype=float), np.asarray(scale, dtype=float)
        require(mean.shape == scale.shape == (array.shape[1],) and np.isfinite(mean).all()
                and np.isfinite(scale).all() and np.all(scale > 0), 'invalid frozen training statistics')
        array = (array - mean) / scale
    shifted = array - array[:1]
    centered = shifted - shifted.mean(axis=0, keepdims=True)
    variances = np.mean(centered ** 2, axis=0)
    result = {'rows': len(array), 'dimensions': array.shape[1],
              'mean_coordinate_variance': float(variances.mean()),
              'per_coordinate_variance': variances.tolist(),
              'exactly_constant_across_rows': bool(np.all(shifted == 0))}
    if effective_rank:
        energies = np.linalg.svd(centered, compute_uv=False) ** 2
        total = float(energies.sum())
        if total > 0:
            probabilities = energies[energies > 0] / total
            entropy_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
            participation_rank = float(total ** 2 / np.sum(energies ** 2))
        else:
            entropy_rank = participation_rank = 0.0
        result.update(covariance_entropy_effective_rank=entropy_rank,
                      covariance_participation_rank=participation_rank,
                      singular_value_energy=energies.tolist(),
                      algebraic_centered_rank_bound=min(len(array) - 1, array.shape[1]))
    return result


def _stats(values, scaler, effective_rank):
    return variance_statistics(values, mean=scaler.mean, scale=scaler.scale,
                               effective_rank=effective_rank)


def _relative(prediction, target, scaler, effective_rank):
    output = _stats(prediction, scaler, effective_rank)
    target_stats = _stats(target, scaler, effective_rank)
    denominator = target_stats['mean_coordinate_variance']
    return {'output': output, 'corresponding_target': target_stats,
            'output_to_target_variance_ratio': output['mean_coordinate_variance'] / denominator if denominator else None,
            'target_variance_is_zero': denominator == 0.0,
            'normalized_rmse': float(np.sqrt(np.mean(((np.asarray(prediction) - target) / scaler.scale) ** 2)))}


def _hash_file(path):
    return io.digest(Path(path).read_bytes())


def _indices(count, maximum):
    return sorted(set(np.linspace(0, count - 1, min(maximum, count), dtype=int).tolist()))


def inspect(run, *, recipient_samples=0, effective_rank=True, require_complete=False):
    run = Path(run).resolve()
    require(type(recipient_samples) is int and recipient_samples >= 0, 'recipient sample count must be nonnegative')
    plan, dataset = io.read(run / 'plan.json'), io.read(run / 'dataset.json')
    config = plan['config']
    require(plan['arms'] == list(ARMS), 'unknown registered study arms')
    require(dataset['fingerprint'] == io.digest(dataset['rows']) == plan['dataset_sha256'],
            'dataset changed after study registration')
    splits = {split: [row for row in dataset['rows'] if row['split'] == split]
              for split in ('train', 'validation', 'test', 'harder')}
    require(plan['split_ids'] == {split: [row['id'] for row in rows] for split, rows in splits.items()},
            'registered dataset row identities changed')
    repository = Path(__file__).resolve().parents[1]
    for relative, expected in plan['source_sha256'].items():
        path = repository / relative
        require(path.resolve().is_relative_to(repository) and _hash_file(path) == expected,
                'registered implementation changed: ' + relative)
    hashes = {'plan.json': _hash_file(run / 'plan.json'), 'dataset.json': _hash_file(run / 'dataset.json')}
    bridges, by_path = [], {}
    cost_paths = [path for path in sorted((run / 'study').rglob('cost.json'))
                  if not any(part.startswith('.') for part in path.relative_to(run).parts)]
    for cost_path in cost_paths:
        directory = cost_path.parent
        if directory.name == 'shared-stage-0':
            condition, stage, arm = directory.parent, 0, None
        else:
            require(directory.name.startswith('stage-') and directory.parent.name in ARMS,
                    'unregistered bridge location: ' + str(directory))
            condition, stage, arm = directory.parent.parent, int(directory.name.split('-')[1]), directory.parent.name
            require(1 <= stage < config['stages'], 'bridge stage outside registration')
        seed, size = map(int, condition.name.split('-'))
        require(seed in config['seeds'] and size in config['sizes'], 'unregistered bridge condition')
        cost = io.read(cost_path)
        pair = load_translator(directory / 'maps')
        tensors = checked_checkpoint(directory / 'vectors.pt')
        require(set(tensors) == {'x', 'y', 'vx', 'vy'} and all(isinstance(value, torch.Tensor) for value in tensors.values()),
                'unexpected bridge vector inventory')
        require(cost['vectors_sha256'] == io.state_digest(tensors), 'bridge vector content hash differs from cost receipt')
        arrays = {key: value.detach().cpu().numpy() for key, value in tensors.items()}
        train, validation = splits['train'][:size], splits['validation']
        require(len(train) == size, 'training pool is incomplete')
        tr = PairedSplit(arrays['x'], arrays['y'], [r['id'] for r in train], [r['id'] for r in train], 'train')
        va = PairedSplit(arrays['vx'], arrays['vy'], [r['id'] for r in validation], [r['id'] for r in validation], 'val')
        pair.validate_training_data(tr, va)
        require(pair.protocol['seed'] == seed + stage, 'bridge seed differs from registered stage')
        producer = cost['receipt']
        require(producer['train_ids_sha256'] == io.digest([r['id'] for r in train])
                and producer['validation_ids_sha256'] == io.digest([r['id'] for r in validation]),
                'bridge producer rows differ from the registered condition')
        require(producer['source_sha256'] == plan['source_sha256'], 'bridge was produced by another implementation')
        encoder_path = condition / 'encoder.pt'
        encoder = checked_checkpoint(encoder_path)
        encoder_metrics = io.read(condition / 'encoder-metrics.json')
        encoder_hash = io.state_digest(encoder['best_state'])
        require(encoder_hash == encoder['selected_state_sha256'] == encoder_metrics['selected_state_sha256']
                == producer['encoder_selected_state_sha256'], 'selected proof encoder producer changed')
        selected_step = encoder_metrics['selected_step']
        selected_trace = [row for row in encoder_metrics['trace'] if row['step'] == selected_step]
        require(len(selected_trace) == 1, 'selected proof encoder variance receipt is missing')
        snapshot_path = (run / 'starting' / f'{seed}.pt' if stage == 0
                         else condition / arm / f'stage-{stage-1}' / 'recipient.pt')
        snapshot = checked_checkpoint(snapshot_path)
        require(snapshot['model_sha256'] == io.state_digest(snapshot['model']) == producer['frozen_recipient_parameter_sha256'],
                'language representations refer to another frozen recipient')
        directions = {}
        for name, inputs, targets, scaler, method in (
            ('math_to_language', arrays['vx'], arrays['vy'], pair.language_scaler, pair.to_language),
            ('language_to_math', arrays['vy'], arrays['vx'], pair.math_scaler, pair.to_math),
        ):
            directions[name] = {}
            for control in ('paired', 'shuffled', 'mean'):
                prediction = method(inputs, control=control)
                values = _relative(prediction, targets, scaler, effective_rank)
                if control == 'mean':
                    require(values['output']['mean_coordinate_variance'] <= 1e-12,
                            'training-mean control is not constant')
                directions[name][control] = values
        record = {'path': str(directory.relative_to(run)), 'seed': seed, 'size': size,
                  'stage': stage, 'recipient_arm': arm or 'shared_all_arms',
                  'frozen_producer': producer,
                  'training': pair.protocol['training'],
                  'bridge_config': pair.protocol['config'],
                  'shuffled_permutation_sha256': io.digest(pair.protocol['shuffled_train_permutation']),
                  'validation_ids_sha256': io.digest([r['id'] for r in validation]),
                  'full_vector_content_sha256': {key: io.state_digest(value) for key, value in tensors.items()},
                  'source_math_validation_raw': variance_statistics(arrays['vx'], effective_rank=effective_rank),
                  'source_math_validation_standardized': _stats(arrays['vx'], pair.math_scaler, effective_rank),
                  'language_target_validation_raw': variance_statistics(arrays['vy'], effective_rank=effective_rank),
                  'proof_encoder': {'selected_step': selected_step, 'selected_state_sha256': encoder_hash,
                                    'selected_validation_embedding_variance_recorded': selected_trace[0]['embedding_variance'],
                                    'recorded_variance_convention': 'torch.var(dim=0), sample variance with n-1 denominator',
                                    'measured_variance_convention': 'population variance with n denominator',
                                    'validation_embedding_variance_measured_raw': variance_statistics(arrays['vx'], effective_rank=False)['mean_coordinate_variance']},
                  'directions': directions}
        bridges.append(record)
        by_path[directory.resolve()] = (record, arrays, pair)
        for path in (cost_path, directory / 'vectors.pt', directory / 'vectors.sha256.json',
                     directory / 'maps/manifest.json', directory / 'maps/maps.npz', encoder_path,
                     encoder_path.with_suffix('.sha256.json'), condition / 'encoder-metrics.json', snapshot_path,
                     snapshot_path.with_suffix('.sha256.json')):
            hashes[str(path.relative_to(run))] = _hash_file(path)
        del encoder, snapshot, tensors
    expected_count = len(config['seeds']) * len(config['sizes']) * (1 + len(ARMS) * (config['stages'] - 1))
    require(not require_complete or len(bridges) == expected_count, 'registered bridge study is incomplete')
    recipients, incomplete_recipients = [], []
    if recipient_samples:
        indices = _indices(len(splits['validation']), recipient_samples)
        selected = [splits['validation'][i] for i in indices]
        for seed in config['seeds']:
            for size in config['sizes']:
                condition = run / 'study' / f'{seed}-{size}'
                for arm in ARMS:
                    directory = condition / arm / f'stage-{config["stages"]-1}'
                    checkpoint_path = directory / 'recipient.pt'
                    if not (checkpoint_path.exists() and (directory / 'budget.json').exists()
                            and directory.resolve() in by_path):
                        incomplete_recipients.append(str(directory.relative_to(run))); continue
                    record, arrays, pair = by_path[directory.resolve()]
                    state = _recipient_state(checkpoint_path, config, config['pretrain_steps'] + config['recipient_steps'])
                    model = net.LM(net.ModelConfig(**config['model'])).eval().requires_grad_(False)
                    model.load_state_dict(state['model'], strict=True)
                    # Only the validation QUESTION is supplied to the recipient.
                    values = net.represent(model, [row['question'] for row in selected], 'cpu').numpy()
                    require(io.state_digest(model.state_dict()) == state['model_sha256'], 'recipient diagnostic changed its parameters')
                    control = {'swap_paired': 'paired', 'swap_shuffled': 'shuffled', 'swap_mean': 'mean'}.get(arm)
                    target = pair.to_language(arrays['vx'][indices], control=control) if control else None
                    recipients.append({'seed': seed, 'size': size, 'arm': arm,
                                       'checkpoint': str(checkpoint_path.relative_to(run)),
                                       'model_state_sha256': state['model_sha256'],
                                       'question_ids_sha256': io.digest([row['id'] for row in selected]),
                                       'rows': len(selected), 'question_only': True,
                                       'question_states_sha256': io.state_digest(torch.tensor(values)),
                                       'question_states_raw': variance_statistics(values, effective_rank=effective_rank),
                                       'question_states_standardized': _stats(values, pair.language_scaler, effective_rank),
                                       'frozen_solution_targets_standardized': _stats(arrays['vy'][indices], pair.language_scaler, effective_rank),
                                       'auxiliary_control_targets_standardized': _stats(target, pair.language_scaler, effective_rank) if target is not None else None,
                                       'normalization': 'final bridge pre-update training-language mean/std; no recipient/validation refitting'})
                    hashes[str(checkpoint_path.relative_to(run))] = _hash_file(checkpoint_path)
                    hashes[str(checkpoint_path.with_suffix('.sha256.json').relative_to(run))] = _hash_file(checkpoint_path.with_suffix('.sha256.json'))
                    del model, state
        require(not require_complete or not incomplete_recipients, 'registered recipient study is incomplete')
    return {'schema': 'lemmaferry.scratch.collapse-diagnostics.v1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'completed' if len(bridges) == expected_count and not incomplete_recipients else 'partial',
            'run': str(run), 'plan_sha256': hashes['plan.json'], 'dataset_fingerprint': dataset['fingerprint'],
            'script_sha256': _hash_file(Path(__file__)), 'registered_source_sha256': plan['source_sha256'],
            'policy': {'post_fit_only': True, 'data_split': 'validation', 'fit_or_update_performed': False,
                       'selection_or_threshold_tuning': False, 'coordinate_variance': 'population variance; divide by frozen training std before measuring',
                       'rank_definition': 'exp entropy of normalized squared singular values of centered states',
                       'mean_control_zero_tolerance': 1e-12, 'recipient_samples_requested': recipient_samples,
                       'effective_rank_computed': effective_rank, 'test_scores_read': False},
            'completed_bridges': len(bridges), 'expected_bridges': expected_count, 'bridges': bridges,
            'recipient_diagnostics': recipients, 'incomplete_recipients': incomplete_recipients,
            'file_sha256': hashes,
            'interpretation': 'Low output variance relative to target variance indicates a nearly constant mapping on these validation examples. Variance, effective rank and regression error do not establish semantic or reasoning transfer.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--recipient-samples', type=int, default=0)
    parser.add_argument('--no-effective-rank', action='store_true')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = inspect(args.run, recipient_samples=args.recipient_samples,
                     effective_rank=not args.no_effective_rank, require_complete=args.require_complete)
    io.write(args.output, report)
    print({'status': report['status'], 'bridges': report['completed_bridges'],
           'recipients': len(report['recipient_diagnostics']), 'output': str(args.output)})


if __name__ == '__main__':
    main()
