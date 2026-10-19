"""Independent, bounded replay of the from-scratch study's recorded evidence.

All certificates, row identities, target arrays, checkpoints and budgets are
checked. Expensive neural re-extraction and question-only answer replay use a
declared deterministic subset by default; the report states those exact counts.
This module does not call the experiment's training or evaluation functions.
"""
from __future__ import annotations

from collections import Counter
import gc
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from . import io, logic, model as net
from ..rosetta import PairedSplit
from ..translators import load_translator

ARMS = ('text_only', 'swap_paired', 'swap_shuffled', 'swap_mean')
QUESTION_KEYS = ('id', 'question', 'answer', 'family', 'depth')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tree_hash(value):
    # Public helper is shared only for canonical serialization, not for deciding
    # which states, rows or optimizer predecessors should be compared.
    return io.state_digest(value)


def checked_checkpoint(path):
    """Load numeric checkpoint data only after validating its fixed sidecar."""
    path = Path(path)
    try:
        pointer = io.read(path.with_suffix('.sha256.json'))
        _require(isinstance(pointer, dict) and set(pointer) == {'sha256'}, 'invalid checkpoint checksum pointer')
        _require(isinstance(pointer['sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', pointer['sha256']), 'invalid checkpoint SHA256')
        _require(io.digest(path.read_bytes()) == pointer['sha256'], 'checkpoint integrity failure')
        return torch.load(path, map_location='cpu', weights_only=True)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid checksum-pointed checkpoint: {path}') from exc


def validate_surface_controls(row):
    """Independently reject answer-length or ordered connective-shape shortcuts."""
    def skeleton(formula, depth=0):
        _require(depth < 64, 'surface formula exceeds structural budget')
        if type(formula) is int:
            _require(0 <= formula < 4, 'surface atom outside the declared scope')
            return 'atom'
        _require(isinstance(formula, list) and len(formula) == 3 and formula[0] in ('and', 'imp'),
                 'unrecognized connective in surface control')
        return (formula[0], skeleton(formula[1], depth + 1), skeleton(formula[2], depth + 1))
    _require(skeleton(row['goal']) == skeleton(row['distractor']), 'answer choices have different ordered connective skeletons')
    lines = row['question'].splitlines()
    a = [line[3:] for line in lines if line.startswith('A: ')]
    b = [line[3:] for line in lines if line.startswith('B: ')]
    _require(len(a) == len(b) == 1, 'question does not expose exactly two choice renderings')
    _require(len(a[0].encode()) == len(b[0].encode()), 'answer choices have different displayed byte lengths')


def validate_dataset(dataset):
    """Pure certificate/split checks; this does not claim that Lean was run."""
    _require(dataset.get('schema') == 2 and isinstance(dataset.get('rows'), list), 'unsupported dataset schema')
    rows = dataset['rows']
    _require(rows and io.digest(rows) == dataset.get('fingerprint'), 'dataset fingerprint mismatch')
    _require(dataset.get('common') == logic.COMMON and dataset.get('common_training_examples') == 0,
             'common library changed or became a training example')
    ids, splits = set(), Counter()
    for row in rows:
        split = row.get('split')
        _require(split in {'train', 'validation', 'test', 'harder'}, 'unknown frozen dataset split')
        plain = {k: v for k, v in row.items() if k != 'split'}
        logic.validate(plain)
        validate_surface_controls(row)
        _require(row['id'] == row['structural_identity'] and row['id'] not in ids, 'duplicate canonical theorem structure across splits')
        ids.add(row['id']); splits[split] += 1
        _require(set(row['dependencies']) <= {'identity', 'project'}, 'held-out theorem dependency entered the common library')
        _require(all(type(bit) is bool for bit in row['countermodel']), 'countermodel must be four Boolean values')
        bucket = int(row['id'][:8], 16) % 10
        if split == 'train': _require(bucket < 8, 'training theorem belongs to a held-out hash bucket')
        if split == 'validation': _require(bucket == 8, 'validation theorem has wrong canonical bucket')
        if split == 'test': _require(bucket == 9, 'test theorem has wrong canonical bucket')
        _require(row['proof_depth'] == logic.depth(row['proof']), 'proof-depth metadata mismatch')
        _require((row['depth'] >= 4 and row['proof_depth'] >= 5) if split == 'harder' else row['depth'] <= 3,
                 'harder/ordinary proof structures overlap by depth')
        _require(row['answer'] in ('A', 'B'), 'invalid expected answer')
        _require(max(len(text.encode()) + 2 for text in
                     (row['question'] + ' A', row['question'] + ' B', row['solution'], 'Proof:\n' + row['canonical_proof'])) <= dataset['context'],
                 'dataset exceeds the declared context; truncation is forbidden')
    _require(all(splits[split] for split in ('train', 'validation', 'test', 'harder')), 'a frozen dataset split is empty')
    return {'records': len(rows), 'split_counts': dict(splits), 'fingerprint': dataset['fingerprint'],
            'all_typed_proofs_checked': True, 'all_negative_countermodels_checked': True,
            'canonical_structures_disjoint': True, 'only_fixed_common_dependencies': True,
            'answer_choice_skeletons_and_display_lengths_matched': True}


def verify_dataset(root, *, replay_lean=True):
    root = Path(root)
    dataset = io.read(root / 'dataset.json')
    report = validate_dataset(dataset)
    admission = logic.admit(root, [{k: v for k, v in row.items() if k != 'split'} for row in dataset['rows']], replay=replay_lean)
    report.update(real_lean_admission=admission, fresh_lean_replay=replay_lean)
    extensions, known = [], {row['id'] for row in dataset['rows']}
    for path in sorted((root / 'extensions').glob('extension-*.json')):
        value = io.read(path)
        # Extension files never replace the study's registered dataset.
        parent = value.get('parent_fingerprint', value.get('parent_dataset_sha256', value.get('parent_dataset_fingerprint')))
        _require(parent == dataset['fingerprint'], 'extension points to another frozen dataset')
        extra = value.get('rows', [])
        _require(extra and value.get('fingerprint') == io.digest(extra), 'extension fingerprint mismatch')
        for row in extra:
            logic.validate({k: v for k, v in row.items() if k != 'split'})
            validate_surface_controls(row)
            _require(row['id'] not in known and int(row['id'][:8], 16) % 10 < 8, 'extension duplicates/leaks a held-out structure')
            _require(row.get('split') == 'extension', 'extension has a study/evaluation role')
            known.add(row['id'])
        logic.admit(root, [{k: v for k, v in row.items() if k != 'split'} for row in extra], replay=replay_lean)
        extensions.append({'path': str(path.relative_to(root)), 'records': len(extra), 'fingerprint': value['fingerprint']})
    report['extensions_checked'] = extensions
    return report


def _recipient_state(path, config, updates=None):
    state = checked_checkpoint(path)
    _require(state.get('schema') == net.SCHEMA, 'unsupported recipient checkpoint schema')
    _require(state.get('config') == config['model'], 'recipient architecture differs from the frozen plan')
    _require(state.get('tokenizer') == net.TOKENIZER and state.get('tokenizer_sha256') == io.digest(net.TOKENIZER), 'recipient tokenizer changed')
    for name in ('model', 'optimizer', 'rng'):
        _require(state.get(name + '_sha256') == _tree_hash(state[name]), f'recipient {name} inner checksum mismatch')
    _require(state.get('optimizer_class') == 'AdamW', 'recipient optimizer changed')
    if updates is not None: _require(state['updates'] == updates, 'recipient checkpoint has wrong cumulative updates')
    model = net.LM(net.ModelConfig(**config['model']))
    expected = model.state_dict()
    _require(set(expected) == set(state['model']), 'recipient parameter inventory changed')
    for name, value in state['model'].items():
        _require(isinstance(value, torch.Tensor) and value.shape == expected[name].shape and value.dtype == expected[name].dtype
                 and bool(torch.isfinite(value).all()), 'recipient parameter shape, dtype or finiteness mismatch')
    net._validate_optimizer(state['optimizer'], model)
    steps = {int(value['step']) for value in state['optimizer']['state'].values() if 'step' in value}
    _require(steps == ({state['updates']} if state['updates'] else set()), 'AdamW moment counters were reset or disagree')
    group = state['optimizer']['param_groups'][0]
    _require(group['lr'] == config['learning_rate'] and group['weight_decay'] == .01 and tuple(group['betas']) == (.9, .95), 'recipient optimizer settings differ from plan')
    return state


def _load_model(state, device):
    model = net.LM(net.ModelConfig(**state['config'])).to(device)
    model.load_state_dict(state['model'], strict=True)
    return model.eval().requires_grad_(False)


def _sample_indices(count, maximum):
    if maximum is None or maximum >= count: return list(range(count))
    _require(type(maximum) is int and maximum >= 0, 'replay sample budget must be nonnegative')
    return [] if not maximum else sorted(set(np.linspace(0, count - 1, maximum, dtype=int).tolist()))


def _encoded(text, context):
    ids = [net.BOS] + [byte + 3 for byte in text.encode()] + [net.EOS]
    _require(len(ids) <= context, 'overlength replay input; refusing truncation')
    return ids


@torch.inference_mode()
def _represent(model, texts, device):
    context = model.config.context
    ids = [_encoded(text, context) for text in texts]
    matrix = torch.full((len(ids), context), net.PAD, dtype=torch.long, device=device)
    for i, row in enumerate(ids): matrix[i, :len(row)] = torch.tensor(row, device=device)
    _, hidden = model(matrix)
    return hidden[torch.arange(len(ids), device=device), torch.tensor([len(row) - 2 for row in ids], device=device)].cpu()


def expected_schedule(train_count, config, seed, stage, rows=None):
    _require(config['batch'] % 2 == 0 and config['recipient_steps'] % config['stages'] == 0, 'unmatched stage or QA/proof budget')
    nsteps, per = config['recipient_steps'] // config['stages'], config['batch'] // 2
    order = np.random.default_rng(seed + stage).permutation(train_count).tolist()
    if config.get('priority_ids'):
        _require(rows is not None, 'priority schedule requires the admitted row identities')
        priority = set(config['priority_ids'])
        order = [i for i in order if rows[i]['id'] in priority] + [i for i in order if rows[i]['id'] not in priority]
    order = np.asarray(order)
    schedule = [order[(np.arange(per) + k * per) % train_count].tolist() for k in range(nsteps)]
    return schedule, {'updates': nsteps, 'problem_exposures': nsteps * per,
                      'supervised_tokens': nsteps * per * 67,
                      'padded_tokens': nsteps * config['batch'] * config['model']['context'],
                      'batch_order_sha256': io.digest(schedule),
                      'unique_problems_consumed': len({i for batch in schedule for i in batch})}


def _check_token_budget(rows, context):
    for row in rows:
        question = row['question']; text = question + ' ' + row['answer']
        ids = _encoded(text, context)
        _require(len(ids) - (1 + len(question.encode())) == 3, 'QA answer budget is not three byte tokens')
        proof_ids = _encoded('Proof:\n' + row['canonical_proof'], context)
        _require(len(proof_ids) - (1 + len('Proof:\n'.encode())) >= 64, 'proof cannot supply the registered 64 supervised tokens')


def _encoder(directory, train, validation, config):
    state = checked_checkpoint(directory / 'encoder.pt')
    metrics = io.read(directory / 'encoder-metrics.json')
    _require(state['step'] == config['encoder_steps'], 'proof encoder did not complete the registered update budget')
    _require(metrics['pair_exposures'] == config['encoder_steps'] * config['batch'], 'proof encoder exposure budget changed')
    _require(metrics['train_ids_sha256'] == io.digest([row['id'] for row in train]), 'proof encoder trained on another pool')
    receipt = {'train_ids_sha256': io.digest([row['id'] for row in train]),
               'validation_ids_sha256': io.digest([row['id'] for row in validation]),
               'encoder_train_content_sha256': io.digest([row['canonical_proof'] for row in train]),
               'encoder_validation_content_sha256': io.digest([row['canonical_proof'] for row in validation])}
    _require(state.get('schema') == 'lemmaferry.scratch.proof-encoder.v1' and state.get('receipt') == receipt,
             'proof encoder checkpoint lost its training-content binding')
    for key, expected in (('validation_ids_sha256', io.digest([row['id'] for row in validation])),
                          ('encoder_train_content_sha256', io.digest([row['canonical_proof'] for row in train])),
                          ('encoder_validation_content_sha256', io.digest([row['canonical_proof'] for row in validation]))):
        _require(metrics.get(key) == expected, f'proof encoder content receipt mismatch: {key}')
    _require(state['trace'] == metrics['trace'] and state['trace'], 'proof encoder validation trace changed')
    best = min(state['trace'], key=lambda row: (row['val_loss'], row['step']))
    _require(np.isfinite(best['val_loss']) and abs(state['best'] - best['val_loss']) < 1e-8, 'proof encoder checkpoint selection mismatch')
    _require(state['selected_step'] == best['step'] == metrics['selected_step'], 'proof encoder selected step changed')
    selected = state['best_state']
    _require(state['selected_state_sha256'] == _tree_hash(selected) == metrics['selected_state_sha256'], 'proof encoder selected state changed')
    _require(selected['to_z.weight'].shape[0] == 64, 'proof bottleneck is not the declared 64 dimensions')
    ae = net.ProofAE(context=config['model']['context'])
    ae.load_state_dict(selected, strict=True)
    _require(all(bool(torch.isfinite(value).all()) for value in selected.values()), 'nonfinite proof encoder weights')
    return ae.eval().requires_grad_(False), _tree_hash(selected)


@torch.inference_mode()
def _proof_vectors(ae, rows, context, device):
    ids = torch.full((len(rows), context), net.PAD, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        tokens = _encoded(row['canonical_proof'], context)
        ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
    return ae.to(device).encode(ids).cpu()


def _metadata_match(checkpoint, budget, key, expected):
    metadata = checkpoint['metadata']
    _require(metadata.get(key) == expected, f'recipient provenance mismatch: {key}')
    if key in budget: _require(budget[key] == expected, f'budget provenance mismatch: {key}')


def _evaluation_metrics(rows):
    _require(rows, 'empty independent evaluation')
    return {'correct': sum(row['correct'] for row in rows), 'total': len(rows),
            'accuracy': sum(row['correct'] for row in rows) / len(rows),
            'answer_loss': float(np.mean([row['answer_loss'] for row in rows])),
            'generation_correct': sum(row['generation_correct'] for row in rows),
            'generation_accuracy': sum(row['generation_correct'] for row in rows) / len(rows)}


def verify_continuations(root, config, splits, *, device='cpu', vector_samples=2,
                         evaluation_samples=8, tolerance=1e-4):
    path = root / 'continuations.json'
    if not path.exists(): return []
    history = io.read(path)
    _require(history.get('schema') == 1, 'unknown continuation history schema')
    checked, previous = [], {}
    for index, entry in enumerate(history['entries']):
        _require(entry['index'] == index and entry.get('complete') is True, 'continuation history is incomplete or unordered')
        seed, size, arm = entry['seed'], entry['size'], entry['arm']
        _require(seed in config['seeds'] and size in config['sizes'] and arm in ARMS, 'continuation refers to an unregistered starting condition')
        key = (seed, size, arm)
        expected_parent = previous.get(key, f'study/{seed}-{size}/{arm}/stage-{config["stages"]-1}/recipient.pt')
        _require(entry['parent_checkpoint'] == expected_parent, 'continuation reset or skipped its recipient/optimizer lineage')
        directory = root / entry['directory']
        _require(directory.resolve().is_relative_to(root.resolve()), 'continuation path leaves the run')
        extension = io.read(root / entry['extension'])
        current = [row for row in splits['train'][:size]]
        for earlier in history['entries'][:index+1]:
            if (earlier['seed'], earlier['size'], earlier['arm']) == key:
                current += io.read(root / earlier['extension'])['rows']
        validation = splits['validation']
        continuation_config = io.read(directory / 'config.json')
        expected_config = dict(config, recipient_steps=entry['updates'], stages=1,
                               priority_ids=[row['id'] for row in extension['rows']])
        _require(continuation_config == expected_config, 'continuation changed unregistered settings')
        parent = _recipient_state(root / expected_parent, config, entry['starting_updates'])
        checkpoint_path = root / entry['checkpoint']
        _require(checkpoint_path.resolve().is_relative_to(root.resolve()), 'continuation checkpoint leaves the run')
        _require(entry['ending_updates'] == entry['starting_updates'] + entry['updates'], 'continuation update count is inconsistent')
        state = _recipient_state(checkpoint_path, continuation_config, entry['ending_updates'])
        _require(entry['checkpoint_sha256'] == io.digest(checkpoint_path.read_bytes()), 'continuation checkpoint changed')
        _require(entry['base_dataset_sha256'] == io.read(root / 'dataset.json')['fingerprint']
                 and entry['extension_sha256'] == extension['fingerprint'], 'continuation dataset lineage changed')
        budget = io.read(checkpoint_path.parent / 'budget.json')
        schedule, expected_budget = expected_schedule(len(current), continuation_config, seed, 0, current)
        for name, expected in expected_budget.items(): _require(budget.get(name) == expected, 'continuation budget mismatch: ' + name)
        consumed = {current[i]['id'] for batch in schedule for i in batch}
        new_ids = {row['id'] for row in extension['rows']}
        _require(new_ids <= consumed and len(new_ids) == entry['additional'] == entry['new_problems_consumed'], 'continuation did not consume every newly admitted proof')
        ae, encoder_hash = _encoder(directory, current, validation, continuation_config)
        bridge = directory / 'shared-stage-0'
        pair, vectors = load_translator(bridge / 'maps'), checked_checkpoint(bridge / 'vectors.pt')
        tr = PairedSplit(vectors['x'].numpy(), vectors['y'].numpy(), [r['id'] for r in current], [r['id'] for r in current], 'train')
        va = PairedSplit(vectors['vx'].numpy(), vectors['vy'].numpy(), [r['id'] for r in validation], [r['id'] for r in validation], 'val')
        pair.validate_training_data(tr, va)
        _require(pair.protocol['seed'] == seed + 100 + index, 'continuation bridge seed changed')
        control = {'swap_paired': 'paired', 'swap_shuffled': 'shuffled', 'swap_mean': 'mean'}.get(arm)
        targets = torch.tensor(pair.to_language(vectors['x'].numpy(), control=control), dtype=torch.float32) if control else None
        target_hash = _tree_hash(targets) if targets is not None else None
        for name, expected in (('parameter_sha256_before', parent['model_sha256']),
                               ('optimizer_state_sha256_before', parent['optimizer_sha256']),
                               ('encoder_selected_state_sha256', encoder_hash),
                               ('frozen_recipient_parameter_sha256', parent['model_sha256']),
                               ('target_sha256', target_hash), ('stage_updates', entry['updates']),
                               ('train_ids_sha256', io.digest([r['id'] for r in current]))):
            _metadata_match(state, budget, name, expected)
        _require(budget.get('complete') is True and budget['optimizer_state_sha256_after'] == state['optimizer_sha256']
                 and budget['parameter_sha256_after'] == state['model_sha256'], 'continuation final state/budget mismatch')
        cost = io.read(bridge / 'cost.json')
        _require(cost['vectors_sha256'] == _tree_hash(vectors) and cost['receipt']['encoder_selected_state_sha256'] == encoder_hash
                 and cost['receipt']['frozen_recipient_parameter_sha256'] == parent['model_sha256'], 'continuation bridge producer mismatch')
        indices = _sample_indices(len(current), vector_samples)
        if indices:
            model = _load_model(parent, device)
            selected = [current[i] for i in indices]
            _require(torch.allclose(_proof_vectors(ae, selected, config['model']['context'], device), vectors['x'][indices], atol=tolerance, rtol=tolerance), 'continuation proof representation replay mismatch')
            _require(torch.allclose(_represent(model, [r['solution'] for r in selected], device), vectors['y'][indices], atol=tolerance, rtol=tolerance), 'continuation language snapshot is not its parent recipient')
            del model
        receipt = io.read(directory / 'evaluation.json')
        evaluation = {split: verify_evaluation(checkpoint_path, receipt[split], splits[split], config,
                                               device=device, samples=evaluation_samples, tolerance=tolerance) for split in ('test', 'harder')}
        checked.append({'index': index, 'new_problems_consumed': len(new_ids), 'starting_updates': entry['starting_updates'],
                        'ending_updates': entry['ending_updates'], 'target_sha256': target_hash,
                        'vector_rows_replayed': len(indices), 'question_only_evaluation': evaluation,
                        'scope': 'Separate exploratory continuation; the registered main study remains immutable.'})
        previous[key] = entry['checkpoint']
        del ae, pair, vectors, targets, parent, state; gc.collect()
    return checked


@torch.inference_mode()
def _replay_answer(model, row, device):
    question, answer = row['question'], row['answer']
    ids = _encoded(question + ' ' + answer, model.config.context)
    logits, _ = model(torch.tensor([ids], device=device))
    start = 1 + len(question.encode())
    letter_logits = logits[0, start, [ord('A') + 3, ord('B') + 3]]
    prediction = 'AB'[int(letter_logits.argmax())]
    loss = float(F.cross_entropy(logits[0, start - 1:-1], torch.tensor(ids[start:], device=device)).cpu())
    prefix = [net.BOS] + [byte + 3 for byte in question.encode()]
    generated = []
    for _ in range(4):
        current, _ = model(torch.tensor([prefix], device=device))
        token = int(current[0, -1].argmax())
        if token == net.EOS: break
        generated.append(token); prefix.append(token)
    generation = bytes([value - 3 for value in generated if 3 <= value < net.VOCAB]).decode(errors='replace')
    return prediction, loss, generation


def verify_evaluation(checkpoint_path, receipt, rows, config, *, device='cpu', samples=8, tolerance=1e-4):
    """Check every reported metric and replay a fixed question-only subset."""
    _require(receipt['checkpoint_sha256'] == io.digest(Path(checkpoint_path).read_bytes()), 'evaluation belongs to another recipient checkpoint')
    predictions = receipt['predictions']
    _require(len(predictions) == len(rows), 'evaluation count changed')
    for source, result in zip(rows, predictions):
        _require({key: result[key] for key in QUESTION_KEYS} == {key: source[key] for key in QUESTION_KEYS}, 'evaluation row order/content changed')
        _require(not ({'proof', 'canonical_proof', 'solution', 'translated_target', 'vectors'} & set(result)), 'proof/vector content entered question-only evaluation')
        _require(result['prediction'] in ('A', 'B') and result['correct'] == (result['prediction'] == source['answer']), 'answer accuracy receipt mismatch')
        _require(isinstance(result['generation'], str) and result['generation_correct'] == (result['generation'].strip() == source['answer']), 'generation accuracy receipt mismatch')
        _require(np.isfinite(result['answer_loss']) and result['answer_loss'] >= 0, 'invalid evaluation loss')
    expected = {'overall': _evaluation_metrics(predictions),
                'by_family': {family: _evaluation_metrics([row for row in predictions if row['family'] == family]) for family in sorted({row['family'] for row in predictions})},
                'by_depth': {str(depth): _evaluation_metrics([row for row in predictions if row['depth'] == depth]) for depth in sorted({row['depth'] for row in predictions})}}
    for group, metrics in expected.items():
        _require(receipt[group] == metrics, f'evaluation aggregate mismatch: {group}')
    indices = _sample_indices(len(rows), samples)
    if indices:
        state = _recipient_state(checkpoint_path, config)
        model = _load_model(state, device)
        for index in indices:
            prediction, loss, generation = _replay_answer(model, rows[index], device)
            recorded = predictions[index]
            _require(prediction == recorded['prediction'], 'standalone question-only likelihood answer changed')
            _require(abs(loss - recorded['answer_loss']) <= tolerance, 'standalone question-only answer loss changed')
            _require(generation == recorded['generation'], 'standalone free greedy generation changed')
        del model, state; gc.collect()
    return {'reported_rows_checked': len(rows), 'independently_replayed_rows': len(indices),
            'replayed_ids_sha256': io.digest([rows[index]['id'] for index in indices]),
            'question_only': True, 'greedy_generation_replayed': bool(indices)}


def verify_study(root, *, device='cpu', replay_lean=True, vector_samples=2,
                 evaluation_samples=8, require_complete=True, tolerance=1e-4):
    """Verify all expected arms/pools/seeds; expensive replays are explicitly bounded."""
    root = Path(root)
    dataset_report = verify_dataset(root, replay_lean=replay_lean)
    dataset, plan = io.read(root / 'dataset.json'), io.read(root / 'plan.json')
    config = plan['config']
    _require(plan.get('schema') == 1 and plan['arms'] == list(ARMS), 'frozen study arms/schema changed')
    _require(plan['dataset_sha256'] == dataset['fingerprint'], 'study dataset changed after registration')
    _require(config['model']['context'] == dataset['context'], 'registered model/dataset contexts differ')
    _require(config['stages'] == 2 and len(set(config['seeds'])) == len(config['seeds'])
             and sorted(set(config['sizes'])) == config['sizes'], 'invalid seed, nested-pool or two-stage study configuration')
    splits = {split: [row for row in dataset['rows'] if row['split'] == split]
              for split in ('train', 'validation', 'test', 'harder')}
    _require(plan['split_ids'] == {split: [row['id'] for row in rows] for split, rows in splits.items()}, 'registered split row IDs changed')
    source_hashes = plan.get('source_sha256', {})
    _require(source_hashes, 'study plan did not freeze source hashes')
    repository = Path(__file__).resolve().parents[3]
    expected_sources = {str(path.relative_to(repository)) for path in Path(__file__).parent.rglob('*.py')}
    expected_sources.add('src/lemmaferry/swap.py')
    _require(set(source_hashes) == expected_sources, 'source inventory changed after registration')
    for relative, expected in source_hashes.items():
        path = repository / relative
        _require(path.resolve().is_relative_to(repository) and io.digest(path.read_bytes()) == expected, 'source changed after study registration: ' + relative)
    if plan.get('development'):
        reserved = set(plan['development']['reserved_ids'])
        _require(not (reserved & {row['id'] for row in dataset['rows']}), 'development structures leaked into the main study')
        _require(dataset.get('reserved_ids_sha256') == io.digest(sorted(reserved)), 'development reservation binding changed')
    findings, incomplete, checkpoint_hashes = [], [], {}
    for seed in config['seeds']:
        starting_path = root / 'starting' / f'{seed}.pt'
        starting = _recipient_state(starting_path, config, config['pretrain_steps'])
        starting_model_hash, starting_optimizer_hash = starting['model_sha256'], starting['optimizer_sha256']
        del starting
        for size in config['sizes']:
            train = splits['train'][:size]; validation = splits['validation']
            _require(len(train) == size, 'registered nested pool exceeds the admitted training library')
            _check_token_budget(train, config['model']['context'])
            directory = root / 'study' / f'{seed}-{size}'
            if not (directory / 'encoder.pt').exists():
                incomplete.append(str(directory)); continue
            ae, encoder_hash = _encoder(directory, train, validation, config)
            x_anchor = None
            for arm in ARMS:
                previous_path = starting_path
                previous_model_hash, previous_optimizer_hash = starting_model_hash, starting_optimizer_hash
                arm_stages = []
                for stage in range(config['stages']):
                    folder = directory / arm / f'stage-{stage}'
                    checkpoint_path = folder / 'recipient.pt'
                    if not checkpoint_path.exists() or not (folder / 'budget.json').exists():
                        incomplete.append(str(folder)); break
                    nsteps = config['recipient_steps'] // config['stages']
                    cumulative = config['pretrain_steps'] + (stage + 1) * nsteps
                    checkpoint = _recipient_state(checkpoint_path, config, cumulative)
                    budget = io.read(folder / 'budget.json')
                    schedule, expected_budget = expected_schedule(size, config, seed, stage)
                    for key, expected in expected_budget.items(): _require(budget.get(key) == expected, 'recipient budget/schedule mismatch: ' + key)
                    _require(expected_budget['unique_problems_consumed'] == size, 'fixed budget did not consume every distinct training proof')
                    for key, expected in (('parameter_sha256_before', previous_model_hash),
                                          ('optimizer_state_sha256_before', previous_optimizer_hash),
                                          ('encoder_selected_state_sha256', encoder_hash),
                                          ('frozen_recipient_parameter_sha256', previous_model_hash),
                                          ('arm', arm), ('seed', seed), ('stage', stage), ('stage_updates', nsteps),
                                          ('train_ids_sha256', io.digest([row['id'] for row in train]))):
                        _metadata_match(checkpoint, budget, key, expected)
                    _require(budget.get('optimizer_state_sha256_after') == checkpoint['optimizer_sha256']
                             and budget.get('parameter_sha256_after') == checkpoint['model_sha256']
                             and budget.get('complete') is True, 'recipient final state/budget completion mismatch')
                    bridge = directory / ('shared-stage-0' if stage == 0 else f'{arm}/stage-{stage}')
                    pair, vectors = load_translator(bridge / 'maps'), checked_checkpoint(bridge / 'vectors.pt')
                    cost = io.read(bridge / 'cost.json')
                    _require(cost['vectors_sha256'] == _tree_hash(vectors), 'bridge vector inventory hash changed')
                    producer_receipt = cost['receipt']
                    _require(producer_receipt['encoder_selected_state_sha256'] == encoder_hash
                             and producer_receipt['frozen_recipient_parameter_sha256'] == previous_model_hash
                             and producer_receipt['train_ids_sha256'] == io.digest([row['id'] for row in train])
                             and producer_receipt['validation_ids_sha256'] == io.digest([row['id'] for row in validation]),
                             'bridge producer/data provenance mismatch')
                    arrays = {key: value.detach().cpu().numpy() for key, value in vectors.items()}
                    _require(set(arrays) == {'x', 'y', 'vx', 'vy'}, 'unexpected bridge vector fields')
                    tr = PairedSplit(arrays['x'], arrays['y'], [row['id'] for row in train], [row['id'] for row in train], 'train')
                    va = PairedSplit(arrays['vx'], arrays['vy'], [row['id'] for row in validation], [row['id'] for row in validation], 'val')
                    pair.validate_training_data(tr, va)
                    _require(pair.protocol['seed'] == seed + stage, 'bridge seed differs from registered stage')
                    from ..swap import SwapConfig
                    _require(pair.protocol['config'] == vars(SwapConfig(latent_dim=32, hidden_dim=128, steps=config['bridge_steps'], batch_size=32, validation_every=64, device='cpu')), 'bridge training configuration differs from plan')
                    _require(arrays['x'].shape[1] == 64 and arrays['y'].shape[1] == config['model']['width'], 'proof/language representation dimensions changed')
                    source_anchor = io.digest([_tree_hash(vectors['x']), _tree_hash(vectors['vx'])])
                    if x_anchor is None: x_anchor = source_anchor
                    _require(x_anchor == source_anchor, 'frozen proof encoder vectors changed between arms/stages')
                    indices, vindices = _sample_indices(size, vector_samples), _sample_indices(len(validation), vector_samples)
                    if indices or vindices:
                        model = _load_model(_recipient_state(previous_path, config), device)
                        for source, ix, xx, yy in ((train, indices, 'x', 'y'), (validation, vindices, 'vx', 'vy')):
                            if not ix: continue
                            selected = [source[i] for i in ix]
                            actual_x = _proof_vectors(ae, selected, config['model']['context'], device)
                            actual_y = _represent(model, [row['solution'] for row in selected], device)
                            _require(torch.allclose(actual_x, vectors[xx][ix], atol=tolerance, rtol=tolerance), 'frozen proof vector replay differs')
                            _require(torch.allclose(actual_y, vectors[yy][ix], atol=tolerance, rtol=tolerance), 'bridge language states are not from the recipient before this update')
                        del model; gc.collect()
                    control = {'swap_paired': 'paired', 'swap_shuffled': 'shuffled', 'swap_mean': 'mean'}.get(arm)
                    targets = torch.tensor(pair.to_language(arrays['x'], control=control), dtype=torch.float32) if control else None
                    target_hash = _tree_hash(targets) if targets is not None else None
                    _metadata_match(checkpoint, budget, 'target_sha256', target_hash)
                    geometry = io.read(folder / 'geometry.json')
                    _require(geometry['protocol'] == pair.protocol and geometry['test']['ids'] == [row['id'] for row in splits['test'][:128]], 'held-out geometry uses another map/test set')
                    previous_path, previous_model_hash, previous_optimizer_hash = checkpoint_path, checkpoint['model_sha256'], checkpoint['optimizer_sha256']
                    checkpoint_hashes[str(checkpoint_path.relative_to(root))] = io.digest(checkpoint_path.read_bytes())
                    arm_stages.append({'stage': stage, 'budget': expected_budget, 'target_sha256': target_hash,
                                       'full_vector_hashes': {name: _tree_hash(value) for name, value in vectors.items()},
                                       'train_vectors_replayed': len(indices), 'validation_vectors_replayed': len(vindices)})
                    del checkpoint, vectors, arrays, targets, pair; gc.collect()
                if len(arm_stages) != config['stages']: continue
                path = directory / arm / 'evaluation.json'
                if not path.exists(): incomplete.append(str(path)); continue
                evaluation = io.read(path)
                _require(set(evaluation) == {'test', 'harder'}, 'independent evaluation splits changed')
                replay = {split: verify_evaluation(previous_path, evaluation[split], splits[split], config,
                                                   device=device, samples=evaluation_samples, tolerance=tolerance)
                          for split in ('test', 'harder')}
                findings.append({'seed': seed, 'size': size, 'arm': arm, 'stages': arm_stages, 'question_only_evaluation': replay})
            del ae; gc.collect()
    if require_complete: _require(not incomplete and len(findings) == len(config['seeds']) * len(config['sizes']) * len(ARMS), 'study incomplete: ' + ', '.join(incomplete[:8]))
    continuation_report = verify_continuations(root, config, splits, device=device, vector_samples=vector_samples,
                                               evaluation_samples=evaluation_samples, tolerance=tolerance)
    return {'schema': 1, 'status': 'passed' if not incomplete else 'partial', 'dataset': dataset_report,
            'plan_sha256': io.digest((root / 'plan.json').read_bytes()), 'source_hashes_checked': len(source_hashes),
            'conditions_checked': len(findings), 'incomplete_paths': incomplete, 'conditions': findings,
            'recipient_checkpoint_hashes': checkpoint_hashes,
            'continuations': continuation_report,
            'replay_policy': {'device': device, 'vector_samples_per_split_per_bridge': vector_samples,
                              'question_samples_per_evaluation_split': evaluation_samples, 'tolerance': tolerance,
                              'all_targets_all_checkpoint_hashes_all_metric_rows_checked': True},
            'scope': 'Integrity and explicitly bounded independent replay; no conclusion about useful or sustained general reasoning transfer.'}


def verify(root, **kwargs):
    return verify_study(root, **kwargs)
