"""Offline, self-contained evidence report for the local from-scratch study.

This reader never trains, loads a neural checkpoint, or makes a network request.
Partial runs remain explicitly partial. Capability scores and bridge geometry
are displayed separately; no positive outcome is required for completion.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import statistics

ARMS = ('text_only', 'swap_paired', 'swap_shuffled', 'swap_mean')
LABELS = {'text_only': 'Same-proof text', 'swap_paired': 'Paired translation',
          'swap_shuffled': 'Shuffled translation', 'swap_mean': 'Mean target'}
COLORS = {'text_only': '#a25519', 'swap_paired': '#087e96',
          'swap_shuffled': '#8054a5', 'swap_mean': '#75828c'}
BUDGET_KEYS = ('updates', 'problem_exposures', 'supervised_tokens', 'padded_tokens')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values):
    return statistics.mean(values) if values else None


def _sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.


def _format(value, digits=3):
    if value is None:
        return 'pending'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, int):
        return f'{value:,}'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def _table(headers, rows):
    escaped = lambda value: html.escape(_format(value))
    return ('<div class="scroll"><table><thead><tr>' + ''.join('<th>' + html.escape(h) + '</th>' for h in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + escaped(v) + '</td>' for v in row) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def _metrics(value):
    result = {key: value.get(key) for key in ('correct', 'total', 'accuracy', 'generation_correct',
                                             'generation_accuracy', 'answer_loss')}
    for key in ('correct', 'total', 'generation_correct'):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f'invalid evaluation count {key}')
    total = result['total']
    if not total or result['correct'] > total or result['generation_correct'] > total:
        raise ValueError('invalid evaluation numerator/denominator')
    for key, numerator in (('accuracy', 'correct'), ('generation_accuracy', 'generation_correct')):
        if not _number(result[key]) or abs(result[key] - result[numerator] / total) > 1e-9:
            raise ValueError('evaluation accuracy disagrees with recorded counts')
    if not _number(result['answer_loss']) or result['answer_loss'] < 0:
        raise ValueError('invalid answer NLL')
    return result


def _curve(rows, split, key='accuracy', *, title='', ylabel='Accuracy (%)', percent=True):
    sizes = sorted({row['size'] for row in rows if row['split'] == split and row['arm'] in ARMS})
    if not sizes:
        return '<p class="muted">Curve pending completed recipient evaluations.</p>'
    width, height, left, top, right, bottom = 760, 310, 68, 28, 24, 58
    plotw, ploth = width - left - right, height - top - bottom
    grouped = {}
    for row in rows:
        if row['split'] == split and row['arm'] in ARMS and _number(row.get(key)):
            grouped.setdefault((row['arm'], row['size']), []).append(row[key] * (100 if percent else 1))
    upper = 100 if percent else max(1., max((max(v) for v in grouped.values()), default=1.) * 1.15)
    lower = 0.
    def xx(size):
        return left + plotw * (.5 if len(sizes) == 1 else sizes.index(size) / (len(sizes) - 1))
    def yy(value):
        return top + ploth * (1 - (max(lower, min(upper, value)) - lower) / (upper - lower))
    elements = [f'<svg role="img" aria-label="{html.escape(title)}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#fff"/>']
    for index in range(6):
        value = lower + (upper - lower) * index / 5
        y = yy(value)
        elements += [f'<path d="M{left},{y:.2f}H{width-right}" stroke="#e4e9ee"/>',
                     f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end">{value:.0f}</text>']
    for size in sizes:
        elements.append(f'<text x="{xx(size):.2f}" y="{height-bottom+25}" text-anchor="middle">{size:,}</text>')
    elements += [f'<text x="{left+plotw/2}" y="{height-9}" text-anchor="middle">Distinct verified training problems</text>',
                 f'<text transform="translate(17,{top+ploth/2}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>']
    for arm in ARMS:
        points = []
        for size in sizes:
            values = grouped.get((arm, size), [])
            if not values:
                continue
            mean, sd = _mean(values), _sd(values)
            x, y = xx(size), yy(mean)
            points.append(f'{x:.2f},{y:.2f}')
            description = html.escape(f'{LABELS[arm]}, {size} proofs: mean {mean:.3f}, sample SD {sd:.3f}, seeds={len(values)}')
            elements.append(f'<g stroke="{COLORS[arm]}" fill="{COLORS[arm]}"><title>{description}</title>'
                f'<path d="M{x:.2f},{yy(mean-sd):.2f}V{yy(mean+sd):.2f}" stroke-width="2"/>'
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4"/></g>')
            for index, value in enumerate(values):
                jitter = (index - (len(values) - 1) / 2) * 5
                elements.append(f'<circle cx="{x+jitter:.2f}" cy="{yy(value):.2f}" r="2" fill="{COLORS[arm]}" opacity=".45"/>')
        if points:
            elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[arm]}" stroke-width="2"/>')
    elements.append('</svg>')
    legend = ''.join(f'<span style="color:{COLORS[a]}">● {LABELS[a]}</span> ' for a in ARMS)
    return '<figure>' + ''.join(elements) + '<figcaption>' + legend + '<br>Mean across seeds; whiskers show ± one sample standard deviation, clipped to the plot. Faint dots show seeds. These are not confidence intervals.</figcaption></figure>'


def report(root, destination=None):
    """Write root/report.html, root/summary.json, root/results.csv; return summary.

    An optional destination receives additional copies (a directory, or an HTML
    filename with JSON/CSV beside it). The original run remains the source.
    """
    root = Path(root).resolve()
    receipts = {}
    def read(relative, default=None):
        path = root / relative
        if not path.exists():
            return default
        value = _json(path)
        receipts[str(relative)] = _sha(path)
        return value
    plan = read('plan.json', {})
    config = plan.get('config') or read('selected-config.json', {})
    pilot, development = read('pilot.json', {}), read('development.json', {})
    progress = read('progress.json', {})
    seeds, sizes = config.get('seeds', []), config.get('sizes', [])
    arms = plan.get('arms', list(ARMS))
    rows, encoders, geometry, budgets, bridge_costs, pretraining, missing = [], [], [], [], [], [], []
    if not plan:
        missing.append('plan.json (study not registered)')
    for seed in seeds:
        pre = read(f'starting/{seed}-pretraining.json')
        if pre is not None:
            pretraining.append({'seed': seed, **pre})
        else:
            missing.append(f'starting/{seed}-pretraining.json')
        baseline = read(f'starting/{seed}-evaluation.json')
        if baseline:
            for split in ('test', 'harder'):
                if split in baseline:
                    rows.append({'seed': seed, 'size': 0, 'arm': 'pretrained_baseline', 'split': split,
                                 **_metrics(baseline[split]['overall']), 'delta_vs_text_pp': None,
                                 'checkpoint': str(root / 'starting' / f'{seed}.pt')})
        if not baseline:
            missing.append(f'starting/{seed}-evaluation.json')
        for size in sizes:
            prefix = f'study/{seed}-{size}'
            encoder = read(prefix + '/encoder-metrics.json')
            if encoder:
                trace = encoder.get('trace', [])
                selected = next((r for r in trace if r.get('step') == encoder.get('selected_step')), None)
                if selected is None and trace:
                    selected = min(trace, key=lambda r: r['val_loss'])
                encoders.append({'seed': seed, 'size': size, 'selected_step': encoder.get('selected_step'),
                    'pair_exposures': encoder.get('pair_exposures'), 'seconds': encoder.get('seconds_this_session'),
                    'selected': selected, 'selected_state_sha256': encoder.get('selected_state_sha256'),
                    'diagnostics': {'real_vector_beats_zero': selected['val_loss'] < selected['zero_vector_loss'],
                                    'real_vector_beats_shuffled': selected['val_loss'] < selected['shuffled_vector_loss'],
                                    'positive_embedding_variance': selected['embedding_variance'] > 0} if selected else None})
            else:
                missing.append(prefix + '/encoder-metrics.json')
            shared = read(prefix + '/shared-stage-0/cost.json')
            if shared:
                bridge_costs.append({'seed': seed, 'size': size, 'arm': 'shared', 'stage': 0,
                                     'seconds': shared.get('seconds'), 'pairs_per_arm': shared.get('pairs_per_arm')})
            condition_budgets = {}
            for arm in arms:
                relative = prefix + '/' + arm
                evaluation = read(relative + '/evaluation.json')
                if not evaluation:
                    missing.append(relative + '/evaluation.json')
                else:
                    for split in ('test', 'harder'):
                        if split in evaluation:
                            entry = evaluation[split]
                            rows.append({'seed': seed, 'size': size, 'arm': arm, 'split': split,
                                **_metrics(entry['overall']), 'by_family': entry.get('by_family', {}),
                                'by_depth': entry.get('by_depth', {}),
                                'checkpoint_sha256': entry.get('checkpoint_sha256'),
                                'evaluation_input_sha256': entry.get('input_sha256'),
                                'checkpoint': str(root / relative / f'stage-{config.get("stages", 1)-1}/recipient.pt')})
                        else:
                            missing.append(relative + '/evaluation.json:' + split)
                arm_stages = []
                for stage in range(config.get('stages', 0)):
                    stage_dir = relative + f'/stage-{stage}'
                    budget = read(stage_dir + '/budget.json')
                    if budget:
                        arm_stages.append(budget)
                    else:
                        missing.append(stage_dir + '/budget.json')
                    geo = read(stage_dir + '/geometry.json')
                    if geo:
                        geometry.append({'seed': seed, 'size': size, 'arm': arm, 'stage': stage,
                            **{key: geo[key] for key in ('test_metrics', 'round_trip', 'self_reconstruction', 'empty_routing') if key in geo},
                            'bridge_training': geo.get('protocol', {}).get('training', {}),
                            'geometry_test_count': geo.get('test', {}).get('count')})
                    else:
                        missing.append(stage_dir + '/geometry.json')
                    cost = read(stage_dir + '/cost.json')
                    if cost:
                        bridge_costs.append({'seed': seed, 'size': size, 'arm': arm, 'stage': stage,
                                             'seconds': cost.get('seconds'), 'pairs_per_arm': cost.get('pairs_per_arm')})
                condition_budgets[arm] = arm_stages
            complete = all(len(condition_budgets[a]) == config.get('stages', 0) for a in arms)
            totals = {a: {k: sum(s.get(k, 0) for s in stages) for k in BUDGET_KEYS}
                      for a, stages in condition_budgets.items()}
            matched = all(len({totals[a][k] for a in arms}) == 1 for k in BUDGET_KEYS) if complete else None
            orders_match = (all(len({condition_budgets[a][s]['batch_order_sha256'] for a in arms}) == 1
                                for s in range(config.get('stages', 0))) if complete else None)
            budgets.append({'seed': seed, 'size': size, 'complete': complete, 'matched': matched,
                'batch_orders_match': orders_match, 'totals_by_arm': totals,
                'recipient_seconds_by_arm': {a: sum(s.get('seconds_this_session', 0) for s in stages)
                                           for a, stages in condition_budgets.items()},
                'unique_problems_per_stage': {a: [s.get('unique_problems_consumed') for s in stages]
                                             for a, stages in condition_budgets.items()}})
    reference = {(r['seed'], r['size'], r['split']): r for r in rows if r['arm'] == 'text_only'}
    for row in rows:
        ref = reference.get((row['seed'], row['size'], row['split']))
        row['delta_vs_text_pp'] = 100 * (row['accuracy'] - ref['accuracy']) if ref else None
    aggregates = []
    for split in ('test', 'harder'):
        for size in sizes:
            for arm in arms:
                selected = [r for r in rows if (r['split'], r['size'], r['arm']) == (split, size, arm)]
                if selected:
                    accuracy = [r['accuracy'] for r in selected]
                    deltas = [r['delta_vs_text_pp'] for r in selected if r['delta_vs_text_pp'] is not None]
                    aggregates.append({'split': split, 'size': size, 'arm': arm, 'seeds': len(selected),
                        'correct': sum(r['correct'] for r in selected), 'total': sum(r['total'] for r in selected),
                        'accuracy_mean': _mean(accuracy), 'accuracy_sample_sd': _sd(accuracy),
                        'generation_accuracy_mean': _mean([r['generation_accuracy'] for r in selected]),
                        'answer_loss_mean': _mean([r['answer_loss'] for r in selected]),
                        'paired_seed_delta_vs_text_pp_mean': _mean(deltas),
                        'paired_seed_delta_vs_text_pp_sample_sd': _sd(deltas) if deltas else None})
    condition_splits = {}
    for row in rows:
        if row['arm'] in arms:
            condition_splits.setdefault((row['seed'], row['size'], row['arm']), set()).add(row['split'])
    complete_conditions = sum(value == {'test', 'harder'} for value in condition_splits.values())
    expected = len(seeds) * len(sizes) * len(arms)
    status = 'completed' if expected and complete_conditions == expected and not missing and all(
        b['matched'] and b['batch_orders_match'] for b in budgets) else 'incomplete'
    last_event = progress.get('events', [])[-1] if progress.get('events') else None
    verification = read('independent-verification.json', read('verification.json', read('verify.json')))
    example = None
    dataset = root / 'dataset.json'
    if dataset.exists():
        data = _json(dataset)
        first = next((r for r in data.get('rows', []) if r.get('split') == 'train'), None)
        if first:
            example = {k: first.get(k) for k in ('id', 'question', 'answer', 'canonical_proof', 'solution')}
    summary = {'schema': 'lemmaferry.scratch.summary.v1', 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': status, 'run_directory': str(root), 'scope': 'Controlled constructive deduction by a small language model trained locally from random initialization. No broad-reasoning claim.',
        'interface': 'Translated vectors are auxiliary training targets; recipient-only question evaluation uses no proof, encoder or translator at inference.',
        'expected_conditions': expected, 'completed_conditions': complete_conditions,
        'config': config, 'plan_sha256': receipts.get('plan.json'), 'pilot': pilot, 'development': development,
        'ordinary_corpus': plan.get('ordinary_corpus'), 'pretraining': pretraining,
        'evaluations': rows, 'aggregates': aggregates, 'encoder_diagnostics': encoders,
        'geometry': geometry, 'recipient_budget_checks': budgets, 'bridge_costs': bridge_costs,
        'cost_note': 'Recipient update/token budgets are matched; bridge/proof-encoder costs are reported separately and are not a claim of equal end-to-end compute. Session timings may exclude earlier resumed sessions.',
        'last_progress_event': last_event, 'missing_artifacts': missing, 'independent_verification': verification,
        'example': example, 'input_sha256': receipts,
        'interpretation': 'Descriptive seed variability on fixed held-out structures; ties and negative deltas are retained. Low bridge error does not establish useful transfer.'}
    for row in rows:
        condition = next((b for b in budgets if (b['seed'], b['size']) == (row['seed'], row['size'])), None)
        if condition is not None:
            row.update(condition['totals_by_arm'].get(row['arm'], {}))
            row['recipient_seconds'] = condition['recipient_seconds_by_arm'].get(row['arm'])
    columns = ['seed', 'size', 'arm', 'split', 'correct', 'total', 'accuracy', 'generation_correct',
               'generation_accuracy', 'answer_loss', 'delta_vs_text_pp', 'updates', 'problem_exposures',
               'supervised_tokens', 'padded_tokens', 'recipient_seconds', 'checkpoint', 'checkpoint_sha256']
    csv_text = io.StringIO(newline='')
    writer = csv.DictWriter(csv_text, fieldnames=columns, extrasaction='ignore')
    writer.writeheader(); writer.writerows(rows)
    document = _html(summary)
    outputs = {'report.html': document, 'summary.json': json.dumps(summary, indent=2, allow_nan=False) + '\n',
               'results.csv': csv_text.getvalue()}
    for name, value in outputs.items():
        _write(root / name, value)
    if destination is not None:
        destination = Path(destination)
        if destination.suffix.lower() == '.html':
            _write(destination, document)
            for name in ('summary.json', 'results.csv'):
                _write(destination.parent / name, outputs[name])
        else:
            for name, value in outputs.items():
                _write(destination / name, value)
    return summary


def _html(summary):
    escape = html.escape
    evaluations = summary['evaluations']
    sections = [f'<h1>From-scratch verified reasoning study</h1><p class="status">{escape(summary["status"].upper())} · '
                f'{summary["completed_conditions"]}/{summary["expected_conditions"]} recipient conditions evaluated</p>',
                '<p>' + escape(summary['scope']) + ' ' + escape(summary['interface']) + '</p>',
                '<p class="muted">Generated ' + escape(summary['generated_at_utc']) + '. Run: <code>' + escape(summary['run_directory']) + '</code></p>']
    if summary['status'] != 'completed':
        sections.append('<p class="notice">This study is incomplete. Missing conditions are not zero scores or negative results. Resume the registered run before interpreting a scaling curve.</p>')
    sections.append('<h2>Accuracy versus distinct verified problems</h2>')
    for split, title in (('test', 'Ordinary held-out structures'), ('harder', 'Harder proof structures')):
        sections += ['<h3>' + title + '</h3>', _curve(evaluations, split, title=title)]
    sections += ['<h2>Aggregate answer results</h2>', _table(
        ['Split', 'Proofs', 'Arm', 'Seeds', 'Correct / total', 'Mean accuracy %', 'Seed SD pp', 'Free-generation %', 'Answer-token NLL', 'Δ vs text pp'],
        [[r['split'], r['size'], LABELS.get(r['arm'], r['arm']), r['seeds'], f"{r['correct']}/{r['total']}",
          100*r['accuracy_mean'], 100*r['accuracy_sample_sd'], 100*r['generation_accuracy_mean'],
          r['answer_loss_mean'], r['paired_seed_delta_vs_text_pp_mean']] for r in summary['aggregates']]),
        '<p>Likelihood chooses A/B from the question plus a fixed space, without the answer. Free generation starts from the question and requires exact extracted A/B. NLL includes the separating space, answer byte and EOS; it is reported separately from accuracy.</p>',
        '<details><summary>Every seed, size, arm and evaluation split</summary>', _table(
        ['Seed', 'Proofs', 'Arm', 'Split', 'Correct / total', 'Accuracy %', 'Generated correct / total', 'NLL', 'Δ vs text pp'],
        [[r['seed'], r['size'], LABELS.get(r['arm'], r['arm']), r['split'], f"{r['correct']}/{r['total']}",
          100*r['accuracy'], f"{r['generation_correct']}/{r['total']}", r['answer_loss'], r['delta_vs_text_pp']] for r in evaluations]), '</details>']
    sections += ['<h2>Proof-vector bottleneck diagnostics</h2>',
        '<p>The proof decoder receives only a 64-dimensional vector and positions. These validation diagnostics test reconstruction and sensitivity to vectors; they do not prove semantic sufficiency.</p>', _table(
        ['Seed', 'Proofs', 'Selected update', 'Real-vector NLL', 'Shuffled-vector NLL', 'Zero-vector NLL', 'Token accuracy %', 'Embedding variance'],
        [[r['seed'], r['size'], r['selected_step'], r['selected'].get('val_loss'), r['selected'].get('shuffled_vector_loss'),
          r['selected'].get('zero_vector_loss'), 100*r['selected'].get('val_token_accuracy', 0), r['selected'].get('embedding_variance')]
         for r in summary['encoder_diagnostics'] if r['selected']])]
    geometry_rows, geometry_curve = [], []
    for row in summary['geometry']:
        for direction, metrics in row.get('test_metrics', {}).items():
            paired = metrics.get('paired', {})
            geometry_rows.append([row['seed'], row['size'], LABELS.get(row['arm'], row['arm']), row['stage'], direction,
                paired.get('normalized_rmse'), metrics.get('shuffled_training_pairs', {}).get('normalized_rmse'),
                metrics.get('training_mean', {}).get('normalized_rmse'), paired.get('cosine_retrieval_top1')])
            if direction == 'math_to_language' and row['stage'] == summary['config'].get('stages', 1)-1:
                geometry_curve.append({'seed': row['seed'], 'size': row['size'], 'arm': row['arm'], 'split': 'geometry',
                                       'normalized_rmse': paired.get('normalized_rmse')})
    sections += ['<h2>Bridge geometry, measured separately</h2>', _curve(geometry_curve, 'geometry', key='normalized_rmse',
        title='Held-out bridge normalized error', ylabel='Normalized RMSE', percent=False),
        '<p>Each curve describes the paired map fitted against that recipient arm’s current snapshot. The table also shows shuffled and mean geometric controls; these are not answer-accuracy curves.</p>',
        '<details><summary>Every bridge direction and stage</summary>', _table(
        ['Seed', 'Proofs', 'Recipient arm', 'Stage', 'Direction', 'Paired nRMSE', 'Shuffled nRMSE', 'Mean nRMSE', 'Retrieval top-1'], geometry_rows), '</details>']
    sections += ['<h2>Budgets and costs</h2>', '<p>' + escape(summary['cost_note']) + '</p>', _table(
        ['Seed', 'Proofs', 'Complete', 'Recipient budgets matched', 'Batch order matched', 'Updates / arm', 'Padded tokens / arm', 'Supervised tokens / arm'],
        [[b['seed'], b['size'], b['complete'], b['matched'], b['batch_orders_match'],
          next(iter(b['totals_by_arm'].values()), {}).get('updates'),
          next(iter(b['totals_by_arm'].values()), {}).get('padded_tokens'),
          next(iter(b['totals_by_arm'].values()), {}).get('supervised_tokens')] for b in summary['recipient_budget_checks']]),
        _table(['Seed', 'Proofs', 'Arm', 'Stage', 'Bridge seconds', 'Pair exposures / neural arm'],
        [[r['seed'], r['size'], r['arm'], r['stage'], r['seconds'], r['pairs_per_arm']] for r in summary['bridge_costs']]),
        '<h2>Ordinary-text pretraining and development</h2>', _table(
        ['Seed', 'Pretraining updates', 'Parameters', 'Initial validation NLL', 'Final validation NLL'],
        [[r['seed'], r.get('updates'), r.get('parameters'), r.get('initial_validation_loss'), r.get('final_validation_loss')]
         for r in summary['pretraining']]),
        '<details><summary>Registered configuration, pilot and disjoint development evidence</summary><pre>' + escape(json.dumps(
        {k: summary[k] for k in ('config', 'pilot', 'development', 'ordinary_corpus')}, indent=2)) + '</pre></details>']
    if summary.get('example'):
        sections += ['<h2>One proof through the pipeline</h2>', '<pre>' + escape(json.dumps(summary['example'], indent=2)) + '</pre>',
            '<p>Canonical proof bytes → frozen proof-encoder vector x → fitted T(x). A frozen recipient reads the complete solution for target y; training applies T(x) to the question state. The final evaluator reloads only the recipient and receives the question.</p>']
    sections += ['<h2>Verification and continuation</h2>', '<pre>' + escape(json.dumps(summary.get('independent_verification'), indent=2)) + '</pre>',
        '<p>Latest progress event:</p><pre>' + escape(json.dumps(summary['last_progress_event'], indent=2)) + '</pre>',
        '<details><summary>Missing artifacts and input hashes</summary><pre>' + escape(json.dumps(
        {'missing': summary['missing_artifacts'], 'sha256': summary['input_sha256']}, indent=2)) + '</pre></details>',
        '<p class="muted">Results describe this controlled deduction distribution. More generated proofs, low vector error, or a favorable seed do not establish broad reasoning improvements. Increasing-data comparisons restart from the same pretrained state; later extensions are a separate cumulative experiment.</p>']
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>LemmaFerry: from-scratch study</title><style>body{max-width:1150px;margin:32px auto;padding:0 20px;font:16px/1.55 system-ui,sans-serif;color:#203040;background:#f7f9fb}'
        'h1,h2,h3{line-height:1.2}h2{margin-top:36px}.status{font-weight:700}.muted,figcaption{color:#586778;font-size:13px}.notice{padding:14px;background:#fff1ce;border-left:4px solid #bd8b26}'
        'figure{margin:18px 0;background:white;border:1px solid #dce3e9;padding:12px;border-radius:8px}svg{width:100%;height:auto}svg text{font:12px system-ui;fill:#4b5968}figcaption span{white-space:nowrap;margin-right:15px}'
        '.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{text-align:left;border-bottom:1px solid #dce3e9;padding:8px;white-space:nowrap}th{background:#eaf0f5;position:sticky;top:0}'
        'details{margin:20px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:14px;border:1px solid #dce3e9;font:12px/1.55 ui-monospace,monospace}code{overflow-wrap:anywhere}</style><main>'
        + ''.join(sections) + '</main></html>')
