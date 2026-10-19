#!/usr/bin/env python3
"""Export measured scratch-study results to standalone scientific SVG/PNG plots.

Reads only the report's summary.json. No checkpoint, model, training, dataset
selection, or network operation is performed. Missing conditions remain missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ARMS = ('text_only', 'swap_paired', 'swap_shuffled', 'swap_mean')
LABELS = {'text_only': 'Same-proof text', 'swap_paired': 'Paired translation',
          'swap_shuffled': 'Shuffled translation', 'swap_mean': 'Mean target'}
COLORS = {'text_only': '#a25519', 'swap_paired': '#087e96', 'swap_shuffled': '#8054a5', 'swap_mean': '#75828c'}
MARKERS = {'text_only': 's', 'swap_paired': 'o', 'swap_shuffled': '^', 'swap_mean': 'D'}
CONTROLS = {'paired': ('Paired bridge', '#087e96', 'o'),
            'shuffled_training_pairs': ('Shuffled bridge', '#8054a5', '^'),
            'training_mean': ('Training mean', '#75828c', 'D')}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def aggregate(rows, value):
    """Sample SD across distinct seeds; a single seed has no estimable SD."""
    seeds = [row['seed'] for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError('duplicate seed in plotted condition')
    values = [float(row[value]) for row in rows]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('empty or nonfinite plot data')
    return {'seed_ids': seeds, 'seed_values': values, 'seeds': len(values),
            'mean': statistics.mean(values),
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None}


def _axes(axes, sizes, *, ylabel):
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', color='#e3e7eb', linewidth=.7)
        axis.set_axisbelow(True)
        if sizes:
            axis.set_xscale('log', base=2)
            axis.set_xticks(sizes, [f'{size:,}' for size in sizes])
            axis.set_xlim(min(sizes) / 1.25, max(sizes) * 1.25)
        axis.set_xlabel('Distinct verified training problems')
        axis.set_ylabel(ylabel)


def _line(axis, points, *, label, color, marker, percent=False):
    ordered = sorted(points, key=lambda row: row['size'])
    scale = 100 if percent else 1
    xs = [r['size'] for r in ordered]
    ys = [r['mean'] * scale for r in ordered]
    errors = [(r['sample_sd'] or 0.) * scale for r in ordered]
    axis.errorbar(xs, ys, yerr=errors, label=label, color=color, marker=marker,
                  linewidth=1.6, markersize=5, capsize=3)
    for row in ordered:
        axis.scatter([row['size']] * row['seeds'], [v * scale for v in row['seed_values']],
                     color=color, alpha=.35, s=12, zorder=3)


def _save(fig, output, name, records):
    files = {}
    for suffix in ('svg', 'png'):
        path = output / f'{name}.{suffix}'
        options = {'metadata': {'Date': None}} if suffix == 'svg' else {'dpi': 180}
        fig.savefig(path, **options)
        if path.suffix == '.svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
        files[path.name] = sha256(path)
    return {'files_sha256': files, 'plotted_data': records}


def accuracy_plot(summary, output, plt):
    rows = summary.get('evaluations', [])
    observed = [row for row in rows if row.get('arm') in ARMS and row.get('split') in ('test', 'harder')]
    if not observed:
        return None
    sizes = sorted({row['size'] for row in observed})
    records = {'conditions': [], 'pretraining_baselines': [],
               'chance_reference': {'value': .5, 'meaning': 'theoretical balanced two-choice A/B accuracy; not an observed model score'}}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), layout='constrained', sharey=True)
    _axes(axes, sizes, ylabel='Question-only answer accuracy (%)')
    for axis, split, title in zip(axes, ('test', 'harder'), ('Ordinary held-out structures', 'Harder theorem structures')):
        axis.set_title(title)
        axis.set_ylim(0, 100)
        axis.axhline(50, color='#aeb6bf', linestyle=':', linewidth=1., label='Balanced A/B chance')
        for arm in ARMS:
            points = []
            for size in sizes:
                selected = [r for r in observed if (r['split'], r['size'], r['arm']) == (split, size, arm)]
                if not selected:
                    continue
                for row in selected:
                    if (type(row.get('total')) is not int or row['total'] <= 0
                            or type(row.get('correct')) is not int or not 0 <= row['correct'] <= row['total']
                            or not numeric(row.get('accuracy'))
                            or abs(row['accuracy'] - row['correct'] / row['total']) > 1e-9):
                        raise ValueError('answer accuracy does not match exact counts in summary')
                point = {'split': split, 'size': size, 'arm': arm, **aggregate(selected, 'accuracy'),
                         'correct': sum(r['correct'] for r in selected), 'total': sum(r['total'] for r in selected),
                         'per_seed_counts': [{'seed': r['seed'], 'correct': r['correct'], 'total': r['total']} for r in selected]}
                points.append(point)
                records['conditions'].append(point)
            if points:
                _line(axis, points, label=LABELS[arm], color=COLORS[arm], marker=MARKERS[arm], percent=True)
        baseline_rows = [r for r in rows if r.get('arm') == 'pretrained_baseline' and r.get('split') == split
                         and numeric(r.get('accuracy'))]
        if baseline_rows:
            baseline = {'split': split, **aggregate(baseline_rows, 'accuracy'),
                        'per_seed_counts': [{'seed': r['seed'], 'correct': r['correct'], 'total': r['total']} for r in baseline_rows]}
            records['pretraining_baselines'].append(baseline)
            mean, sd = 100 * baseline['mean'], 100 * (baseline['sample_sd'] or 0)
            axis.axhline(mean, color='#252c34', linestyle='--', linewidth=1.,
                         label=f'Actual pretraining baseline (n={baseline["seeds"]})')
            if baseline['sample_sd'] is not None:
                axis.axhspan(max(0, mean - sd), min(100, mean + sd), color='#252c34', alpha=.07)
        axis.legend(fontsize=7.5, loc='best', frameon=False)
    fig.suptitle('Controlled deduction: recipient trained from random initialization', fontsize=12)
    fig.supxlabel('Means and ± one sample SD across observed seeds; dots are seeds. A single seed has no SD estimate.\n'
                  f'Run status: {summary.get("status", "unknown")}. Likelihood uses question plus a fixed space; no proof or bridge at inference.', fontsize=8)
    result = _save(fig, output, 'scratch-accuracy', records)
    plt.close(fig)
    return result


def geometry_plot(summary, output, plt):
    # Each learned recipient defines its own representation domain. Hold the
    # producer arm fixed instead of combining incomparable target spaces.
    final_stage = summary.get('config', {}).get('stages', 0) - 1
    rows = [r for r in summary.get('geometry', []) if r.get('arm') == 'text_only' and r.get('stage') == final_stage]
    if not rows:
        return None
    sizes = sorted({r['size'] for r in rows})
    points = []
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), layout='constrained')
    _axes(axes, sizes, ylabel='Held-out normalized RMSE')
    available = False
    for axis, direction, title in zip(axes, ('math_to_language', 'language_to_math'),
                                      ('Proof vector → recipient state', 'Recipient state → proof vector')):
        axis.set_title(title)
        for control, (label, color, marker) in CONTROLS.items():
            values = []
            for row in rows:
                metric = row.get('test_metrics', {}).get(direction, {}).get(control, {}).get('normalized_rmse')
                if numeric(metric):
                    values.append({'seed': row['seed'], 'size': row['size'], 'nrmse': metric,
                                   'geometry_test_count': row.get('geometry_test_count')})
            series = []
            for size in sizes:
                selected = [r for r in values if r['size'] == size]
                if selected:
                    point = {'size': size, 'direction': direction, 'control': control,
                             **aggregate(selected, 'nrmse'),
                             'held_out_counts': [r['geometry_test_count'] for r in selected]}
                    series.append(point); points.append(point)
            if series:
                available = True
                _line(axis, series, label=label, color=color, marker=marker)
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=8, frameon=False)
    if not available:
        plt.close(fig)
        return None
    fig.suptitle('Bridge geometry, separate from answer performance', fontsize=12)
    fig.supxlabel(f'Only stage {final_stage} bridges fitted against the same-proof-text recipient’s pre-update state.\n'
                  'Paired/shuffled/mean controls share that producer. Whiskers: sample SD across seeds; lower error is geometric alignment.', fontsize=8)
    result = _save(fig, output, 'scratch-geometry', {'producer_arm': 'text_only', 'stage': final_stage, 'conditions': points})
    plt.close(fig)
    return result


def encoder_plot(summary, output, plt):
    rows = [r for r in summary.get('encoder_diagnostics', []) if r.get('selected')]
    if not rows:
        return None
    sizes = sorted({r['size'] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), layout='constrained')
    _axes(axes, sizes, ylabel='Selected validation byte NLL')
    axes[0].set_title('Proof reconstruction through the vector bottleneck')
    axes[1].set_title('Proof-vector variance across validation examples')
    axes[1].set_ylabel('Mean coordinate variance')
    records = []
    definitions = (('val_loss', 'Real proof vector', '#087e96', 'o'),
                   ('shuffled_vector_loss', 'Shuffled vector', '#8054a5', '^'),
                   ('zero_vector_loss', 'Zero vector', '#75828c', 'D'),
                   ('embedding_variance', 'Embedding variance', '#a25519', 's'))
    for key, label, color, marker in definitions:
        points = []
        for size in sizes:
            selected = [{'seed': r['seed'], key: r['selected'][key], 'selected_step': r.get('selected_step')}
                        for r in rows if r['size'] == size and numeric(r['selected'].get(key))]
            if selected:
                point = {'size': size, 'metric': key, **aggregate(selected, key),
                         'selected_steps': [r['selected_step'] for r in selected]}
                points.append(point); records.append(point)
        if points:
            _line(axes[1] if key == 'embedding_variance' else axes[0], points,
                  label=label, color=color, marker=marker)
    for axis in axes:
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=8, frameon=False)
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    fig.suptitle('Independent proof autoencoder: selected validation checkpoint', fontsize=12)
    fig.supxlabel('Decoder receives only the frozen 64D vector and positions. Reconstruction/variance do not establish semantic sufficiency.\n'
                  'Means and sample SD across observed seeds; missing conditions are omitted.', fontsize=8)
    result = _save(fig, output, 'scratch-proof-encoder', {'conditions': records})
    plt.close(fig)
    return result


def plot_results(run, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.hashsalt': 'lemmaferry-scratch-scientific-results-v1',
                                'svg.fonttype': 'path', 'font.size': 9})
    run, output = Path(run).resolve(), Path(output)
    source = run / 'summary.json'
    source_bytes = source.read_bytes()
    summary = json.loads(source_bytes.decode('utf-8'))
    if summary.get('schema') != 'lemmaferry.scratch.summary.v1':
        raise ValueError('expected report-generated scratch summary.json')
    output.mkdir(parents=True, exist_ok=True)
    figures = {}
    for name, function in (('accuracy', accuracy_plot), ('geometry', geometry_plot), ('proof_encoder', encoder_plot)):
        result = function(summary, output, plt)
        if result is not None:
            figures[name] = result
    manifest = {'schema': 'lemmaferry.scratch.plots.v1', 'source_summary_sha256': hashlib.sha256(source_bytes).hexdigest(),
                'source_summary_status': summary.get('status'), 'source_run': str(run),
                'script_sha256': sha256(Path(__file__)), 'matplotlib': matplotlib.__version__,
                'interpretation': 'Descriptive seed means and sample SD; no confidence intervals, test-based selection, or capability conclusion.',
                'figures': figures, 'omitted_figures': [name for name in ('accuracy', 'geometry', 'proof_encoder') if name not in figures]}
    (output / 'plot-manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Run containing report-generated summary.json')
    parser.add_argument('--output', type=Path, default=Path('artifacts/from-scratch/plots'))
    args = parser.parse_args()
    manifest = plot_results(args.run, args.output)
    print(json.dumps({'summary_status': manifest['source_summary_status'],
                      'figures': list(manifest['figures']), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
