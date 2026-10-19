"""Opt-in local experiment command line: python -m lemmaferry.scratch --help."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import urllib.request
import torch
from .io import digest, read, write
from . import model as net
from . import experiment as exp


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='fresh local artifact directory (not committed)')
    p.add_argument('--config', type=Path)
    p.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--budget-hours', type=float, help='explicit continuation cap; original plan remains frozen')
    sub = p.add_subparsers(dest='command', required=True)
    for cmd in ('preflight', 'generate', 'pretrain', 'study', 'evaluate', 'report'):
        sub.add_parser(cmd)
    s = sub.add_parser('sample'); s.add_argument('--seed', type=int, default=7)
    s.add_argument('--checkpoint', type=Path); s.add_argument('--prompt', default='Alice ')
    s.add_argument('--tokens', type=int, default=64)
    s = sub.add_parser('verify'); s.add_argument('--vector-samples', type=int, default=2)
    s.add_argument('--evaluation-samples', type=int, default=8)
    s.add_argument('--skip-lean-replay', action='store_true', help='check saved Lean receipts without recompiling; reported explicitly')
    s = sub.add_parser('extend'); s.add_argument('--additional', type=int, default=256)
    s.add_argument('--seed', type=int, default=7); s.add_argument('--size', type=int, default=4096)
    s.add_argument('--arm', choices=exp.ARMS, default='swap_paired'); s.add_argument('--updates', type=int, default=128)
    return p


def ensure_corpus(root):
    provenance = read(Path(__file__).parents[3] / 'configs/from-scratch-corpus.json')
    target = root / 'corpus/alice.txt'
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(provenance['source'], timeout=60) as response:
            raw = response.read()
        if digest(raw) != provenance['sha256']:
            raise ValueError('public corpus bytes differ from pinned edition; no experiment started')
        target.write_bytes(raw)
    if digest(target.read_bytes()) != provenance['sha256']: raise ValueError('corpus integrity mismatch')


def main(argv=None):
    args = parser().parse_args(argv)
    if args.threads < 1: raise ValueError('threads must be positive')
    torch.set_num_threads(args.threads)
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'mps' and not torch.backends.mps.is_available(): raise ValueError('MPS unavailable')
    root = args.run.expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    default = Path(__file__).parents[3] / 'configs/from-scratch.json'
    if args.config:
        config = read(args.config)
    elif (root / 'plan.json').exists():
        config = read(root / 'plan.json')['config']
    elif (root / 'selected-config.json').exists():
        config = read(root / 'selected-config.json')
    else: config = read(default)
    exp.validate_config(config)
    command = args.command
    experimental = command in ('preflight', 'generate', 'pretrain', 'study', 'extend', 'evaluate')
    def run():
        if command == 'preflight':
            if (root / 'plan.json').exists(): raise ValueError('cannot retune a frozen study; use a fresh run')
            ensure_corpus(root)
            registered = read(Path(__file__).parents[3] / 'configs/from-scratch-preregistration.json')
            write(root / 'pilot-registration.json', registered)
            c = exp.preflight(root, config, device)
            exp.develop(root, c, device)
        elif command == 'generate':
            reserved = read(root / 'development.json')['reserved_ids'] if (root / 'development.json').exists() else []
            data = exp.generate(root, train=max(config['sizes']), validation=config['validation'], test=config['test'],
                                harder=config['harder'], context=config['model']['context'], reserved_ids=reserved)
            exp.freeze_plan(root, config)
            exp.progress(root, 'generation_complete', fingerprint=data['fingerprint'], count=len(data['rows']))
        elif command == 'pretrain':
            ensure_corpus(root); exp.freeze_plan(root, config); exp.pretrain(root, config, device)
        elif command == 'study':
            ensure_corpus(root); exp.study(root, config, device)
            from .report import report
            report(root)
        elif command == 'extend':
            entry = exp.extend(root, config, device, additional=args.additional, seed=args.seed, size=args.size, arm=args.arm, updates=args.updates)
            print(entry, flush=True)
        elif command == 'evaluate':
            dataset = read(root / 'dataset.json')
            for seed in config['seeds']:
                for size in config['sizes']:
                    for arm in exp.ARMS:
                        directory = root / 'study' / f'{seed}-{size}' / arm
                        checkpoint = directory / f'stage-{config["stages"]-1}/recipient.pt'
                        scores = {s: exp.evaluate_recipient(checkpoint, exp.public_questions([r for r in dataset['rows'] if r['split'] == s]), device)
                                  for s in ('test', 'harder')}
                        write(directory / 'evaluation.json', scores); exp.tick()
        elif command == 'verify':
            from .verify import verify_study
            result = verify_study(root, device=device, replay_lean=not args.skip_lean_replay,
                                  vector_samples=args.vector_samples, evaluation_samples=args.evaluation_samples)
            write(root / 'independent-verification.json', result)
            print(result, flush=True)
        elif command == 'report':
            from .report import report
            result = report(root)
            print(f'Report: {root / "report.html"}', flush=True)
        elif command == 'sample':
            checkpoint = args.checkpoint or root / 'starting' / f'{args.seed}.pt'
            m, opt, state = net.restore(checkpoint, device)
            text = net.generate(m, args.prompt, device, args.tokens)
            print(args.prompt + text)
            write(root / 'sample.json', {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint.read_bytes()),
                  'prompt': args.prompt, 'continuation': text, 'tokens_requested': args.tokens, 'updates': state['updates']})
    try:
        if experimental:
            with exp.ResourceBudget(root, args.budget_hours or config['cap_hours']): run()
        else: run()
    except TimeoutError as error:
        exp.progress(root, 'incomplete_resource_cap', reason=str(error))
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
