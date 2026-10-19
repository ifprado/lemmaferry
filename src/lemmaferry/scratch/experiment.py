"""Staged experiment engine. Frozen plans, sequential arms, atomic resumable state."""
from dataclasses import asdict
import copy
import gc
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import model as net
from .io import digest, state_digest, read, write, save, load
from .logic import generate, validate, admit
from ..rosetta import PairedSplit
from ..swap import SwapPair, SwapConfig

ARMS = ('text_only', 'swap_paired', 'swap_shuffled', 'swap_mean')
_BUDGET = None


class ResourceBudget:
    """Count active experiment commands, excluding installation and coding time."""
    def __init__(self, root, hours):
        self.path = Path(root) / 'experiment-clock.json'
        self.hours = hours
        self.used = read(self.path)['seconds'] if self.path.exists() else 0.
        self.started = time.monotonic()
    def check(self):
        elapsed = self.used + time.monotonic() - self.started
        write(self.path, {'seconds': elapsed, 'cap_hours': self.hours,
                         'policy': 'active experiment commands; coding/install excluded; updated at checkpoints'})
        if elapsed > self.hours * 3600:
            raise TimeoutError('Experiment budget reached; saved checkpoints can be resumed with an explicit larger --budget-hours.')
    def __enter__(self):
        global _BUDGET
        _BUDGET = self
        self.check()
        return self
    def __exit__(self, *args):
        global _BUDGET
        try: self.check()
        finally: _BUDGET = None


def tick():
    if _BUDGET is not None: _BUDGET.check()


def source_hashes():
    package = Path(__file__).parents[1]
    files = sorted((package / 'scratch').rglob('*.py')) + [package / 'swap.py']
    return {str(p.relative_to(package.parents[1])): digest(p.read_bytes()) for p in files}


def validate_config(config):
    net.ModelConfig(**config['model'])
    for key in ('batch', 'pretrain_steps', 'recipient_steps', 'stages', 'encoder_steps', 'bridge_steps', 'checkpoint_every'):
        if type(config[key]) is not int or config[key] <= 0: raise ValueError(f'invalid {key}')
    if config['batch'] % 2 or config['recipient_steps'] % config['stages']:
        raise ValueError('even batch and equal integral stages are required')
    exposures = (config['recipient_steps'] // config['stages']) * (config['batch'] // 2)
    if exposures < max(config['sizes']): raise ValueError('every stage must consume every largest-pool problem')
    if config['encoder_steps'] * config['batch'] < max(config['sizes']): raise ValueError('encoder must consume its full pool')
    if len(set(config['seeds'])) != len(config['seeds']) or len(set(config['sizes'])) != len(config['sizes']):
        raise ValueError('duplicate seeds or sizes')
    if not 0 < config['alpha'] <= 1: raise ValueError('alpha outside registered bounded range')
    return config


def sync(device):
    if device == 'mps': torch.mps.synchronize()


def progress(root, stage, **extra):
    path = Path(root) / 'progress.json'
    state = read(path) if path.exists() else {'events': []}
    memory = {'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if torch.backends.mps.is_available():
        memory.update(mps_current_bytes=torch.mps.current_allocated_memory(), mps_driver_bytes=torch.mps.driver_allocated_memory())
    state['events'].append({'stage': stage, 'time_unix': time.time(), 'memory': memory, **extra})
    state['resume'] = f'.venv-scratch/bin/python -m lemmaferry.scratch --run {root} --device mps study'
    write(path, state)
    print(stage, extra, flush=True)


def environment(device):
    return {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__,
            'machine': platform.machine(), 'system': platform.system(), 'os_release': platform.release(),
            'device': device, 'threads': torch.get_num_threads(), 'mps_high_watermark': os.getenv('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'default'),
            'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def step(model, opt, x, labels, positions=None, targets=None, alpha=0., scale=None):
    model.train(); opt.zero_grad(set_to_none=True)
    logits, h = model(x)
    text = net.loss(logits, labels)
    aux = torch.zeros((), device=x.device)
    if targets is not None:
        # Caller-owned/bridge graphs can never enter recipient optimization.
        target = targets.detach().clone().to(h)
        state = h[torch.arange(len(x), device=x.device), positions]
        aux = (((state - target) / scale.to(h)) ** 2).mean()
    total = text + alpha * aux
    if not torch.isfinite(total): raise ValueError('nonfinite recipient loss')
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step()
    return float(text.detach().cpu()), float(aux.detach().cpu())


def corpus(root):
    path = Path(root) / 'corpus/alice.txt'
    provenance = read(Path(__file__).parents[3] / 'configs/from-scratch-corpus.json')
    raw = path.read_bytes()
    if digest(raw) != provenance['sha256']: raise ValueError('starter corpus changed')
    text = raw.decode('utf-8')
    text = text.split('*** START OF THE PROJECT GUTENBERG EBOOK')[1].split('\n', 1)[1].split('*** END OF THE PROJECT GUTENBERG EBOOK')[0]
    data = text.encode()
    cut = int(len(data) * .9)
    # Byte split fixed before tokenizer/model training, no synthetic input involved.
    return data[:cut], data[cut:]


def ordinary_batch(data, step_index, seed, config, device):
    r = random.Random(seed * 100000 + step_index)
    length = config['model']['context']
    x = torch.tensor([[net.BOS] + list(data[(start := r.randrange(len(data) - length)):start + length - 1])
                      for _ in range(config['batch'])], dtype=torch.long)
    # Ordinary corpus bytes have the same byte offset as all other model text.
    x[:, 1:] += 3
    y = x.clone()
    return x.to(device), y.to(device)


def preflight(root, config, device):
    """Discarded development-only weights. Never reads frozen capability tests."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(811)
    train, val = corpus(root)
    records = []
    choices = [config['model'], {'layers': 3, 'width': 128, 'heads': 4, 'context': 512}]
    for arch in choices:
        c = copy.deepcopy(config); c['model'] = arch
        m = net.LM(net.ModelConfig(**arch)).to(device); opt = net.optimizer(m)
        x, y = ordinary_batch(train, 0, 811, c, device)
        before = float(net.loss(m(x)[0], y).detach().cpu())
        for _ in range(3): step(m, opt, x, y)
        sync(device); start = time.monotonic()
        losses = [step(m, opt, x, y)[0] for _ in range(12)]
        sync(device); seconds = (time.monotonic() - start) / 12
        for _ in range(45): step(m, opt, x, y)
        after = float(net.loss(m(x)[0], y).detach().cpu())
        records.append({'model': arch, 'parameters': sum(p.numel() for p in m.parameters()),
                        'batch': c['batch'], 'seconds_per_update': seconds, 'initial_loss': before,
                        'development_batch_loss_after_60': after, 'loss_reduced': after < before,
                        'estimated_recipient_hours': seconds * c['recipient_steps'] * len(config['seeds']) * len(config['sizes']) * len(ARMS) / 3600})
        del m, opt; gc.collect()
        if device == 'mps': torch.mps.empty_cache()
    # Reserve 35% of the cap for evaluation, representations, bridges and I/O.
    feasible = [r for r in records if r['estimated_recipient_hours'] < config['cap_hours'] * .65]
    if not feasible: raise ValueError('no architecture fits cap; use a documented smaller config')
    selected = feasible[0]
    final = copy.deepcopy(config); final['model'] = selected['model']
    write(root / 'pilot.json', {'environment': environment(device), 'measurements': records,
                              'selection': selected, 'alpha': final['alpha'],
                              'alpha_policy': 'provisional; final value is selected by the disjoint development pilot',
                              'cap_reserved_fraction': .35, 'discarded_weights': True})
    write(root / 'selected-config.json', final)
    tick()
    return final


def freeze_plan(root, config):
    root = Path(root)
    dataset = read(root / 'dataset.json')
    plan = {'schema': 1, 'config': config, 'dataset_sha256': dataset['fingerprint'],
            'split_ids': {s: [r['id'] for r in dataset['rows'] if r['split'] == s] for s in ['train', 'validation', 'test', 'harder']},
            'arms': list(ARMS), 'nested_pools': True, 'starting_commit': '2c7c1c580f3e46ca789c56562d1678627771593e',
            'metrics': ['question_only_letter_likelihood_accuracy', 'answer_nll', 'greedy_exact_answer_accuracy', 'bidirectional_normalized_rmse'],
            'selection': 'architecture: throughput pilot; alpha: disjoint development pilot; encoder and bridge: validation; recipient: full fixed budget',
            'primary_budget': 'identical update count, fixed padded token count, 64 proof tokens and 3 QA tokens per problem exposure',
            'scope': 'controlled constructive deduction; no claim of broad intelligence',
            'sequential': True, 'network_required': False,
            'boundary': 'final layer-normalized state before head; last UTF-8 byte of solution rendering for bridge y; last question byte for recipient h(q)',
            'auxiliary_scaling': 'per-coordinate training language std from each frozen bridge; training only, detached targets',
            'ordinary_corpus': read(Path(__file__).parents[3] / 'configs/from-scratch-corpus.json'),
            'source_sha256': source_hashes(), 'development': read(root / 'development.json') if (root / 'development.json').exists() else None,
            'acceptance': {'pipeline': 'all admitted proofs have axiom-free real Lean receipts; train-only fitting; matched recipient budgets; independent reload verification',
                           'research': 'report paired minus text-only held-out accuracy for every size and seed, including ties/declines; positive transfer not required'},
            'encoder': {'objective': 'independent proof-byte reconstruction through a 64D vector only', 'selection': 'minimum validation CE', 'ablations': ['zero_vector', 'shuffled_vector']},
            'bridge': {'backend': 'swap', 'latent': 32, 'hidden': 128, 'paired': 1., 'self': .25, 'cycle': .10, 'latent_loss': .01,
                       'validation_every': 64, 'batch': 32, 'learning_rate': .002, 'device': 'cpu'},
            'evaluation_boundary': 'question-only recipient reload, fixed A/B space prefix for likelihood, unconstrained greedy continuation; no bridge or proof at inference'}
    path = root / 'plan.json'
    if path.exists():
        if read(path) != plan: raise ValueError('frozen study plan changed')
    else: write(path, plan)
    return plan


def pretrain(root, config, device):
    root = Path(root); train, val = corpus(root)
    for seed in config['seeds']:
        path = root / 'starting' / f'{seed}.pt'
        torch.manual_seed(seed)
        if path.exists():
            m, opt, state = net.restore(path, device); start = state['updates']; trace = state['metadata']['trace']
        else:
            m = net.LM(net.ModelConfig(**config['model'])).to(device); opt = net.optimizer(m, config['learning_rate']); start = 0; trace = []
            xv, yv = ordinary_batch(val, 0, seed, config, device)
            with torch.no_grad(): initial_loss = float(net.loss(m(xv)[0], yv).cpu())
            trace.append({'step': 0, 'validation_loss': initial_loss})
        t = time.monotonic()
        for k in range(start, config['pretrain_steps']):
            x, y = ordinary_batch(train, k, seed, config, device)
            l, _ = step(m, opt, x, y)
            if k % 50 == 0: trace.append({'step': k, 'train_loss': l})
            if (k + 1) % config['checkpoint_every'] == 0 or k + 1 == config['pretrain_steps']:
                xv, yv = ordinary_batch(val, 0, seed, config, device)
                with torch.no_grad(): vl = float(net.loss(m(xv)[0], yv).cpu())
                net.checkpoint(path, m, opt, k + 1, trace=trace, validation_loss=vl,
                               corpus_train_sha256=digest(train), corpus_val_sha256=digest(val), seconds_this_session=time.monotonic() - t)
                progress(root, 'pretrain', seed=seed, updates=k + 1, train_loss=l, validation_loss=vl)
                tick()
        if path.exists():
            saved = load(path)
            write(path.with_name(f'{seed}-pretraining.json'), {'updates': saved['updates'], 'trace': saved['metadata']['trace'],
                  'initial_validation_loss': trace[0].get('validation_loss'), 'final_validation_loss': saved['metadata']['validation_loss'],
                  'model_sha256': state_digest(saved['model']), 'optimizer_sha256': state_digest(saved['optimizer']),
                  'parameters': sum(p.numel() for p in m.parameters()), 'environment': environment(device)})
        del m, opt; gc.collect()


def proof_data(rows, context, device):
    return net.batch([r['canonical_proof'] for r in rows], context, device)[0]


def proof_encoder(root, train, validation, config, seed, device):
    path = Path(root) / 'encoder.pt'
    torch.manual_seed(seed + 1000)
    ae = net.ProofAE(context=config['model']['context']).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=.001)
    start, best, best_state, trace, selected_step = 0, float('inf'), None, [], None
    receipt = {'train_ids_sha256': digest([r['id'] for r in train]),
               'validation_ids_sha256': digest([r['id'] for r in validation]),
               'encoder_train_content_sha256': digest([r['canonical_proof'] for r in train]),
               'encoder_validation_content_sha256': digest([r['canonical_proof'] for r in validation])}
    if path.exists():
        state = load(path, device); ae.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        start, best, best_state, trace = state['step'], state['best'], state['best_state'], state['trace']
        if state.get('receipt') != receipt: raise ValueError('encoder training corpus changed')
        selected_step = state['selected_step']
        if start == config['encoder_steps'] and (Path(root) / 'encoder-metrics.json').exists():
            ae.load_state_dict(best_state); ae.eval().requires_grad_(False)
            return ae
    train_ids = proof_data(train, config['model']['context'], 'cpu')
    val_ids = proof_data(validation, config['model']['context'], device)
    def reconstruction(z, ids):
        logits = ae.decode(z, ids.shape[1]); mask = ids.ne(net.PAD)
        l = F.cross_entropy(logits.reshape(-1, net.VOCAB), ids.reshape(-1), ignore_index=net.PAD)
        acc = (logits.argmax(-1).eq(ids) & mask).sum() / mask.sum()
        return l, acc
    t = time.monotonic()
    order = np.random.default_rng(seed).permutation(len(train))
    for k in range(start, config['encoder_steps']):
        indices = order[(np.arange(config['batch']) + k * config['batch']) % len(train)]
        ids = train_ids[indices].to(device)
        ae.train(); opt.zero_grad(set_to_none=True)
        l, _ = reconstruction(ae.encode(ids), ids); l.backward(); torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.); opt.step()
        if (k + 1) % config['checkpoint_every'] == 0 or k + 1 == config['encoder_steps']:
            ae.eval()
            with torch.no_grad():
                z = ae.encode(val_ids); vl, acc = reconstruction(z, val_ids)
                shuffled, _ = reconstruction(z.roll(1, 0), val_ids); zero, _ = reconstruction(torch.zeros_like(z), val_ids)
                info = {'step': k + 1, 'train_loss': float(l.detach().cpu()), 'val_loss': float(vl.cpu()),
                        'val_token_accuracy': float(acc.cpu()), 'shuffled_vector_loss': float(shuffled.cpu()),
                        'zero_vector_loss': float(zero.cpu()), 'embedding_variance': float(z.var(0).mean().cpu())}
            trace.append(info)
            if info['val_loss'] < best:
                best = info['val_loss']; best_state = {k: v.detach().cpu().clone() for k, v in ae.state_dict().items()}
                selected_step = k + 1
            save(path, {'model': ae.state_dict(), 'optimizer': opt.state_dict(), 'step': k + 1, 'best': best,
                        'best_state': best_state, 'trace': trace, 'schema': 'lemmaferry.scratch.proof-encoder.v1',
                        'selected_step': selected_step, 'receipt': receipt,
                        'selected_state_sha256': state_digest(best_state),
                        'architecture': {'width': 96, 'latent': 64, 'context': config['model']['context']},
                        'seed': seed + 1000, 'training_budget': config['encoder_steps']})
            progress(Path(root).parents[1], 'proof_encoder', seed=seed, size=len(train), **info)
            tick()
    ae.load_state_dict(best_state); ae.eval().requires_grad_(False)
    write(Path(root) / 'encoder-metrics.json', {'trace': trace, 'pair_exposures': config['encoder_steps'] * config['batch'],
             **receipt, 'selected_step': selected_step, 'selected_state_sha256': state_digest(best_state),
             'seconds_this_session': time.monotonic() - t})
    return ae


@torch.no_grad()
def proof_vectors(ae, rows, config, device):
    vectors = []
    for i in range(0, len(rows), 32): vectors.append(ae.encode(proof_data(rows[i:i + 32], config['model']['context'], device)).cpu())
    return torch.cat(vectors).numpy()


def fit_bridge(directory, ae, model, train, validation, config, seed, device):
    directory = Path(directory)
    receipt = {'encoder_selected_state_sha256': state_digest(ae.state_dict()),
               'frozen_recipient_parameter_sha256': state_digest(model.state_dict()),
               'train_ids_sha256': digest([r['id'] for r in train]),
               'validation_ids_sha256': digest([r['id'] for r in validation]),
               'math_boundary': 'selected proof encoder 64D bottleneck; canonical proof bytes',
               'language_boundary': 'final layer norm, final solution byte; no EOS',
               'source_sha256': source_hashes()}
    complete = all((directory / p).exists() for p in ('maps/manifest.json', 'vectors.pt', 'vectors.sha256.json', 'cost.json'))
    if complete:
        cost = read(directory / 'cost.json')
        if cost['receipt'] != receipt: raise ValueError('frozen bridge producers or pairing changed')
        pair, vectors = SwapPair.load(directory / 'maps'), load(directory / 'vectors.pt')
        if state_digest(vectors) != cost['vectors_sha256']: raise ValueError('frozen bridge vectors changed')
        pair.validate_training_data(PairedSplit(vectors['x'].numpy(), vectors['y'].numpy(), [r['id'] for r in train], [r['id'] for r in train], 'train'),
                                    PairedSplit(vectors['vx'].numpy(), vectors['vy'].numpy(), [r['id'] for r in validation], [r['id'] for r in validation], 'val'))
        return pair, vectors
    if directory.exists():
        if (directory / 'recipient.pt').exists(): raise ValueError('recipient exists with an incomplete bridge; evidence needs inspection')
        directory.rename(directory.with_name(f'.{directory.name}.interrupted-{time.time_ns()}'))
    staging = directory.with_name(f'.{directory.name}.staging-{time.time_ns()}')
    t = time.monotonic()
    x = proof_vectors(ae, train, config, device)
    vx = proof_vectors(ae, validation, config, device)
    y = net.represent(model, [r['solution'] for r in train], device).numpy()
    vy = net.represent(model, [r['solution'] for r in validation], device).numpy()
    tr = PairedSplit(x, y, [r['id'] for r in train], [r['id'] for r in train], 'train')
    va = PairedSplit(vx, vy, [r['id'] for r in validation], [r['id'] for r in validation], 'val')
    pair = SwapPair.fit(tr, va, config=SwapConfig(latent_dim=32, hidden_dim=128, steps=config['bridge_steps'],
                         batch_size=32, validation_every=64, device='cpu'), seed=seed)
    pair.save(staging / 'maps')
    vectors = {'x': torch.from_numpy(x), 'y': torch.from_numpy(y), 'vx': torch.from_numpy(vx), 'vy': torch.from_numpy(vy)}
    save(staging / 'vectors.pt', vectors)
    write(staging / 'cost.json', {'seconds': time.monotonic() - t, 'pairs_per_arm': config['bridge_steps'] * 32,
                                   'snapshot_state': 'frozen detached numeric arrays; no snapshot gradients',
                                   'solution_rendering': 'question, correct answer, complete canonical Lean proof term',
                                   'receipt': receipt,
                                   'vectors_sha256': state_digest(vectors),
                                   'environment': environment(device)})
    staging.rename(directory)
    tick()
    return pair, vectors


def training_batch(rows, indices, config, device):
    selected = [rows[i] for i in indices]
    # QA is separate from proof-as-text. All arms receive both, in this order.
    texts, prompts = [], []
    for row in selected:
        texts += [row['question'] + ' ' + row['answer'], 'Proof:\n' + row['canonical_proof']]
        prompts += [row['question'], 'Proof:\n']
    x, y, pos = net.batch(texts, config['model']['context'], device, prompts)
    for i in range(1, len(texts), 2):
        positions = torch.where(y[i] != -100)[0]
        if len(positions) < 64: raise ValueError('proof example too short for matched 64-token budget')
        chosen = positions[torch.linspace(0, len(positions) - 1, 64, device=device).long()]
        y[i].fill_(-100); y[i, chosen] = x[i, chosen]
    return x, y, pos


def batch_schedule(rows, config, seed, stage):
    order = np.random.default_rng(seed + stage).permutation(len(rows)).tolist()
    if config.get('priority_ids'):
        priority = set(config['priority_ids'])
        order = [i for i in order if rows[i]['id'] in priority] + [i for i in order if rows[i]['id'] not in priority]
    per = config['batch'] // 2
    return [[order[(k * per + j) % len(order)] for j in range(per)]
            for k in range(config['recipient_steps'] // config['stages'])]


def train_recipient(directory, model, opt, starting_updates, train, pair, vectors, config, seed, arm, stage, device):
    directory = Path(directory); path = directory / 'recipient.pt'
    nsteps = config['recipient_steps'] // config['stages']
    before = {'parameter_sha256_before': state_digest(model.state_dict()),
              'optimizer_state_sha256_before': state_digest(opt.state_dict())}
    done, trace = 0, []
    if path.exists():
        model, opt, state = net.restore(path, device)
        done, trace = state['metadata']['stage_updates'], state['metadata']['trace']
        before = {key: state['metadata'][key] for key in before}
    control = {'swap_paired': 'paired', 'swap_shuffled': 'shuffled', 'swap_mean': 'mean'}.get(arm)
    targets = torch.tensor(pair.to_language(vectors['x'].numpy(), control=control), dtype=torch.float32) if control else None
    target_hash = state_digest(targets) if targets is not None else None
    scale = torch.tensor(pair.language_scaler.scale, dtype=torch.float32, device=device)
    # An ordered pass per stage guarantees every largest-pool proof is consumed.
    per = config['batch'] // 2
    schedule = batch_schedule(train, config, seed, stage)
    frozen = read(directory.parent.parent / 'shared-stage-0/cost.json')['receipt'] if stage == 0 else read(directory / 'cost.json')['receipt']
    receipt = {**before, 'target_sha256': target_hash,
               'encoder_selected_state_sha256': frozen['encoder_selected_state_sha256'],
               'frozen_recipient_parameter_sha256': frozen['frozen_recipient_parameter_sha256'],
               'train_ids_sha256': digest([r['id'] for r in train]), 'batch_order_sha256': digest(schedule)}
    if before['parameter_sha256_before'] != frozen['frozen_recipient_parameter_sha256']:
        raise ValueError('bridge does not describe pre-update recipient')
    t = time.monotonic()
    for k in range(done, nsteps):
        indices = schedule[k]
        x, y, pos = training_batch(train, indices, config, device)
        # Only question rows get auxiliary targets; proof explanation rows remain text-only.
        opt.zero_grad(set_to_none=True); model.train()
        logits, h = model(x); text = net.loss(logits, y); auxiliary = torch.zeros((), device=device)
        if targets is not None:
            target = targets[indices].detach().clone().to(device)
            states = h[torch.arange(0, len(x), 2, device=device), pos[::2]]
            auxiliary = (((states - target) / scale) ** 2).mean()
        total = text + config['alpha'] * auxiliary
        if not torch.isfinite(total): raise ValueError('nonfinite training loss')
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        if k % 32 == 0: trace.append({'step': k + 1, 'text_loss': float(text.detach().cpu()), 'auxiliary': float(auxiliary.detach().cpu())})
        if (k + 1) % config['checkpoint_every'] == 0 or k + 1 == nsteps:
            net.checkpoint(path, model, opt, starting_updates + k + 1, stage_updates=k + 1, trace=trace,
                           arm=arm, seed=seed, stage=stage, **receipt)
            print('recipient', seed, len(train), arm, stage, k + 1, trace[-1], flush=True)
            tick()
    budget = {'updates': nsteps, 'problem_exposures': nsteps * per, 'supervised_tokens': nsteps * per * 67,
              'padded_tokens': nsteps * config['batch'] * config['model']['context'], 'batch_order_sha256': digest(schedule),
              'unique_problems_consumed': len(set(i for ids in schedule for i in ids)), 'seconds_this_session': time.monotonic() - t,
              **receipt, 'parameter_sha256_after': state_digest(model.state_dict()),
              'optimizer_state_sha256_after': state_digest(opt.state_dict()), 'complete': True}
    write(directory / 'budget.json', budget)
    return model, opt


def develop(root, config, device):
    """Discarded disjoint proof development pilot; choose alpha without main tests."""
    root = Path(root)
    if (root / 'development.json').exists():
        c = copy.deepcopy(config); c['alpha'] = read(root / 'development.json')['selected_alpha']
        return c
    development = root / 'development'
    development.mkdir(parents=True, exist_ok=True)
    cp = development / 'corpus/alice.txt'; cp.parent.mkdir(exist_ok=True)
    if not cp.exists(): cp.write_bytes((root / 'corpus/alice.txt').read_bytes())
    data = generate(development, train=64, validation=16, test=0, harder=0,
                    context=config['model']['context'], seed=9000000)
    rows = data['rows']; train = [r for r in rows if r['split'] == 'train']; val = [r for r in rows if r['split'] == 'validation']
    small = copy.deepcopy(config)
    small.update(seeds=[7], sizes=[64], pretrain_steps=64, recipient_steps=128, stages=2,
                 encoder_steps=256, bridge_steps=256)
    pretrain(development, small, device)
    directory = development / 'study/7-64'
    ae = proof_encoder(directory, train, val, small, 7, device)
    records = []
    for alpha in (.01, .10):
        c = copy.deepcopy(small); c['alpha'] = alpha
        m, opt, state = net.restore(development / 'starting/7.pt', device)
        pair, vectors = fit_bridge(directory / 'shared-stage-0', ae, m, train, val, c, 7, device)
        stage_dir = directory / 'swap_paired' / f'stage-0-alpha-{alpha}'
        m, opt = train_recipient(stage_dir, m, opt, 64, train, pair, vectors, c, 7, 'swap_paired', 0, device)
        del m, opt; gc.collect()
        evaluation = evaluate_recipient(stage_dir / 'recipient.pt', public_questions(val), device)
        records.append({'alpha': alpha, 'validation': evaluation['overall']})
    selected = min(records, key=lambda r: (r['validation']['answer_loss'], r['alpha']))['alpha']
    write(root / 'development.json', {'schema': 1, 'seed': 9000000,
            'reserved_ids': [r['id'] for r in rows], 'selected_alpha': selected,
            'selection_metric': 'development validation answer NLL after fixed 64 recipient updates',
            'candidates': records, 'config': small, 'discarded_weights': True,
            'common_hyperparameter_information': {'train': len(train), 'validation': len(val)},
            'main_test_opened': False})
    result = copy.deepcopy(config); result['alpha'] = selected
    write(root / 'selected-config.json', result)
    pilot = read(root / 'pilot.json')
    pilot.update(alpha=selected, alpha_policy='minimum disjoint development validation answer NLL; see development.json')
    write(root / 'pilot.json', pilot)
    progress(root, 'development_complete', alpha=selected, candidates=records)
    return result


@torch.no_grad()
def evaluate_recipient(checkpoint, questions, device, batch_size=16):
    """Independent reload: this function can only access recipient + questions.

    `questions` has question, answer, family, depth, id; no proof/solution/vector.
    Likelihood selects A/B conditional on the fixed separating space. Generation
    emits an unconstrained space/letter/EOS sequence, exact extraction thereafter.
    """
    m, opt, state = net.restore(checkpoint, device); del opt
    m.eval(); predictions = []
    for start in range(0, len(questions), batch_size):
        rows = questions[start:start + batch_size]
        prompts = [r['question'] for r in rows]
        # Predictions receive question + a fixed space only, never the answer.
        qx, _, qp = net.batch([p + ' ' for p in prompts], m.config.context, device,
                              pad_to_context=False, include_eos=False)
        qlogits, _ = m(qx)
        ll = qlogits[torch.arange(len(qx), device=device), qp][:, [ord('A') + 3, ord('B') + 3]]
        chosen = ll.argmax(-1).cpu().tolist()
        # Teacher forcing is restricted to the separate NLL measurement.
        x, y, pos = net.batch([p + ' ' + r['answer'] for p, r in zip(prompts, rows)], m.config.context, device, prompts, pad_to_context=False)
        logits, _ = m(x)
        token_losses = F.cross_entropy(logits[:, :-1].transpose(1, 2), y[:, 1:], ignore_index=-100, reduction='none')
        losses = token_losses.sum(1) / (y[:, 1:] != -100).sum(1)
        # Batched free greedy generation. Padding follows each current prefix;
        # each selected last position therefore has only real causal ancestors.
        prefixes = [net.encode(p) for p in prompts]; generated = [[] for _ in rows]; ended = [False] * len(rows)
        for _ in range(4):
            gx = torch.zeros((len(rows), max(map(len, prefixes))), dtype=torch.long, device=device)
            for i, ids in enumerate(prefixes): gx[i, :len(ids)] = torch.tensor(ids, device=device)
            gl, _ = m(gx)
            values = gl[torch.arange(len(rows), device=device), torch.tensor([len(ids) - 1 for ids in prefixes], device=device)].argmax(-1).cpu().tolist()
            for i, value in enumerate(values):
                if not ended[i]:
                    if value == net.EOS: ended[i] = True
                    else: generated[i].append(value)
                prefixes[i].append(value)
        for i, row in enumerate(rows):
            text = net.decode(generated[i]); answer = text.strip()
            predictions.append({**row, 'prediction': 'AB'[chosen[i]], 'correct': 'AB'[chosen[i]] == row['answer'],
                                'answer_loss': float(losses[i].cpu()), 'generation': text,
                                'generation_correct': answer == row['answer'],
                                'answer_letter_log_probabilities': torch.log_softmax(ll[i], -1).cpu().tolist()})
    del m; gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    def metric(rs):
        correct = sum(r['correct'] for r in rs); total = len(rs)
        return {'correct': correct, 'total': total, 'accuracy': correct / total if total else 0.,
                'answer_loss': float(np.mean([r['answer_loss'] for r in rs])),
                'generation_correct': sum(r['generation_correct'] for r in rs),
                'generation_accuracy': sum(r['generation_correct'] for r in rs) / total}
    return {'overall': metric(predictions), 'by_family': {f: metric([r for r in predictions if r['family'] == f]) for f in sorted({r['family'] for r in predictions})},
            'by_depth': {str(d): metric([r for r in predictions if r['depth'] == d]) for d in sorted({r['depth'] for r in predictions})},
            'predictions': predictions, 'checkpoint_sha256': digest(Path(checkpoint).read_bytes()),
            'input_sha256': digest(questions), 'input_fields': ['id', 'question', 'answer', 'family', 'depth'],
            'prediction_boundary': 'question and constant space; answer only used for scoring',
            'schema': 'lemmaferry.scratch.evaluation.v1'}


def public_questions(rows):
    return [{k: r[k] for k in ('id', 'question', 'answer', 'family', 'depth')} for r in rows]


def study(root, config, device):
    root = Path(root); validate_config(config); plan = freeze_plan(root, config)
    dataset = read(root / 'dataset.json'); all_train = [r for r in dataset['rows'] if r['split'] == 'train']
    validation = [r for r in dataset['rows'] if r['split'] == 'validation']
    tests = {s: [r for r in dataset['rows'] if r['split'] == s] for s in ('test', 'harder')}
    pretrain(root, config, device)
    for seed in config['seeds']:
        for size in config['sizes']:
            train = all_train[:size]
            if len(train) != size: raise ValueError('insufficient distinct training data')
            directory = root / 'study' / f'{seed}-{size}'
            ae = proof_encoder(directory, train, validation, config, seed, device)
            for arm in ARMS:
                result = directory / arm / 'evaluation.json'
                if result.exists(): continue
                tick(); m, opt, state = net.restore(root / 'starting' / f'{seed}.pt', device)
                for stage in range(config['stages']):
                    stage_dir = directory / arm / f'stage-{stage}'
                    path = stage_dir / 'recipient.pt'
                    if path.exists():
                        saved = load(path)
                        if saved['metadata']['stage_updates'] == config['recipient_steps'] // config['stages'] and (stage_dir / 'budget.json').exists():
                            m, opt, _ = net.restore(path, device)
                            continue
                        # Keep the preceding complete model for cached producer validation;
                        # train_recipient restores any partial update checkpoint itself.
                    bridge_dir = directory / ('shared-stage-0' if stage == 0 else f'{arm}/stage-{stage}')
                    pair, vectors = fit_bridge(bridge_dir, ae, m, train, validation, config, seed + stage, device)
                    m, opt = train_recipient(stage_dir, m, opt, config['pretrain_steps'] + stage * (config['recipient_steps'] // config['stages']),
                                            train, pair, vectors, config, seed, arm, stage, device)
                    # Held-out geometry is opened after fitting; never affects selection.
                    if not (stage_dir / 'geometry.json').exists():
                        test_rows = tests['test'][:128]
                        tx = proof_vectors(ae, test_rows, config, device)
                        # Snapshot language vectors must come from the PRE-update recipient.
                        snap, _, _ = net.restore(root / 'starting' / f'{seed}.pt' if stage == 0 else directory / arm / f'stage-{stage - 1}/recipient.pt', device)
                        ty = net.represent(snap, [r['solution'] for r in test_rows], device).numpy(); del snap
                        report = pair.evaluate(PairedSplit(tx, ty, [r['id'] for r in test_rows], [r['id'] for r in test_rows], 'test'))
                        write(stage_dir / 'geometry.json', report)
                    tick()
                del m, opt; gc.collect()
                final = directory / arm / f'stage-{config["stages"] - 1}/recipient.pt'
                for stage in range(config['stages']):
                    sd = directory / arm / f'stage-{stage}'
                    if not (sd / 'geometry.json').exists():
                        bd = directory / ('shared-stage-0' if stage == 0 else f'{arm}/stage-{stage}')
                        pair = SwapPair.load(bd / 'maps')
                        snapshot = root / 'starting' / f'{seed}.pt' if stage == 0 else directory / arm / f'stage-{stage - 1}/recipient.pt'
                        snap, snap_opt, _ = net.restore(snapshot, device)
                        tr = tests['test'][:128]
                        tx = proof_vectors(ae, tr, config, device)
                        ty = net.represent(snap, [r['solution'] for r in tr], device).numpy()
                        write(sd / 'geometry.json', pair.evaluate(PairedSplit(tx, ty, [r['id'] for r in tr], [r['id'] for r in tr], 'test')))
                        del snap, snap_opt
                write(result, {s: evaluate_recipient(final, public_questions(rows), device) for s, rows in tests.items()})
                progress(root, 'arm_complete', seed=seed, size=size, arm=arm)
            del ae; gc.collect()
        baseline = root / 'starting' / f'{seed}-evaluation.json'
        if not baseline.exists():
            write(baseline, {s: evaluate_recipient(root / 'starting' / f'{seed}.pt', public_questions(rows), device)
                             for s, rows in tests.items()})
    tick(); progress(root, 'study_complete', environment=environment(device))


def extend(root, config, device, *, additional=256, seed=7, size=4096, arm='swap_paired', updates=128):
    """Append a separate continuation, preserving the frozen independent study."""
    root = Path(root)
    if arm not in ARMS or seed not in config['seeds'] or size not in config['sizes']:
        raise ValueError('continuation must name an existing study condition')
    if additional <= 0 or updates <= 0 or updates * (config['batch'] // 2) < additional:
        raise ValueError('positive budgets must consume all newly admitted problems')
    # Complete a pending continuation before creating another dataset extension.
    history_path = root / 'continuations.json'
    history = read(history_path) if history_path.exists() else {'schema': 1, 'entries': []}
    pending = history['entries'][-1] if history['entries'] and not history['entries'][-1].get('complete') else None
    if pending:
        if (pending['seed'], pending['size'], pending['arm'], pending['additional'], pending['updates']) != (seed, size, arm, additional, updates):
            raise ValueError('resume pending continuation with its original arguments first')
        entry = pending
        extension = read(root / entry['extension'])
    else:
        extension = generate(root, extra=additional, context=config['model']['context'])
        ext_files = sorted((root / 'extensions').glob('extension-*.json'))
        parent = root / 'study' / f'{seed}-{size}' / arm / f'stage-{config["stages"] - 1}/recipient.pt'
        previous = [e for e in history['entries'] if e.get('complete') and (e['seed'], e['size'], e['arm']) == (seed, size, arm)]
        if previous: parent = root / previous[-1]['checkpoint']
        index = len(history['entries'])
        entry = {'index': index, 'seed': seed, 'size': size, 'arm': arm, 'additional': additional, 'updates': updates,
                 'extension': str(ext_files[-1].relative_to(root)), 'parent_checkpoint': str(parent.relative_to(root)),
                 'directory': f'continuation-{index:03d}', 'complete': False}
        history['entries'].append(entry); write(history_path, history)
    base = read(root / 'dataset.json')
    train = [r for r in base['rows'] if r['split'] == 'train'][:size]
    eligible = [e for e in history['entries'] if (e['seed'], e['size'], e['arm']) == (seed, size, arm) and e['index'] <= entry['index']]
    for prior in eligible: train += read(root / prior['extension'])['rows']
    validation = [r for r in base['rows'] if r['split'] == 'validation']
    c = copy.deepcopy(config); c.update(recipient_steps=updates, stages=1, priority_ids=[r['id'] for r in extension['rows']])
    directory = root / entry['directory']
    write(directory / 'config.json', c)
    ae = proof_encoder(directory, train, validation, c, seed, device)
    m, opt, parent_state = net.restore(root / entry['parent_checkpoint'], device)
    pair, vectors = fit_bridge(directory / 'shared-stage-0', ae, m, train, validation, c, seed + 100 + entry['index'], device)
    stage = directory / arm / 'stage-0'
    m, opt = train_recipient(stage, m, opt, parent_state['updates'], train, pair, vectors, c, seed, arm, 0, device)
    final = stage / 'recipient.pt'; del m, opt, ae; gc.collect()
    evaluation = {s: evaluate_recipient(final, public_questions([r for r in base['rows'] if r['split'] == s]), device) for s in ('test', 'harder')}
    write(directory / 'evaluation.json', evaluation)
    entry.update(complete=True, checkpoint=str(final.relative_to(root)), checkpoint_sha256=digest(final.read_bytes()),
                 base_dataset_sha256=base['fingerprint'], extension_sha256=extension['fingerprint'],
                 starting_updates=parent_state['updates'], ending_updates=parent_state['updates'] + updates,
                 new_problems_consumed=additional)
    write(history_path, history); progress(root, 'continuation_complete', **entry)
    return entry
