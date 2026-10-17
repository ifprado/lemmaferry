#!/usr/bin/env python3
"""One cached pretrained adapter update; no download and no capability claim."""
from __future__ import annotations
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import torch
from lemmaferry.learning import (HFRecipient, TrainingBudget, evaluate, freeze_evaluation,
    load_recipient, parameter_sha256, save_recipient, train_update)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/constructive-loop/hf-adapter-smoke.json'
WORK = ROOT / 'runs/constructive-loop/hf-adapter-smoke'
REV = '8faed761d45a263340a0528343f099c05c9a4323'
OUT.parent.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
torch.set_num_threads(2)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
report = {'status': 'started', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'One optimizer step validates the actual pretrained adapter backend; not a verified-data transfer or capability-gain experiment.',
          'model_id': 'Qwen/Qwen2.5-1.5B', 'revision': REV, 'tokenizer_revision': REV,
          'local_files_only': True, 'device': device, 'torch_version': str(torch.__version__),
          'command': '.venv-llm/bin/python experiments/smoke_pretrained_recipient.py',
          'training_fixture': {'lean_verified': False, 'use': 'synthetic_fixture_backend_smoke'}}
def save():
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
save()
try:
    fixed = freeze_evaluation(OUT.with_name('hf-adapter-evaluation.json'), [
        {'id': 'adapter-smoke-external:0', 'prompt': 'What is two plus three?\nA: five\nB: six\nAnswer:',
         'answer': 'A', 'family': 'independent_arithmetic'},
        {'id': 'adapter-smoke-external:1', 'prompt': 'The report states that the parcel is blue. What color is the parcel?\nA: red\nB: blue\nAnswer:',
         'answer': 'B', 'family': 'independent_fact_retrieval'}])
    report['fixed_evaluation'] = {'path': str(Path(fixed.path).relative_to(ROOT)), 'sha256': fixed.sha256,
                                   'examples': list(fixed.examples)}
    row = {'id': 'adapter-smoke-training:0',
           'prompt': 'If the green lamp is on, the bell rings. The green lamp is on. What follows?\nA: the bell rings\nB: the bell is silent\nAnswer:',
           'answer': 'A', 'family': 'fixture_modus_ponens', 'split': 'train',
           'provenance': {'use': 'synthetic_fixture_backend_smoke', 'lean_verified': False}}
    report['training_example'] = row
    report['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['implementation_sha256'] = hashlib.sha256((ROOT / 'src/lemmaferry/learning.py').read_bytes()).hexdigest()
    load_started = time.perf_counter()
    model = HFRecipient(model_id=report['model_id'], revision=REV, rank=8, max_length=128,
                        device=device, local_files_only=True)
    report['load_seconds'] = time.perf_counter() - load_started
    report['parameter_counts'] = {'frozen_base': sum(p.numel() for p in model.module.base.parameters()),
                                   'trainable_adapter': sum(p.numel() for p in model.trainable_parameters())}
    report['boundary'] = model.representation_boundary
    before_versions = {n: p._version for n, p in model.module.base.named_parameters()}
    report['base_requires_grad_false'] = all(not p.requires_grad for p in model.module.base.parameters())
    report['before_evaluation'] = evaluate(model, fixed)
    train_started = time.perf_counter()
    report['update'] = train_update(model, [row], budget=TrainingBudget(steps=1,batch_size=1,max_length=128),
                                    fixed_evaluation=fixed,seed=1729)
    if device == 'mps':
        torch.mps.synchronize()
    report['training_seconds'] = time.perf_counter() - train_started
    report['base_tensor_versions_unchanged'] = before_versions == {n:p._version for n,p in model.module.base.named_parameters()}
    report['base_has_no_gradients'] = all(p.grad is None for p in model.module.base.parameters())
    trainable = {id(p) for p in model.trainable_parameters()}
    optimizer_parameters = {id(p) for group in model.optimizer.param_groups for p in group['params']}
    report['optimizer_contains_only_adapter'] = trainable == optimizer_parameters
    report['trainable_gradient_l2'] = {n: float(p.grad.float().norm().detach().cpu()) for n,p in model.module.named_parameters() if p.requires_grad}
    report['after_evaluation'] = evaluate(model, fixed)
    checkpoint = WORK / 'adapter.pt'
    report['checkpoint'] = save_recipient(model, checkpoint)
    report['checkpoint']['path'] = str(checkpoint.relative_to(ROOT))
    expected = report['after_evaluation']
    del model
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    reload_started = time.perf_counter()
    restored = load_recipient(checkpoint, device=device)
    actual = evaluate(restored, fixed)
    report['reload_seconds'] = time.perf_counter() - reload_started
    report['reload_evaluation'] = actual
    report['reload_exact_predictions_and_losses'] = actual == expected
    report['reload_optimizer_steps'] = sorted({int(v['step'].detach().cpu()) for v in restored.optimizer.state.values()})
    fixed.verify()
    report['fixed_evaluation_unchanged'] = True
    checks = ('base_requires_grad_false', 'base_tensor_versions_unchanged', 'base_has_no_gradients',
              'optimizer_contains_only_adapter', 'reload_exact_predictions_and_losses', 'fixed_evaluation_unchanged')
    if not all(report[k] for k in checks):
        raise ValueError('one adapter smoke invariant failed')
    if not any(v > 0 for v in report['trainable_gradient_l2'].values()):
        raise ValueError('no adapter gradient')
    if report['update']['parameter_sha256_before'] == report['update']['parameter_sha256_after']:
        raise ValueError('adapter did not update')
    if device == 'mps':
        report['sampled_mps_allocated_bytes'] = torch.mps.current_allocated_memory()
        report['sampled_mps_driver_bytes'] = torch.mps.driver_allocated_memory()
    del restored
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_lifetime_process_rss_bytes'] = peak if sys.platform == 'darwin' else peak * 1024
    report.update(status='passed', total_seconds=time.perf_counter()-started)
    save()
    print(json.dumps({'status':report['status'],'device':device,'training_seconds':report['training_seconds'],
                      'total_seconds':report['total_seconds'],'adapter_hash_before':report['update']['parameter_sha256_before'],
                      'adapter_hash_after':report['update']['parameter_sha256_after']},indent=2),flush=True)
except Exception as exc:
    report.update(status='failed',error=f'{type(exc).__name__}: {exc}',total_seconds=time.perf_counter()-started)
    save()
    raise
