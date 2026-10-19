"""Randomly initialized byte causal LM and a vector-only proof autoencoder.

Training batches use fixed right-padded shapes. Representation extraction can use
shorter right-padded batches: causal attention makes each real position invariant
to later padding. No tokenizer or pretrained model is downloaded by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .io import digest, save, load, state_digest

PAD, BOS, EOS, VOCAB = 0, 1, 2, 259
TOKENIZER = {'kind': 'utf8-byte', 'pad': PAD, 'bos': BOS, 'eos': EOS, 'offset': 3, 'vocab': VOCAB}
SCHEMA = 'lemmaferry.scratch.recipient.v1'


def encode(text):
    """One BOS plus literal UTF-8 bytes; reserved tokens cannot alias a byte."""
    if not isinstance(text, str):
        raise TypeError('byte tokenizer expects text')
    return [BOS] + [value + 3 for value in text.encode('utf-8')]


def decode(ids):
    values = [int(value) for value in ids]
    if any(value < 0 or value >= VOCAB for value in values):
        raise ValueError('token ID outside byte vocabulary')
    return bytes(value - 3 for value in values if value >= 3).decode('utf-8', errors='replace')


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 6
    width: int = 256
    heads: int = 4
    context: int = 512

    def __post_init__(self):
        if any(type(value) is not int or value < 1
               for value in (self.layers, self.width, self.heads, self.context)):
            raise ValueError('architecture values must be positive integers')
        if self.width % self.heads:
            raise ValueError('model width must be divisible by attention heads')


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(config.width), nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.attn = nn.Linear(config.width, config.width)
        self.ff = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(),
                                nn.Linear(4 * config.width, config.width))
        self.heads = config.heads

    def forward(self, x):
        batch_size, length, width = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(
            batch_size, length, 3, self.heads, width // self.heads).unbind(2)
        # Right padding follows real tokens and therefore cannot be an ancestor
        # of a scored position. Avoid fully masked rows and backend-specific NaNs.
        attended = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                      v.transpose(1, 2), dropout_p=0.0, is_causal=True)
        x = x + self.attn(attended.transpose(1, 2).reshape(batch_size, length, width))
        return x + self.ff(self.n2(x))


class LM(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError('LM requires ModelConfig')
        self.config = config
        self.tokens = nn.Embedding(VOCAB, config.width, padding_idx=PAD)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, VOCAB, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding) and module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    def forward(self, ids):
        if ids.ndim != 2 or min(ids.shape) < 1 or ids.dtype != torch.long:
            raise ValueError('LM input must be a nonempty [batch,length] int64 tensor')
        if ids.shape[1] > self.config.context:
            raise ValueError('overlength input; refusing truncation')
        x = self.tokens(ids) + self.positions(torch.arange(ids.shape[1], device=ids.device))
        hidden = self.norm(self.blocks(x))
        return self.head(hidden), hidden


def batch(texts, context, device, prompts=None, full=False, *, pad_to_context=True, include_eos=True):
    """Prepare exact bytes and unshifted labels, with an explicit no-truncation gate.

    With ``prompts``, the final prompt byte is the state position. The first
    supervised continuation may be a separating space; the next logit predicts
    the answer letter without seeing that letter. ``loss`` shifts labels once.
    Without prompts, position is the final content byte (or BOS for empty text).
    Fixed padding is the training default; dynamic padding is only opt-in.
    """
    texts = list(texts)
    if not texts or type(context) is not int or context < 1:
        raise ValueError('a nonempty text batch and positive context are required')
    if prompts is not None:
        prompts = list(prompts)
        if len(prompts) != len(texts):
            raise ValueError('one prompt prefix is required per text')
    arrays, labels, positions = [], [], []
    for index, text in enumerate(texts):
        content = encode(text)
        ids = content + ([EOS] if include_eos else [])
        if len(ids) > context:
            raise ValueError(f'overlength ({len(ids)} > {context}); refusing truncation')
        position = len(content) - 1
        ys = ids.copy()
        if prompts is not None:
            if not isinstance(prompts[index], str) or not text.startswith(prompts[index]):
                raise ValueError('supervised text must begin with its exact prompt prefix')
            position = len(encode(prompts[index])) - 1
            if not full:
                ys[:position + 1] = [-100] * (position + 1)
        arrays.append(ids)
        labels.append(ys)
        positions.append(position)
    padded_length = context if pad_to_context else max(map(len, arrays))
    x = torch.full((len(texts), padded_length), PAD, dtype=torch.long)
    y = torch.full_like(x, -100)
    for index, (ids, ys) in enumerate(zip(arrays, labels)):
        x[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        y[index, :len(ids)] = torch.tensor(ys, dtype=torch.long)
    return x.to(device), y.to(device), torch.tensor(positions, dtype=torch.long, device=device)


def loss(logits, labels):
    """Teacher-forced CE: position t predicts label t+1; padding is ignored."""
    if logits.ndim != 3 or logits.shape[-1] != VOCAB or labels.shape != logits.shape[:2]:
        raise ValueError('logits/labels have incompatible sequence shapes')
    if logits.shape[1] < 2 or not bool(labels[:, 1:].ne(-100).any()):
        raise ValueError('at least one next-token target is required')
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


def optimizer(model, lr=.0005):
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('learning rate must be finite and positive')
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01, betas=(.9, .95))



def state_hash(value: Any) -> str:
    """Alias for state_digest, used by execution receipts."""
    return state_digest(value)


def _rng_state(device):
    numpy_state = np.random.get_state()
    return {'torch_cpu': torch.get_rng_state().cpu(), 'python': random.getstate(),
            'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            'torch_mps': torch.mps.get_rng_state().cpu() if device == 'mps' else None}


def _restore_rng(state, device):
    torch.set_rng_state(state['torch_cpu'].cpu())
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if device == 'mps' and state['torch_mps'] is not None:
        torch.mps.set_rng_state(state['torch_mps'].cpu())


def checkpoint(path, model, opt, updates, **metadata):
    """Atomic, hash-bound recipient/optimizer/RNG checkpoint; all weights persist."""
    if type(updates) is not int or updates < 0 or not isinstance(opt, torch.optim.AdamW):
        raise ValueError('checkpoint requires nonnegative updates and an AdamW optimizer')
    model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer_state = opt.state_dict()
    rng = _rng_state(next(model.parameters()).device.type)
    save(path, {'schema': SCHEMA, 'config': asdict(model.config), 'config_sha256': digest(asdict(model.config)),
                'tokenizer': dict(TOKENIZER),
                'tokenizer_sha256': digest(TOKENIZER), 'model': model_state,
                'model_sha256': state_digest(model_state), 'optimizer': optimizer_state,
                'optimizer_sha256': state_digest(optimizer_state), 'optimizer_class': 'AdamW',
                'updates': updates, 'rng': rng, 'rng_sha256': state_digest(rng), 'metadata': metadata})


def _validate_optimizer(state, model):
    if not isinstance(state, dict) or set(state) != {'state', 'param_groups'} or len(state['param_groups']) != 1:
        raise ValueError('unsupported recipient optimizer schema')
    group = state['param_groups'][0]
    parameters = list(model.parameters())
    indices = group['params']
    if len(indices) != len(parameters) or len(set(indices)) != len(indices):
        raise ValueError('optimizer parameter inventory differs from recipient')
    if set(state['state']) - set(indices):
        raise ValueError('optimizer state refers to unknown parameters')
    for index, parameter in zip(indices, parameters):
        for key, value in state['state'].get(index, {}).items():
            if key == 'step':
                if not isinstance(value, torch.Tensor) or value.numel() != 1 or not bool(torch.isfinite(value)) or float(value) < 0:
                    raise ValueError('invalid optimizer step')
            elif key in {'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'}:
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError('optimizer moment shape/dtype/value mismatch')
            else:
                raise ValueError('unknown optimizer moment')


def restore(path, device):
    """Strictly restore weights, optimizer moments and RNG after initialization."""
    # Parse on CPU; a checkpoint's CPU RNG must never be remapped to MPS.
    state = load(path, 'cpu')
    if state.get('schema') != SCHEMA or state.get('optimizer_class') != 'AdamW':
        raise ValueError('unsupported recipient checkpoint schema')
    if state['config_sha256'] != digest(state['config']):
        raise ValueError('recipient configuration hash mismatch')
    if state['tokenizer'] != TOKENIZER or state['tokenizer_sha256'] != digest(TOKENIZER):
        raise ValueError('tokenizer mismatch')
    for name in ('model', 'optimizer', 'rng'):
        if state[name + '_sha256'] != state_digest(state[name]):
            raise ValueError(f'{name} state hash mismatch')
    if type(state['updates']) is not int or state['updates'] < 0:
        raise ValueError('invalid recipient update count')
    model = LM(ModelConfig(**state['config'])).to(device)
    expected = model.state_dict()
    if set(expected) != set(state['model']):
        raise ValueError('recipient tensor keys differ from architecture')
    for key, value in state['model'].items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                or value.dtype != expected[key].dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('recipient tensor shape/dtype/value mismatch')
    model.load_state_dict(state['model'], strict=True)
    _validate_optimizer(state['optimizer'], model)
    if any(float(value['step']) != state['updates'] for value in state['optimizer']['state'].values()):
        raise ValueError('optimizer steps differ from declared recipient updates')
    opt = optimizer(model)
    opt.load_state_dict(state['optimizer'])
    # AdamW step counters remain on CPU unless the optimizer is capturable.
    for parameter_state in opt.state.values():
        for key, value in parameter_state.items():
            if isinstance(value, torch.Tensor) and key != 'step':
                parameter_state[key] = value.to(device)
    _restore_rng(state['rng'], torch.device(device).type)
    return model, opt, state


@torch.no_grad()
def represent(model, texts, device, size=16):
    """Frozen final-normalized state at the final text byte, before any EOS."""
    texts = list(texts)
    if type(size) is not int or size < 1:
        raise ValueError('representation batch size must be positive')
    model.eval()
    result = []
    for start in range(0, len(texts), size):
        x, _, positions = batch(texts[start:start + size], model.config.context, device,
                                pad_to_context=False, include_eos=False)
        _, hidden = model(x)
        result.append(hidden[torch.arange(len(x), device=device), positions].detach().cpu())
    return torch.cat(result) if result else torch.empty((0, model.config.width))


@torch.no_grad()
def generate(model, prompt, device, tokens=32):
    model.eval()
    ids = encode(prompt)
    if len(ids) > model.config.context or type(tokens) is not int or tokens < 0:
        raise ValueError('overlength prompt or invalid generation budget')
    output = []
    for _ in range(tokens):
        if len(ids) >= model.config.context:
            break
        logits, _ = model(torch.tensor([ids], dtype=torch.long, device=device))
        value = int(logits[0, -1].argmax())
        if value == EOS:
            break
        output.append(value)
        ids.append(value)
    return decode(output)


class ProofAE(nn.Module):
    """Masked convolutional encoder and parallel decoder receiving ONLY z+position.

    No teacher forcing, input attention, token skip path or copied prefix reaches
    the decoder. Canonical proof content must survive the fixed vector bottleneck.
    The training pipeline freezes this separately trained model before extraction.
    """
    def __init__(self, width=96, latent=64, context=512):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (width, latent, context)):
            raise ValueError('proof-autoencoder dimensions must be positive integers')
        self.context, self.latent = context, latent
        self.emb = nn.Embedding(VOCAB, width, padding_idx=PAD)
        self.conv = nn.Sequential(nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(width, width, 5, padding=2), nn.GELU())
        self.to_z = nn.Linear(width * 2, latent)
        self.positions = nn.Embedding(context, width)
        self.from_z = nn.Linear(latent, width)
        self.decoder = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, VOCAB))

    def encode(self, ids):
        if ids.ndim != 2 or min(ids.shape) < 1 or ids.shape[1] > self.context or ids.dtype != torch.long:
            raise ValueError('invalid proof-encoder input shape or context')
        mask = ids.ne(PAD)
        if not bool(mask.any(1).all()):
            raise ValueError('proof encoder cannot pool an empty padded sequence')
        hidden = self.emb(ids).transpose(1, 2)
        # Mask each convolutional stage, so extra right padding cannot feed bias
        # activations back into a real token through the next convolution.
        for layer in self.conv:
            hidden = layer(hidden) * mask[:, None]
        hidden = hidden.transpose(1, 2)
        average = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        maximum = hidden.masked_fill(~mask[..., None], float('-inf')).amax(1)
        return self.to_z(torch.cat([average, maximum], dim=1))

    def decode(self, z, length):
        if (z.ndim != 2 or len(z) < 1 or z.shape[1] != self.latent
                or type(length) is not int or not 1 <= length <= self.context):
            raise ValueError('invalid proof vector or decoder length')
        hidden = self.from_z(z)[:, None] + self.positions(torch.arange(length, device=z.device))[None]
        return self.decoder(hidden)

    def forward(self, ids):
        return self.decode(self.encode(ids), ids.shape[1])
