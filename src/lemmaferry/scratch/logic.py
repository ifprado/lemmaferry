"""Typed compositional certificates, countermodels, structural splits and Lean.

Shape depth is independent of proof spelling, not a minimum discovery difficulty.
The common lemmas are fixed atomic rules, never a held-out solution library.
"""
from collections import Counter
import copy
import itertools
import random
import re
from pathlib import Path
from ..library import VerifiedLibrary, PIN, LEAN_COMMIT
from .io import digest, read, write

ATOMS = 4
OPS = {'hyp', 'lemma', 'intro', 'pair', 'apply', 'left', 'right'}
COMMON = [
    {'id': 'identity', 'formula': ['imp', 0, 0], 'proof': ['intro', 0, ['hyp', 0]]},
    {'id': 'project', 'formula': ['imp', ['and', 0, 1], 0],
     'proof': ['intro', ['and', 0, 1], ['left', ['hyp', 0]]]},
]
COMMON_SOURCE = ('theorem lf_identity (p : Prop) : p → p := fun h => h\n'
                 'theorem lf_project (p q : Prop) : (p ∧ q) → p := fun h => And.left h\n')


def fmt(f, names=None, lean=False, _fuel=128):
    names = names if names is not None else ['p0', 'p1', 'p2', 'p3']
    if _fuel <= 0: raise ValueError('formula exceeds budget')
    if type(f) is int:
        if not 0 <= f < min(ATOMS, len(names)): raise ValueError('atom out of scope')
        return names[f]
    if not isinstance(f, list) or len(f) != 3 or f[0] not in ('and', 'imp'):
        raise ValueError('invalid formula')
    a, b = (fmt(v, names, lean, _fuel - 1) for v in f[1:])
    if lean: return f'({a} {"∧" if f[0] == "and" else "→"} {b})'
    return f'({a} and {b})' if f[0] == 'and' else f'(if {a} then {b})'


def _shape(p, bound, budget=None):
    budget = [1024] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0 or not isinstance(p, list) or not p or not isinstance(p[0], str) or p[0] not in OPS:
        raise ValueError('invalid or excessive proof constructor')
    op, *a = p
    if op == 'hyp' and len(a) == 1 and type(a[0]) is int and 0 <= a[0] < bound: return
    if op == 'lemma' and len(a) == 3 and a[0] in ('identity', 'project'):
        fmt(a[1]); fmt(a[2]); return
    if op == 'intro' and len(a) == 2:
        fmt(a[0]); _shape(a[1], bound + 1, budget); return
    arity = 2 if op in ('pair', 'apply') else 1 if op in ('left', 'right') else None
    if arity is not None and len(a) == arity:
        for child in a: _shape(child, bound, budget)
        return
    raise ValueError('ill-scoped or forbidden proof constructor')


def infer(p, context, fuel=256):
    if not isinstance(context, list): raise ValueError('context must be a list')
    for f in context: fmt(f)
    _shape(p, len(context))
    def go(node, ctx, left):
        if left <= 0: raise ValueError('proof exceeds depth budget')
        op, *a = node
        rec = lambda v: go(v, ctx, left - 1)
        if op == 'hyp': return ctx[a[0]]
        if op == 'lemma': return ['imp', a[1], a[1]] if a[0] == 'identity' else ['imp', ['and', a[1], a[2]], a[1]]
        if op == 'intro': return ['imp', a[0], go(a[1], ctx + [a[0]], left - 1)]
        if op == 'pair': return ['and', rec(a[0]), rec(a[1])]
        if op == 'apply':
            fn, arg = rec(a[0]), rec(a[1])
            if isinstance(fn, list) and fn[0] == 'imp' and fn[1] == arg: return fn[2]
        if op in ('left', 'right'):
            value = rec(a[0])
            if isinstance(value, list) and value[0] == 'and': return value[1 if op == 'left' else 2]
        raise ValueError('ill-typed proof')
    return go(p, context, fuel)


def term(p, context=0):
    _shape(p, context)
    def go(node, bound):
        op, *a = node
        if op == 'hyp': return f'h{a[0]}'
        if op == 'lemma':
            return (f'(lf_project {fmt(a[1], lean=True)} {fmt(a[2], lean=True)})' if a[0] == 'project'
                    else f'(lf_identity {fmt(a[1], lean=True)})')
        if op == 'intro': return f'(fun h{bound} : {fmt(a[0], lean=True)} => {go(a[1], bound + 1)})'
        if op == 'pair': return f'(And.intro {go(a[0], bound)} {go(a[1], bound)})'
        if op == 'apply': return f'({go(a[0], bound)} {go(a[1], bound)})'
        return f'(And.{op} {go(a[0], bound)})'
    return go(p, context)


def truth(f, bits):
    fmt(f)
    if len(bits) != ATOMS or any(type(b) is not bool for b in bits): raise ValueError('four Boolean atoms required')
    def go(v):
        if type(v) is int: return bits[v]
        a, b = go(v[1]), go(v[2])
        return a and b if v[0] == 'and' else not a or b
    return go(f)


def countermodel(premises, goal):
    for bits in itertools.product((False, True), repeat=ATOMS):
        if all(truth(p, bits) for p in premises) and not truth(goal, bits): return list(bits)
    return None


def substitute(f, perm):
    return perm[f] if type(f) is int else [f[0], substitute(f[1], perm), substitute(f[2], perm)]


def structure(premises, goal):
    """Exact atom-renaming/premise-order identity, not semantic equivalence."""
    for f in [*premises, goal]: fmt(f)
    def encode(f, mapping):
        return str(mapping[f]) if type(f) is int else '(' + ('&' if f[0] == 'and' else '>') + encode(f[1], mapping) + encode(f[2], mapping) + ')'
    def atoms(f): return {f} if type(f) is int else atoms(f[1]) | atoms(f[2])
    used = sorted(set().union(*(atoms(f) for f in [*premises, goal])))
    return digest(min((','.join(sorted({encode(p, dict(zip(used, values))) for p in premises})), encode(goal, dict(zip(used, values))))
                      for values in itertools.permutations(range(len(used)))))


def formula_depth(f):
    return 1 if type(f) is int else 1 + max(formula_depth(f[1]), formula_depth(f[2]))


def formula_shape(f):
    """Erase atom identity while preserving every ordered connective and leaf."""
    return 'atom' if type(f) is int else [f[0], formula_shape(f[1]), formula_shape(f[2])]


def depth(p):
    return 1 + max([depth(v) for v in p[1:] if isinstance(v, list) and v and isinstance(v[0], str) and v[0] in OPS] or [0])


def dependencies(p):
    if p[0] == 'lemma': return [p[1]]
    return sorted(set(d for v in p[1:] if isinstance(v, list) and v and isinstance(v[0], str) and v[0] in OPS for d in dependencies(v)))


def shift(p, cutoff, delta=1):
    """Shift absolute indices including nested binders when inserting a context slot."""
    if p[0] == 'hyp': return ['hyp', p[1] + (delta if p[1] >= cutoff else 0)]
    if p[0] == 'lemma': return copy.deepcopy(p)
    if p[0] == 'intro': return ['intro', copy.deepcopy(p[1]), shift(p[2], cutoff, delta)]
    return [p[0]] + [shift(v, cutoff, delta) for v in p[1:]]


def _replace_bound(p, index, value, nested=0):
    if p[0] == 'hyp': return shift(value, index, nested) if p[1] == index else ['hyp', p[1] - int(p[1] > index)]
    if p[0] == 'lemma': return copy.deepcopy(p)
    if p[0] == 'intro': return ['intro', p[1], _replace_bound(p[2], index, value, nested + 1)]
    return [p[0]] + [_replace_bound(v, index, value, nested) for v in p[1:]]


def normalize(p, context):
    """Remove beta/projection detours before counting certificate depth."""
    op, *a = p
    if op in ('hyp', 'lemma'): return copy.deepcopy(p)
    if op == 'intro': return ['intro', a[0], normalize(a[1], context + 1)]
    a = [normalize(v, context) for v in a]
    if op in ('left', 'right') and a[0][0] == 'pair': return a[0][1 if op == 'left' else 2]
    if op == 'apply' and a[0][0] == 'intro': return normalize(_replace_bound(a[0][2], context, a[1]), context)
    if op == 'apply' and a[0][0] == 'lemma' and a[0][1] == 'identity': return a[1]
    if op == 'apply' and a[0][0] == 'lemma' and a[0][1] == 'project': return normalize(['left', a[1]], context)
    return [op] + a


def _trim(premises, proof):
    used = set()
    def visit(p):
        if p[0] == 'hyp' and p[1] < len(premises): used.add(p[1])
        elif p[0] not in ('hyp', 'lemma'):
            for child in (p[2:] if p[0] == 'intro' else p[1:]): visit(child)
    visit(proof)
    selected = sorted(used); mapping = {old: new for new, old in enumerate(selected)}
    def remap(p):
        if p[0] == 'hyp': return ['hyp', mapping[p[1]] if p[1] < len(premises) else p[1] - len(premises) + len(selected)]
        if p[0] == 'lemma': return copy.deepcopy(p)
        if p[0] == 'intro': return ['intro', p[1], remap(p[2])]
        return [p[0]] + [remap(v) for v in p[1:]]
    return [premises[i] for i in selected], remap(proof)


def candidate(seed, harder=False):
    r = random.Random(seed)
    def formula(d=1):
        if not d or r.random() < .60: return r.randrange(ATOMS)
        return [r.choice(['and', 'imp']), formula(d - 1), formula(d - 1)]
    premises = list({fmt(p): p for p in [formula(2) for _ in range(r.randint(3, 7))]}.values())
    values = [(p, ['hyp', i]) for i, p in enumerate(premises)]
    for _ in range(r.randint(10, 20) if harder else r.randint(5, 12)):
        a, pa = values[-1] if harder and r.random() < .6 else r.choice(values)
        b, pb = r.choice(values); op = r.randrange(7)
        if op == 0: value, proof = ['and', a, b], ['pair', pa, pb]
        elif op == 1 and isinstance(a, list) and a[0] == 'and':
            side = r.randrange(2); value, proof = a[side + 1], ['left' if side == 0 else 'right', pa]
        elif op == 2:
            funcs = [(f, p) for f, p in values if isinstance(f, list) and f[0] == 'imp' and f[1] == a]
            if funcs and r.random() < .4:
                f, p = r.choice(funcs)
            else:
                # Grow a typed proof graph rather than waiting for accidental
                # random premise matches. New arrows consume already-derived
                # compound values, producing multi-premise mixed-operation DAGs.
                target = formula(1)
                f = ['imp', a, target]
                if len(fmt(f)) > 90 or (not harder and formula_depth(f) > 3): continue
                if f not in premises:
                    insertion = len(premises)
                    values = [(v, shift(p, insertion)) for v, p in values]
                    pa = shift(pa, insertion)
                    premises.append(f)
                    values.append((f, ['hyp', insertion]))
                p = ['hyp', premises.index(f)]
            value, proof = f[2], ['apply', p, pa]
        elif op == 3:
            c = formula(1); value = ['imp', c, ['and', a, c]]
            proof = ['intro', c, ['pair', shift(pa, len(premises)), ['hyp', len(premises)]]]
        elif op == 4:
            c = formula(1); value, proof = ['imp', c, c], ['intro', c, ['hyp', len(premises)]]
        elif op == 5: value, proof = ['imp', ['and', a, b], a], ['lemma', 'project', a, b]
        else: value, proof = ['imp', a, a], ['lemma', 'identity', a, b]
        if len(fmt(value)) <= 90 and depth(proof) <= 9: values.append((value, proof))
    options = []
    for goal, proof in values:
        if goal in premises: continue
        proof = normalize(proof, len(premises)); ps, proof = _trim(premises, proof)
        theorem_depth = max(formula_depth(v) for v in [*ps, goal])
        if not ps or depth(proof) < (5 if harder else 2): continue
        if not harder and len(ps) < 2 and depth(proof) < 4: continue
        if (theorem_depth < 4) if harder else (theorem_depth > 3): continue
        if depth(proof) > (9 if harder else 5): continue
        options.append((ps, goal, proof, theorem_depth))
    if not options: return None
    # Prefer compositions using several assumptions without fixing a template.
    premises, goal, proof, theorem_depth = r.choices(options, weights=[(len(ps) + depth(p)) ** 3 for ps, _, p, _ in options], k=1)[0]
    def resample_leaves(f):
        return r.randrange(ATOMS) if type(f) is int else [f[0], resample_leaves(f[1]), resample_leaves(f[2])]
    # Match the whole connective tree rather than its outer operator alone.
    # Leaf assignments are independent: matching each atom's multiplicity would
    # prevent negative alternatives for basic identity theorems such as p→p.
    for _ in range(64):
        bad = resample_leaves(goal)
        model = countermodel(premises, bad)
        if model is not None: break
    else: return None
    if infer(proof, premises) != goal: raise ValueError('generator inference mismatch')
    sid = structure(premises, goal)
    return {'id': sid, 'structural_identity': sid, 'premises': premises, 'goal': goal, 'proof': proof,
            'distractor': bad, 'countermodel': model, 'seed': seed, 'depth': theorem_depth,
            'proof_depth': depth(proof), 'family': proof[0], 'dependencies': dependencies(proof)}


def render(row, answer):
    if answer not in ('A', 'B'): raise ValueError('answer must be A or B')
    r = random.Random(row['seed'])
    # Every atomic label has four ASCII bytes. Matched formula skeletons then
    # yield identical A/B byte lengths despite differing leaf multiplicities.
    names = r.sample(r.choice([['rain', 'snow', 'wind', 'mist'], ['ruby', 'blue', 'gold', 'teal'], ['oaks', 'elms', 'pine', 'firs']]), ATOMS)
    a, b = (row['goal'], row['distractor']) if answer == 'A' else (row['distractor'], row['goal'])
    order = list(range(len(row['premises']))); r.shuffle(order)
    question = r.choice(['Assume ', 'Given ', 'Suppose ', 'Known: ']) + '; '.join(fmt(row['premises'][i], names) for i in order)
    question += r.choice(['. Which must follow?\n', '. Choose the implied statement.\n', '. Which conclusion follows?\n'])
    question += f'A: {fmt(a, names)}\nB: {fmt(b, names)}\nAnswer:'
    proof = term(row['proof'], len(row['premises']))
    formal = ';'.join(fmt(p) for p in row['premises']) + '|-' + fmt(row['goal']) + ':' + proof
    row.update(question=question, answer=answer, canonical_proof=formal,
               solution=f'{question} {answer}\nProof: {proof}', lean_proof=proof,
               natural_language=question, choices={'A': a, 'B': b})
    row['lean_name'] = 'lf_' + row['id']
    binders = ' '.join(f'(h{i} : {fmt(p, lean=True)})' for i, p in enumerate(row['premises']))
    row['lean_declaration'] = f'theorem {row["lean_name"]} (p0 p1 p2 p3 : Prop) {binders} : {fmt(row["goal"], lean=True)} := {proof}\n'
    return row


def validate(row):
    if not isinstance(row, dict): raise ValueError('record must be an object')
    base = {'id', 'structural_identity', 'premises', 'goal', 'proof', 'distractor', 'countermodel', 'seed', 'depth', 'proof_depth', 'family', 'dependencies', 'question', 'answer', 'canonical_proof', 'solution', 'lean_proof', 'natural_language', 'choices', 'lean_name', 'lean_declaration'}
    if set(row) not in (base, base | {'split'}): raise ValueError('unknown or missing record fields')
    if type(row['seed']) is not int or not isinstance(row['premises'], list): raise ValueError('invalid record metadata')
    if infer(row['proof'], row['premises']) != row['goal']: raise ValueError('goal mismatch')
    if formula_shape(row['goal']) != formula_shape(row['distractor']): raise ValueError('choice connective shape mismatch')
    if structure(row['premises'], row['goal']) != row['id'] or row['structural_identity'] != row['id']: raise ValueError('structural mismatch')
    if dependencies(row['proof']) != row['dependencies']: raise ValueError('dependency mismatch')
    theorem_depth = max(formula_depth(f) for f in [*row['premises'], row['goal']])
    if row['depth'] != theorem_depth or row['proof_depth'] != depth(row['proof']) or row['family'] != row['proof'][0]: raise ValueError('depth or family mismatch')
    m = row['countermodel']
    if not all(truth(p, m) for p in row['premises']) or truth(row['distractor'], m): raise ValueError('invalid negative certificate')
    if not truth(row['goal'], m): raise ValueError('proof contradicts countermodel')
    if render(dict(row), row['answer']) != row: raise ValueError('unbound rendering')
    question = row['question']
    a = question.split('\nA: ', 1)[1].split('\nB: ', 1)[0]
    b = question.split('\nB: ', 1)[1].split('\nAnswer:', 1)[0]
    if len(a.encode()) != len(b.encode()): raise ValueError('choice byte-length mismatch')
    if 'split' in row:
        if row['split'] not in ('train', 'validation', 'test', 'harder', 'extension'): raise ValueError('unknown split')
        if (row['depth'] < 4 or row['proof_depth'] < 5) if row['split'] == 'harder' else row['depth'] > 3: raise ValueError('depth split mismatch')
        bucket = int(row['id'][:8], 16) % 10
        if ((row['split'] in ('train', 'extension') and bucket >= 8)
                or (row['split'] == 'validation' and bucket != 8)
                or (row['split'] == 'test' and bucket != 9)):
            raise ValueError('reserved structural split mismatch')


class _AdmissionLibrary(VerifiedLibrary):
    @staticmethod
    def _source(records):
        text = 'import Init\nset_option autoImplicit false\nset_option linter.unusedVariables false\n\n' + COMMON_SOURCE
        for name in ('lf_identity', 'lf_project'): text += f'#print axioms {name}\n'
        for record in records: text += record['lean_declaration'] + f"#print axioms {record['lean_name']}\n"
        return text

    @staticmethod
    def _audits(output, records):
        expected = ['lf_identity', 'lf_project'] + [r['lean_name'] for r in records]
        observed = re.findall(r"^'([^']+)' does not depend on any axioms\s*$", output, re.M)
        return Counter(observed) == Counter(expected) and 'sorryAx' not in output


def admit(root, rows, replay=False):
    rows = list(rows)
    for row in rows: validate(row)
    if len({r['id'] for r in rows}) != len(rows): raise ValueError('duplicate admission IDs')
    lib = _AdmissionLibrary(Path(root) / 'lean-cache')
    for offset in range(0, len(rows), 128):
        batch = [{'definition_name': r['lean_name'], 'definition': '', 'lean_name': r['lean_name'], 'lean_declaration': r['lean_declaration']} for r in rows[offset:offset + 128]]
        source = lib._source(batch)
        binding = {'schema': 2, 'foundation_hash': lib.foundation.hash, 'source_sha256': digest(source.encode()), 'row_ids': [r['lean_name'] for r in batch]}
        receipt = lib.root / 'audits' / (digest(binding) + '.json')
        if receipt.exists() and not replay:
            cached = read(receipt)
            if set(cached) != {'binding', 'verification'} or cached['binding'] != binding: raise ValueError('Lean cache batch mismatch')
            value = cached['verification']
            for kind in ('source', 'log'):
                path = lib.root / value[kind + '_path']
                if not path.resolve().is_relative_to((lib.root / 'audits').resolve()): raise ValueError('Lean cache path escape')
                if digest(path.read_bytes()) != value[kind + '_sha256']: raise ValueError('corrupt Lean cache')
            if ((lib.root / value['source_path']).read_text() != source or not lib._audits((lib.root / value['log_path']).read_text(), batch)
                    or value.get('status') != 'verified' or value.get('axioms') != [] or LEAN_COMMIT not in value.get('actual_version', '')
                    or 'version 4.35.0,' not in value.get('actual_version', '')
                    or value.get('source_sha256') != binding['source_sha256']
                    or value.get('audited_declarations') != len(batch) + 2): raise ValueError('unverified Lean admission')
            for field, checksum in [('compiler_path', 'compiler_sha256'), ('executable', 'launcher_sha256')]:
                binary = Path(value[field])
                if not binary.is_file() or digest(binary.read_bytes()) != value[checksum]:
                    raise ValueError('Lean cache compiler identity changed; replay required')
        else:
            value = lib._compile(batch, None)
            if value['status'] != 'verified': raise ValueError(f'Lean rejected batch: {value}')
            value['audited_declarations'] = len(batch) + 2
            write(receipt, {'binding': binding, 'verification': value})
    return {'records': len(rows), 'batches': (len(rows) + 127) // 128, 'pin': PIN, 'common_lemmas_audited_per_batch': 2}


def _fits(row, context):
    texts = [row['question'] + ' A', row['question'] + ' B', row['solution'], row['canonical_proof'], 'Proof:\n' + row['canonical_proof']]
    return all(len(s.encode()) + 2 <= context for s in texts) and len(row['canonical_proof'].encode()) >= 63


def _check_dataset(value, context):
    if (value.get('schema') != 2 or value.get('context') != context or value['fingerprint'] != digest(value['rows'])
            or value.get('common') != COMMON or value.get('common_training_examples') != 0): raise ValueError('dataset fingerprint mismatch')
    seen = set()
    for row in value['rows']:
        validate(row)
        if row['id'] in seen or not _fits(row, context): raise ValueError('duplicate or overlength record')
        seen.add(row['id'])


def generate(root, train=4096, validation=128, test=512, harder=512, context=512, extra=0, *, seed=10000, reserved_ids=()):
    """Freeze dataset.json; extensions return only new rows in separate artifacts."""
    if isinstance(train, dict):
        counts = train
        if set(counts) - {'train', 'validation', 'val', 'test', 'harder'}: raise ValueError('unknown split count')
        train, validation, test, harder = counts.get('train', 0), counts.get('validation', counts.get('val', 0)), counts.get('test', 0), counts.get('harder', 0)
    if any(type(v) is not int or v < 0 for v in (train, validation, test, harder, extra, seed)) or type(context) is not int or context < 128: raise ValueError('invalid generation settings')
    root, reserved = Path(root), set(reserved_ids)
    path = root / 'dataset.json'; old = read(path) if path.exists() else None
    if old is not None:
        _check_dataset(old, context)
        if admit(root, old['rows']) != old['admission']: raise ValueError('dataset admission metadata mismatch')
        if not extra: return old
    elif extra: raise ValueError('freeze a base dataset before extension')
    seen = reserved | ({r['id'] for r in old['rows']} if old else set())
    cursor = max(seed, old['next_seed'] if old else seed)
    previous = sorted((root / 'extensions').glob('extension-*.json'))
    for extension in previous:
        data = read(extension); _check_dataset(data, context)
        if data['parent_fingerprint'] != old['fingerprint']: raise ValueError('extension parent changed')
        if seen.intersection(r['id'] for r in data['rows']): raise ValueError('extension overlaps reserved theorem')
        seen.update(r['id'] for r in data['rows']); cursor = max(cursor, data['next_seed'])
    quotas = {'extension': extra} if extra else {'train': train, 'validation': validation, 'test': test, 'harder': harder}
    rows = []
    for split, count in quotas.items():
        added, attempts = 0, 0
        while added < count:
            attempts += 1
            if attempts > max(20000, count * 2000): raise ValueError(f'bounded generation exhausted for {split}: {added}/{count}')
            cursor += 1; row = candidate(cursor, split == 'harder')
            if row is None or row['id'] in seen: continue
            bucket = int(row['id'][:8], 16) % 10
            if split in ('train', 'extension') and bucket >= 8: continue
            if split == 'validation' and bucket != 8: continue
            if split == 'test' and bucket != 9: continue
            row = render(row, 'AB'[added % 2])
            if not _fits(row, context): continue
            row['split'] = split; validate(row)
            seen.add(row['id']); rows.append(row); added += 1
    admission = admit(root, rows)
    value = {'schema': 2, 'rows': rows, 'next_seed': cursor, 'context': context, 'seed': seed, 'common': COMMON, 'common_training_examples': 0, 'admission': admission,
             'split_policy': 'alpha/premise canonical IDs before rendering; buckets0..7 train,8 validation,9 test; theorem formula-depth<=3 ordinary and>=4 harder; unused hypotheses removed',
             'choice_policy': 'identical ordered connective trees; independent atom-leaf assignments with Boolean countermodel; four-byte atomic labels; equal displayed byte lengths',
             'reserved_ids_sha256': digest(sorted(reserved)), 'fingerprint': digest(rows)}
    if extra:
        value['parent_fingerprint'] = old['fingerprint']; destination = root / 'extensions' / f'extension-{len(previous):03d}.json'
    else: destination = path
    if destination.exists(): raise ValueError('refusing to overwrite immutable dataset')
    write(destination, value)
    return value
