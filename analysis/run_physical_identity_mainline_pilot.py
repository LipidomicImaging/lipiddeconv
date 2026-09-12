"""One frozen mainline pilot: production fits followed by MODEL-only identity certificates."""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

GAMMA = 1e-6
SEEDS = {'R71': 7201, 'R72': 7202}
KINDS = ('SUPPORTED_MISMATCH', 'close_neighbor', 'relatively_isolated')
PHYSICAL_FP = '3b2e75c563a0d23ac16db2ce87f1815ae0c3574fac59c3d8e89b794768697df7'
CONTRACT = dict(version='PHYSICAL_IDENTITY_MAINLINE_PILOT_V1', physical_contract=PHYSICAL_FP,
    K=125, mapping_seeds=SEEDS, fits=6, CAL='R71', EVAL='R72',
    donor_partition='all identity/source-file connected components; hashed groups in MODEL/MODEL/MODEL/CAL/EVAL slots',
    donor_ce_pair='shortest CE span, then numeric pair, with every role and both mapped support sets present',
    target='one full library per donor role; same pair; whole MODEL-excluded CAL/EVAL donor endpoints',
    missing='same original five omissions per arm; observations now combine supported mismatch and omission',
    evidence='unweighted original foreground mean; same observation for full and molecular deletion',
    inference='MODEL donors only plus nominal; all retained library candidates compete',
    full='nominal and each common-pair relaxation rounded once per identity, L1 refit, best physical feasible witness',
    deletion='all-MODEL endpoint cone relaxation; conservative global lower bound, physical replaceability not inferred',
    gamma_num=GAMMA, LP_gap_tolerance=1e-6, LP_proof_guard=1e-8,
    calibration='maximize CAL retained TP subject pooled FDP<=.01; ties fewer FP then larger epsilon',
    selection='physical full_upper<=epsilon-gamma and relaxation deleted_lower>epsilon+gamma',
    go='pooled FDP<=.01; overall and EACH challenge TP retention>=.40 and nonempty; supported perturbed truth retention overall and EACH challenge>=.40',
    strong_success='GO and aggregate TP retention>=.60',
    raw_reporting='any candidate foreground mean>0.001; individual lipid_name identity',
    supported_truth='truth with candidate in supported mapping AND generated relative column change>1e-10; defined before fits',
    uncertainty_boundary='censoring NOT_ESTIMATED; pixel NOT_SEPARATELY_IDENTIFIABLE; no added noise',
    full_relaxation_gap='model relaxation uncertainty; not the per-LP numerical proof gap',
    no_new_rho=True, no_new_spatial_variant=True, no_outcome_tuning=True,
    process='unchanged production early-stop, 3000 hard cap; normal finite completion only',
    run_storage='all final arrays/models/proofs retained; required-case handoff ACK before next fit',
    eval_guard='no EVAL confidence computation/accounting before CAL threshold seal',
    standalone_real_data_claim=False)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def ah(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def membership():
    return [dict(key=f'{kind}__{pool}_{rep}_K125', kind=kind, base=f'{pool}_{rep}_K125',
                 role=role, replicate=rep) for rep, role in [('R71', 'CAL'), ('R72', 'EVAL')]
            for kind, pool in [('SUPPORTED_MISMATCH', 'CAL'), ('close_neighbor', 'HOLD'), ('relatively_isolated', 'HOLD')]]


@contextmanager
def mapping_namespace(v57):
    original = v57.MAPPING_SEEDS
    v57.MAPPING_SEEDS = dict(SEEDS)
    try:
        yield
    finally:
        v57.MAPPING_SEEDS = original


def verify_files(root, files):
    for name, digest in files.items():
        require(sha(Path(root)/name) == digest, 'ARTIFACT_CHANGED:' + name)


def physical(args):
    base = args.snapshot/'results/physical_perturbation_contract_v1'
    seal = read(base/'seal.json')
    require(fp(seal['content']) == seal['fingerprint'] == PHYSICAL_FP, 'PHYSICAL_CONTRACT_CHANGED')
    for row in seal['content']['inputs']:
        require(sha(args.snapshot/row['path']) == row['sha256'], 'PHYSICAL_SOURCE_CHANGED:' + row['path'])
    verify_files(base, seal['content']['outputs'])
    return read(base/'joint_patterns.json'), read(base/'candidate_mapping.json')


def sources(args):
    sys.path[1:1] = [str(args.root/'analysis'), str(args.root/'src')]
    import numpy as np
    import torch
    import run_v58_spectral_library_mismatch_fdr_recalibration as v
    import run_missing_library_challenge as missing
    from config_758 import Cfg
    from physical_identity_pilot_design import build_donor_split
    v.dependencies(); torch.set_num_threads(1)
    va = v.parse_args(['--audit-design', '--v57-output', str(args.root/'results/v57_spectral_spatial_identity_confidence_benchmark'),
                       '--asset-root', str(args.asset_root)])
    parent, _, parent_hashes = v.parent_provenance(va)
    context = v.load_context(va, parent); context['units'] = parent['scientific']['matched_units']
    for key, value in {**v.v54.v50.CFG_DEFAULTS, 'parent_channel_weight_multiplier': 1.0}.items():
        require(getattr(Cfg, key) == value, 'PRODUCTION_CONFIG_CHANGED:' + key)
    require(v.v54.MAX_EPOCH == 3000 and tuple(v.v54.CHECKPOINT_EPOCHS) == (1000,1500,2000,2500,3000), 'EPOCH_CONTRACT_CHANGED')
    for key in ('ISTA_WEIGHT_REFERENCE_A_PATH', 'ISTA_WEIGHT_REFERENCE_META_PATH'):
        require(not os.environ.get(key), 'EXTERNAL_WEIGHT_REFERENCE_FORBIDDEN')
    old_missing = args.root/'results/missing_library_challenge_k125'
    omission = read(old_missing/'execution_design.json')
    require(fp(omission['scientific']) == omission['fingerprint'] == 'b51137f3c6f12cf9bdf0e3e9f1c00c1fff53f9e4d62dc1c54de0b04a7910d9ef', 'OMISSION_CHANGED')
    require(read(old_missing/'execution_freeze.json')['design_sha256'] == sha(old_missing/'execution_design.json'), 'OMISSION_SEAL_CHANGED')
    patterns, mapping = physical(args)
    records_path = args.snapshot/'results/rho_mismatch_mechanism_ce_audit/ce_identity_records.json'
    donors = build_donor_split(patterns, read(records_path))
    with np.load(args.snapshot/'results/ce133_uncertainty_v1_ready/component_fractions.npz') as z:
        fractions = z['fractions'].copy()
    implementations = {**v.implementation_hashes(), 'run_missing_library_challenge.py': sha(Path(missing.__file__))}
    for name in ('run_physical_identity_mainline_pilot.py', 'physical_identity_pilot_design.py',
                 'physical_identity_confidence_core.py', 'run_ce_uncertainty_identity_pilot.py'):
        implementations[name] = sha(Path(__file__).parent/name)
    binding = dict(parent_fingerprint=parent['design_fingerprint'], parent_files=parent_hashes,
        omission_design_sha256=sha(old_missing/'execution_design.json'), physical_fingerprint=PHYSICAL_FP,
        donor_records_sha256=sha(records_path), donor_split_fingerprint=fp(donors), implementation=implementations,
        A_solver_sha256=ah(context['A_solver']),
        training_config={k: getattr(Cfg, k) for k in v.v54.v50.CFG_DEFAULTS},
        production_training=v.v54.v50.training_contract())
    return dict(np=np, v=v, missing=missing, context=context, parent=parent, arms=omission['scientific']['arms'],
                patterns=patterns, mapping=mapping, fractions=fractions, donors=donors, binding=binding)


def case_for(pack, spatial, entry, target):
    np, v, context = pack['np'], pack['v'], pack['context']
    # The original constructor applies its ONE global scalar using the new generative A.
    # Original nominal context is separately retained for every solver input.
    with mapping_namespace(v.v57):
        case = v.v57.construct_case({**context, 'A_solver': target}, spatial, entry['base'])
    if entry['kind'] == 'SUPPORTED_MISMATCH':
        reduced, kept = context, np.arange(391)
    else:
        arm = pack['arms'][entry['kind']]
        reduced, kept = pack['missing'].reduced_context(context, arm, np)
        require(set(arm['removed_original_indices']) <= set(case['active_indices']), 'OMISSION_NOT_TRUE')
    case['case_name'] = 'PHYSICAL_IDENTITY_MAINLINE_V1__' + entry['key']
    return case, reduced, kept


def prepare(args):
    require(not args.output.exists(), 'OUTPUT_EXISTS_NO_REDRAW')
    pack = sources(args); np, v, context = pack['np'], pack['v'], pack['context']
    from physical_identity_pilot_design import build_target_library
    from physical_identity_confidence_core import build_model_bank
    spatial = copy.deepcopy(pack['parent']['scientific'])
    _, bases = v.v57.healthy_pool(context)
    require(sorted(bases) == spatial['healthy_indices'], 'HEALTHY_POOL_CHANGED')
    context['bases'] = bases
    with mapping_namespace(v.v57):
        assignments = v.v57.spatial_assignments(context, spatial['spectral_pair_design'], bases,
            spatial['healthy_pair_correlations'], spatial['spatial_low_q25'])
    require(not ({r['map_sha256'] for r in assignments} &
                 {r['map_sha256'] for r in spatial['spatial_assignments']}), 'OLD_SPATIAL_MAP_REUSED')
    spatial['spatial_assignments'] = assignments
    args.output.mkdir(parents=True)
    write(args.output/'donor_split.json', pack['donors'])
    model_ids = sorted(pid for pid, role in pack['donors']['pattern_roles'].items() if role == 'MODEL')
    model_patterns = [r for r in pack['patterns'] if r['pattern_id'] in set(model_ids)]
    write(args.output/'model_patterns.json', model_patterns)
    np.save(args.output/'A_solver.npy', context['A_solver'])
    # MODEL bank construction inspects only nominal A and MODEL patterns; no B or outcomes.
    bank = build_model_bank(context['A_solver'].astype(float), pack['fractions'], model_patterns,
                           pack['mapping'], model_ids, [{'lipid_name': str(n)} for n in context['metadata']['lipid_name']], list(range(391)))
    frozen = dict(contract=CONTRACT, source_binding=pack['binding'], membership=membership(),
                  spatial_design=spatial, model_pattern_ids=model_ids, donor_split=pack['donors'], targets={}, cases={},
                  A_solver_file_sha256=sha(args.output/'A_solver.npy'), model_patterns_sha256=sha(args.output/'model_patterns.json'))
    targets = {}
    for role in ('CAL', 'EVAL'):
        target, target_binding = build_target_library(context['A_solver'], pack['fractions'], pack['patterns'],
            pack['mapping'], pack['donors'], role, context['metadata'])
        targets[role] = target
        np.save(args.output/f'A_target_{role}.npy', target)
        frozen['targets'][role] = dict(binding=target_binding, array_sha256=ah(target),
                                      file_sha256=sha(args.output/f'A_target_{role}.npy'))
    require(not np.array_equal(targets['CAL'], targets['EVAL']), 'TARGETS_NOT_NEW_NO_REDRAW')
    # No target is passed to the inference engine. Input-only containment check is below,
    # independently reconstructed from MODEL endpoint transforms in prepare only.
    containment = target_containment(pack, targets, model_ids)
    require(containment['status'] == 'PASS', 'TARGET_IN_INFERENCE_REPRESENTATION_ONLY_NO_GO')
    write(args.output/'target_containment.json', containment)
    for entry in membership():
        case, reduced, kept = case_for(pack, spatial, entry, targets[entry['role']])
        require(case['global_truth_K'] == 125, 'TRUTH_K_CHANGED')
        dest = args.output/'cases'/entry['key']; dest.mkdir(parents=True)
        np.savez_compressed(dest/'observation_truth.npz', B=case['B_sim'], X_true=case['X_true'], mask=case['foreground_mask'])
        change = np.linalg.norm(targets[entry['role']].astype(float)-context['A_solver'].astype(float), axis=0)
        change /= np.linalg.norm(context['A_solver'].astype(float), axis=0)
        perturbed = sorted(set(case['active_indices']) & {r['candidate_index'] for r in pack['mapping'] if change[r['candidate_index']] > 1e-10})
        require(len(perturbed) > 0, 'NO_SUPPORTED_PERTURBED_TRUTH')
        info = dict(**entry, B_sha256=ah(case['B_sim']), X_true_sha256=ah(case['X_true']),
            A_solver_sha256=ah(reduced['A_solver']), A_target_sha256=ah(targets[entry['role']]),
            observation_file_sha256=sha(dest/'observation_truth.npz'), kept=kept.tolist(),
            truth_indices=case['active_indices'], reportable_truth_indices=case['reportable_truth_indices'],
            supported_perturbed_truth_indices=perturbed, global_scale=case['global_scale'],
            achieved_foreground_signal_norm=case['achieved_foreground_B_l2_p50'])
        write(dest/'input.json', info); frozen['cases'][entry['key']] = info
    write(args.output/'design.json', dict(fingerprint=fp(frozen), scientific=frozen))
    write(args.output/'design_seal.json', dict(status='FROZEN_BEFORE_TRAINING_AND_CONFIDENCE',
        fingerprint=fp(frozen), design_sha256=sha(args.output/'design.json'),
        target_containment_sha256=sha(args.output/'target_containment.json')))
    print('DESIGN_FROZEN', fp(frozen), flush=True)


def target_containment(pack, targets, model_ids):
    np = pack['np']; A = pack['context']['A_solver'].astype(float)
    rows = []; ids = set(model_ids)
    for mapped in pack['mapping']:
        j = mapped['candidate_index']; D = pack['fractions'][:, mapped['component_indices']] * A[:, j, None]
        fixed = A[:, j]-D.sum(axis=1); options = [A[:, j]]
        for pattern in pack['patterns']:
            if pattern['pattern_id'] not in ids or pattern['channels'] != mapped['channels'] or pattern['donor_identity'][2] != mapped['rule']:
                continue
            t = fixed + D @ np.asarray(pattern['multiplier'])
            options.append(t*np.linalg.norm(A[:, j])/np.linalg.norm(t))
        options = np.stack(options)
        for role, target in targets.items():
            distance = float(np.min(np.linalg.norm(options-target[:, j].astype(float), axis=1))/np.linalg.norm(A[:, j]))
            changed = float(np.linalg.norm(target[:, j].astype(float)-A[:, j])/np.linalg.norm(A[:, j]))
            rows.append(dict(candidate_index=j, role=role, minimum_relative_distance_to_MODEL_endpoint=distance,
                             actual_relative_change=changed, same_endpoint=bool(changed > 1e-10 and distance <= 1e-7)))
    return dict(status='FAIL' if any(r['same_endpoint'] for r in rows) else 'PASS', tolerance=1e-7,
                reason='input-only same-candidate endpoint exclusion, not an identity-performance result', rows=rows)


def validate(args, pack=None):
    d = read(args.output/'design.json'); seal = read(args.output/'design_seal.json')
    require(d['fingerprint'] == fp(d['scientific']) == seal['fingerprint'], 'DESIGN_CONTENT_CHANGED')
    require(seal['design_sha256'] == sha(args.output/'design.json'), 'DESIGN_FILE_CHANGED')
    require(d['scientific']['contract'] == json.loads(json.dumps(CONTRACT)), 'PILOT_CONTRACT_CHANGED')
    require(d['scientific']['membership'] == membership(), 'MEMBERSHIP_CHANGED')
    require(seal['target_containment_sha256'] == sha(args.output/'target_containment.json'), 'CONTAINMENT_CHANGED')
    require(d['scientific']['A_solver_file_sha256'] == sha(args.output/'A_solver.npy'), 'NOMINAL_A_FILE_CHANGED')
    require(d['scientific']['model_patterns_sha256'] == sha(args.output/'model_patterns.json'), 'MODEL_PATTERNS_CHANGED')
    if pack is not None:
        require(d['scientific']['source_binding'] == pack['binding'], 'SOURCE_IMPLEMENTATION_CHANGED')
    return d


def train_case(args):
    pack = sources(args); d = validate(args, pack); np, v, context = pack['np'], pack['v'], pack['context']
    entries = membership(); entry = next(e for e in entries if e['key'] == args.case)
    prior = entries[:entries.index(entry)]
    for previous in prior:
        ack = read(args.output/'cases'/previous['key']/'handoff_ack.json')
        require(ack['status'] == 'VERIFIED_DOWNLOADED_AND_PUSHED' and ack['case'] == previous['key'], 'PRIOR_CASE_HANDOFF_REQUIRED')
        require(ack['training_complete_sha256'] == sha(args.output/'cases'/previous['key']/'training_complete.json'), 'HANDOFF_CASE_CHANGED')
    role = entry['role']; target_path = args.output/f'A_target_{role}.npy'
    require(sha(target_path) == d['scientific']['targets'][role]['file_sha256'], 'TARGET_CHANGED')
    target = np.load(target_path)
    case, reduced, kept = case_for(pack, d['scientific']['spatial_design'], entry, target)
    dest = args.output/'cases'/entry['key']; info = d['scientific']['cases'][entry['key']]
    require(ah(case['B_sim']) == info['B_sha256'] and ah(case['X_true']) == info['X_true_sha256'], 'GENERATIVE_CASE_CHANGED')
    require(ah(reduced['A_solver']) == info['A_solver_sha256'], 'SOLVER_LIBRARY_CHANGED')
    require(sha(dest/'observation_truth.npz') == info['observation_file_sha256'], 'OBSERVATION_CHANGED')
    training = dest/'training'
    pack['missing'].bind_runtime(training, dict(design_fingerprint=d['fingerprint'], case=entry['key'], input=info))
    if (dest/'training_complete.json').exists():
        verify_files(dest, read(dest/'training_complete.json')['files']); print('CACHED_FIT_VERIFIED', entry['key']); return
    require(shutil.disk_usage(args.output).free > 1400*1024**2, 'STOP_STORAGE_BEFORE_FIT')
    write(args.output/'status.json', dict(status='TRAINING', case=entry['key']))
    learned = v.v54.train_resumable({k: case[k] for k in ('case_name', 'B_sim', 'foreground_mask')},
                                   v.source_training_context(reduced), training, device_name=args.device)
    require(all(v.solver_validity(learned)) and v.finite_loss_history(read(training/'training_history.json')), 'ABNORMAL_OR_NONFINITE_FIT')
    require(learned['X_hat'].shape == (len(kept), *case['foreground_mask'].shape), 'LEARNED_SHAPE_CHANGED')
    np.savez_compressed(training/'learned_arrays.npz', X_hat=learned['X_hat'], B_hat=learned['B_hat'])
    write(training/'solver_run.json', {k: val for k, val in learned.items() if k not in ('X_hat', 'B_hat')})
    names = [str(n) for n in context['metadata']['lipid_name']]
    truth = {names[i] for i in info['truth_indices']}; reportable = {names[i] for i in info['reportable_truth_indices']}
    perturbed = {names[i] for i in info['supported_perturbed_truth_indices']}
    means = learned['X_hat'][:, case['foreground_mask']].mean(axis=1, dtype=float)
    candidates = []; inverse = {int(j): i for i, j in enumerate(kept)}
    for j, name in enumerate(names):
        local = inverse.get(j); abundance = float(means[local]) if local is not None else 0.
        candidates.append(dict(candidate_index=j, reduced_index=local, lipid_name=name,
            candidate_id=str(context['metadata']['candidate_id'][j]), lipid_class=str(context['metadata']['lipid_class'][j]),
            X_hat=abundance, raw_solver_reported=bool(local is not None and abundance > .001),
            molecular_truth=name in truth, reportable_truth=name in reportable, supported_perturbed_truth=name in perturbed,
            omitted_from_solver=local is None, K=125, replicate=entry['replicate'], split=entry['role']))
    molecular = []
    for i, name in enumerate(dict.fromkeys(names)):
        rows = [r for r in candidates if r['lipid_name'] == name]
        molecular.append(dict(lipid_name=name, evidence_index=i, X_hat=sum(r['X_hat'] for r in rows),
            raw_solver_reported=any(r['raw_solver_reported'] for r in rows), molecular_truth=name in truth,
            reportable_truth=name in reportable, supported_perturbed_truth=name in perturbed,
            removed=[r['reduced_index'] for r in rows if r['reduced_index'] is not None]))
    b = case['B_sim'][:, case['foreground_mask']].mean(axis=1, dtype=float)
    np.savez_compressed(dest/'evidence.npz', A=reduced['A_solver'].astype(float), kept=kept, global_b=b)
    write(dest/'molecular_records.json', molecular); write(dest/'candidate_records.json', candidates)
    write(dest/'metadata.json', {k: [str(x) for x in context['metadata'][k]]
                                for k in ('lipid_name', 'candidate_id', 'lipid_class')})
    files = {p.relative_to(dest).as_posix(): sha(p) for p in training.iterdir() if p.is_file()}
    files.update({n: sha(dest/n) for n in ('input.json','observation_truth.npz','evidence.npz','molecular_records.json','candidate_records.json','metadata.json')})
    write(dest/'training_complete.json', dict(status='NORMAL_FINITE_COMPLETE', case=entry['key'],
        fingerprint=d['fingerprint'], files=files, stopped_epoch=learned['stopped_epoch'], stop_reason=learned['stop_reason'],
        training_uses_no_truth_or_target_library=True, outcomes_NOT_reviewed=True))
    write(args.output/'status.json', dict(status='CASE_COMPLETE_AWAITING_REVIEW_GIT_HANDOFF', case=entry['key']))
    print('FIT_COMPLETE_AWAITING_HANDOFF', entry['key'], flush=True)


def classify(row, epsilon):
    full, deleted = row['full'], row.get('deleted')
    if full['status'] != 'BOUNDS_VALID' or not deleted or deleted['status'] != 'BOUNDS_VALID':
        return 'NUMERICALLY_UNRESOLVED'
    if full['lower'] > epsilon + GAMMA:
        return 'FULL_MODEL_INCOMPATIBLE'
    if full['upper'] > epsilon - GAMMA:
        return 'THRESHOLD_UNRESOLVED_FULL'
    if deleted['lower'] > epsilon + GAMMA:
        return 'RETAINED'
    if deleted['upper'] <= epsilon - GAMMA:
        return 'RELAXATION_REPLACEABLE'
    return 'THRESHOLD_UNRESOLVED'


def metrics(cases, epsilon, supported_only=False):
    from collections import Counter
    all_rows = [r for c in cases for r in c['records']]
    selected = [r for r in all_rows if classify(r, epsilon) == 'RETAINED']
    truth_flag = 'supported_perturbed_truth' if supported_only else 'molecular_truth'
    denominator = 'supported_perturbed_truth_count' if supported_only else 'truth_count'
    truth = sum(c[denominator] for c in cases)
    raw_tp = sum(r[truth_flag] for r in all_rows); tp = sum(r[truth_flag] for r in selected)
    # Risk is always calculated on the FULL retained set, never by assigning true-only subsets a fake FDR.
    fp_count = sum(not r['molecular_truth'] for r in selected)
    reportable = sum(c['reportable_truth_count'] for c in cases)
    return dict(raw_solver_TP=raw_tp, raw_solver_FP=None if supported_only else len(all_rows)-raw_tp,
        raw_solver_FN=truth-raw_tp, filtered_TP=tp, filtered_FP=None if supported_only else fp_count,
        filtered_FN=truth-tp, filter_induced_true_loss=raw_tp-tp,
        filter_induced_true_loss_fraction=(raw_tp-tp)/raw_tp if raw_tp else None,
        TP_retention=tp/raw_tp if raw_tp else None, all_truth_recall=tp/truth if truth else None,
        reportable_truth_recall=None if supported_only else sum(r['reportable_truth'] for r in selected)/reportable if reportable else None,
        FDP=None if supported_only else fp_count/len(selected) if selected else None,
        retained_count=len(selected) if not supported_only else tp, truth_count=truth,
        distinct_identities=len({r['lipid_name'] for r in selected if not supported_only or r[truth_flag]}),
        status_counts=dict(Counter(classify(r, epsilon) for r in all_rows if not supported_only or r[truth_flag])),
        risk_scope='NOT_APPLICABLE_TRUE_SUBSET' if supported_only else 'all reported molecular identities')


def calibrated(cases):
    import numpy as np
    events = {0., 1.+GAMMA}
    for c in cases:
        for r in c['records']:
            if r['full']['status'] != 'BOUNDS_VALID' or r['deleted']['status'] != 'BOUNDS_VALID':
                continue
            for value in (r['full']['upper']+GAMMA, r['deleted']['lower']-GAMMA):
                if value >= 0:
                    events.update([float(value), float(np.nextafter(value, -np.inf)), float(np.nextafter(value, np.inf))])
    options = [dict(epsilon=e, **metrics(cases, e)) for e in sorted(events) if e >= 0]
    eligible = [v for v in options if v['retained_count'] and v['FDP'] <= .01]
    best = max(eligible, key=lambda v: (v['filtered_TP'], -v['filtered_FP'], v['epsilon'])) if eligible else dict(epsilon=1.+GAMMA, **metrics(cases, 1.+GAMMA))
    return dict(status='CALIBRATED' if eligible else 'EMPTY_CALIBRATION', epsilon=best['epsilon'], chosen=best, curve=options)


def go_rule(cases, epsilon):
    aggregate = metrics(cases, epsilon)
    supported = metrics(cases, epsilon, True)
    by = {kind: dict(all_truth=metrics([c for c in cases if c['kind'] == kind], epsilon),
                    supported_perturbed_truth=metrics([c for c in cases if c['kind'] == kind], epsilon, True)) for kind in KINDS}
    okay = lambda x: x['retained_count'] > 0 and x['TP_retention'] is not None and x['TP_retention'] >= .4
    go = bool(okay(aggregate) and aggregate['FDP'] <= .01 and okay(supported) and
              all(okay(row['all_truth']) and okay(row['supported_perturbed_truth']) for row in by.values()))
    return dict(aggregate=aggregate, supported_perturbed_truth=supported, by_challenge=by,
                go=go, strong_success=bool(go and aggregate['TP_retention'] >= .6))


def init_score_worker(bank, b):
    global SCORE_BANK, SCORE_OBSERVATION
    SCORE_BANK, SCORE_OBSERVATION = bank, b


def score_identity(name):
    score, proof = SCORE_BANK.score_delete(name)
    SCORE_BANK.verify_delete(SCORE_OBSERVATION, name, score, proof)
    return score, proof


def short_bounds(score):
    return {k: score.get(k) for k in ('status', 'lower', 'upper', 'numerical_error')}


def score_case(args, d, entry):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from physical_identity_confidence_core import build_model_bank
    source = args.output/'cases'/entry['key']; dest = args.output/'scores'/entry['key']; dest.mkdir(parents=True, exist_ok=True)
    complete = read(source/'training_complete.json')
    for name in ('evidence.npz', 'metadata.json', 'molecular_records.json'):
        require(sha(source/name) == complete['files'][name], 'SCORE_INPUT_CHANGED:' + name)
    with np.load(source/'evidence.npz') as z:
        A, kept, b = z['A'].copy(), z['kept'].tolist(), z['global_b'].copy()
    with np.load(args.snapshot/'results/ce133_uncertainty_v1_ready/component_fractions.npz') as z:
        fractions = z['fractions'].copy()
    metadata = [{'lipid_name': n} for n in read(source/'metadata.json')['lipid_name']]
    # Deliberate inference boundary: no target library, target assignment or truth passed.
    model_patterns = read(args.output/'model_patterns.json')
    mapping = read(args.snapshot/'results/physical_perturbation_contract_v1/candidate_mapping.json')
    bank = build_model_bank(A, fractions, model_patterns, mapping, d['scientific']['model_pattern_ids'], metadata, kept)
    binding = dict(design_fingerprint=d['fingerprint'], evidence_sha256=sha(source/'evidence.npz'),
        records_sha256=sha(source/'molecular_records.json'), bank_fingerprint=bank.fingerprint,
        evidence_seal_sha256=sha(args.output/'evidence_seal.json'),
        evaluation_access_sha256=sha(args.output/'evaluation_access.json') if entry['role'] == 'EVAL' else None)
    full_path = dest/'full.json'
    if full_path.exists():
        cached = read(full_path); require(cached['binding'] == binding, 'FULL_CACHE_BINDING_CHANGED')
        require(read(dest/'bank_manifest.json') == bank.manifest, 'BANK_MANIFEST_CHANGED')
        require(sha(dest/'full_proof.npz') == cached['proof_sha256'], 'FULL_PROOF_CHANGED')
        with np.load(dest/'full_proof.npz') as z:
            full_proof = {k: z[k].copy() for k in z.files}
        full = cached['result']; bank.restore_full(b, full, full_proof)
    else:
        full, full_proof = bank.score_full(b); bank.verify_full(b, full, full_proof)
        np.savez_compressed(dest/'full_proof.npz', **full_proof)
        write(dest/'bank_manifest.json', bank.manifest)
        write(full_path, dict(binding=binding, result=full, proof_sha256=sha(dest/'full_proof.npz')))
    all_rows = read(source/'molecular_records.json'); reported = [r for r in all_rows if r['raw_solver_reported']]
    output = []; proof_files = {'full_proof.npz': sha(dest/'full_proof.npz')}; pending = []
    for row in reported:
        index = row['evidence_index']; checkpoint = dest/f'record_{index:04d}.json'
        if checkpoint.exists():
            item = read(checkpoint); require(item['binding'] == binding, 'DELETION_CACHE_BINDING_CHANGED')
            require(item['input_record'] == row, 'DELETION_INPUT_RECORD_CHANGED')
            require(item['full_sha256'] == sha(full_path), 'DELETION_FULL_CHANGED')
            verify_files(dest, item['proof_files'])
            with np.load(dest/item['proof_file']) as z:
                proof = {k: z[k].copy() for k in z.files}
            bank.verify_delete(b, row['lipid_name'], item['deletion_result'], proof)
            reconstructed = dict(row, full=short_bounds(full), deleted=short_bounds(item['deletion_result']))
            require(item['record'] == reconstructed, 'CACHED_RESULT_NOT_DERIVED_FROM_PROOF')
            proof_files.update(item['proof_files']); output.append(reconstructed)
        else:
            pending.append(row)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'),
                             initializer=init_score_worker, initargs=(bank, b)) as pool:
        futures = [(row, pool.submit(score_identity, row['lipid_name'])) for row in pending]
        for number, (row, future) in enumerate(futures, 1):
            deleted, proof = future.result()
            record = dict(row, full=short_bounds(full), deleted=short_bounds(deleted))
            index = row['evidence_index']; name = f'proof_{index:04d}.npz'
            np.savez_compressed(dest/name, **proof); digest = sha(dest/name)
            item = dict(binding=binding, input_record=row, record=record, full_sha256=sha(full_path),
                        deletion_result=deleted, proof_file=name, proof_files={name: digest})
            write(dest/f'record_{index:04d}.json', item)
            proof_files[name] = digest; output.append(record)
            if number % 25 == 0 or number == len(futures):
                write(dest/'progress.json', dict(completed=len(output), total=len(reported), case=entry['key']))
                print('SCORE_PROGRESS', entry['key'], len(output), len(reported), flush=True)
    result = dict(**entry, binding=binding, full_sha256=sha(full_path),
        truth_count=sum(r['molecular_truth'] for r in all_rows), reportable_truth_count=sum(r['reportable_truth'] for r in all_rows),
        supported_perturbed_truth_count=sum(r['supported_perturbed_truth'] for r in all_rows),
        records=sorted(output, key=lambda r: r['evidence_index']), proof_files=proof_files)
    write(dest/'scores.json', result)
    return result


def score(args):
    pack = sources(args); d = validate(args, pack)
    for entry in membership():
        dest = args.output/'cases'/entry['key']
        ack = read(dest/'handoff_ack.json')
        require(ack['status'] == 'VERIFIED_DOWNLOADED_AND_PUSHED' and
                ack['training_complete_sha256'] == sha(dest/'training_complete.json'), 'CASE_HANDOFF_REQUIRED')
    seals = {entry['key']: {name: sha(args.output/'cases'/entry['key']/name)
             for name in ('training_complete.json', 'evidence.npz', 'molecular_records.json')} for entry in membership()}
    evidence_seal = dict(design_fingerprint=d['fingerprint'], cases=seals)
    path = args.output/'evidence_seal.json'
    if path.exists(): require(read(path) == evidence_seal, 'EVIDENCE_SEAL_CHANGED')
    else: write(path, evidence_seal)
    write(args.output/'status.json', dict(status='SCORING_CAL'))
    cal = [score_case(args, d, e) for e in membership() if e['role'] == 'CAL']
    threshold = calibrated(cal)
    threshold.update(design_fingerprint=d['fingerprint'], physical_contract=PHYSICAL_FP,
        model_patterns_sha256=sha(args.output/'model_patterns.json'),
        CAL_source_hashes={c['key']: sha(args.output/'scores'/c['key']/'scores.json') for c in cal})
    path = args.output/'calibration_seal.json'
    if path.exists(): require(read(path) == threshold, 'CALIBRATION_SEAL_CHANGED')
    else: write(path, threshold)
    access = dict(status='CAL_SEALED_BEFORE_EVAL_CONFIDENCE', design_fingerprint=d['fingerprint'],
                  calibration_seal_sha256=sha(path))
    if (args.output/'evaluation_access.json').exists(): require(read(args.output/'evaluation_access.json') == access, 'EVAL_ACCESS_CHANGED')
    else: write(args.output/'evaluation_access.json', access)
    write(args.output/'status.json', dict(status='SCORING_EVAL', calibration_seal_sha256=sha(path)))
    evaluated = [score_case(args, d, e) for e in membership() if e['role'] == 'EVAL']
    require(sha(path) == access['calibration_seal_sha256'], 'CALIBRATION_CHANGED_DURING_EVAL')
    epsilon = threshold['epsilon']; result = go_rule(evaluated, epsilon)
    result.update(status='COMPLETE_AWAITING_INDEPENDENT_REVIEW', epsilon=epsilon,
        calibration=metrics(cal, epsilon), CAL_supported=metrics(cal, epsilon, True),
        by_case={c['key']: dict(all_truth=metrics([c], epsilon), supported=metrics([c], epsilon, True)) for c in cal+evaluated},
        independent_real_experiment_validation=False)
    for case in cal + evaluated:
        write(args.output/'scores'/case['key']/'classified_records.json',
              [dict(r, classification=classify(r, epsilon)) for r in case['records']])
    write(args.output/'summary.json', result)
    write(args.output/'status.json', dict(status='COMPLETE_AWAITING_INDEPENDENT_REVIEW', go=result['go']))
    print('MAINLINE_PILOT_COMPLETE', json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'train-case', 'score'))
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/lipiddeconv'))
    parser.add_argument('--asset-root', type=Path, default=Path('/root/autodl-tmp/decon-lipid'))
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=[e['key'] for e in membership()])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, choices=(1,4), default=4)
    args = parser.parse_args()
    if args.action == 'train-case' and not args.case:
        parser.error('train-case requires --case')
    try:
        {'prepare': prepare, 'train-case': train_case, 'score': score}[args.action](args)
    except BaseException as exc:
        if args.output.exists():
            write(args.output/'failure.json', dict(action=args.action, case=args.case, error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
