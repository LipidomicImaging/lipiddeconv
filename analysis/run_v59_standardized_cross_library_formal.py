#!/usr/bin/env python3
"""V59: 18 CLEAN standardized cross-library datasets; explicit gated lifecycle.

No production code is modified. Preparation and tests never train a model.
Three sentinel datasets are members of the 18 formal datasets, not extra runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'src'))
VERSION = 'v59_standardized_cross_library_formal'
DOMAINS = {'D0': ('W0_STD', 748, 798), 'D1': ('W800_850', 800, 850), 'D2': ('W850_900', 850, 900)}
REPLICATES = ('R1', 'R2', 'R3')
K = 125
SENTINELS = tuple(f'{d}__CAL_R1_K125' for d in DOMAINS)
CLEAN_FIXED = {'FDR5': 1.272651000158792e-20, 'FDR1': 6.683949277609072e-20}
CONTRACT = {
    'version': VERSION, 'domains': DOMAINS, 'K': K, 'replicates': REPLICATES,
    'external_scope': [700, 900], 'formal_runs': 18, 'sentinels': SENTINELS,
    'split_seed_by_domain': {'D0': 5900, 'D1': 5901, 'D2': 5902},
    'split_rule': 'source singleton columns sorted by (lipid_name,candidate_id); PCG64 permutation; first125 CAL, next125 HOLD; fixed across replicates',
    'spatial_rule': 'reuse V57 frozen K125 unit slots, R1-R3 maps and bounded abundance assignment; no additional template normalization or identity rescaling',
    'spatial_role_interpretation': 'inherited spatial high/low structure only; old spectral hard/easy labels do not classify new identities',
    'signal_target': .6036783456802368, 'report_gate': .001,
    'synthesis': 'A_synthesis=A_solver=A_std, full candidate library; one global scalar per dataset',
    'measurement_noise': False, 'spectral_mismatch': False, 'new_tissue': False,
    'sentinel_max_residual': .10, 'oracle_max_residual': 1e-6,
    'oracle': 'independent full-library foreground-mean NNLS; candidate and molecular FP=FN=0; not a uniqueness proof',
    'hard_epoch_cap': 3000, 'checkpoint_epochs': [1000, 1500, 2000, 2500, 3000],
    'early_stop': 'unchanged production semantics; no minimum sentinel duration',
    'primary': 'domain-local CAL maximum retained molecular units at empirical FDR<=alpha; ties lower threshold',
    'molecular_rule': 'max rho_zero of reported candidates; sum reported candidate X_hat; no group-rho',
    'CLEAN_FIXED': CLEAN_FIXED, 'geometry_role': 'descriptive only; no window selection',
    'low_retention_warning': 'N_retained=0 or TP_retention<0.10; descriptive warning, never a process gate',
    'implementation_hash_rule': 'SHA256 of source bytes after CRLF-to-LF normalization; asset and artifact hashes remain byte-exact',
    'claim': 'recalibratability across these standardized candidate libraries; no acquisition/checkpoint/universal-threshold claim',
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def deps():
    global np, v57, v56, v54, inventory
    import numpy as np
    import run_v57_spectral_spatial_identity_confidence_benchmark as v57
    import run_v59_standardized_cross_library_inventory as inventory
    v56, v54 = v57.v56, v57.v54


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    if isinstance(value, dict):
        require(len(set(map(str, value))) == len(value), 'JSON_KEY_COLLISION')
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if hasattr(value, 'tolist'):
        return canonical(value.tolist())
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ahash(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(canonical(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def ids(domain=None, split=None):
    return [f'{d}__{s}_{r}_K125' for d in ([domain] if domain else DOMAINS)
            for s in ([split] if split else ('CAL', 'HOLD')) for r in REPLICATES]


def parts(dataset):
    require(dataset in ids(), 'INVALID_DATASET_ID')
    domain, base = dataset.split('__')
    split, replicate, _ = base.split('_')
    return domain, base, split, replicate


def directory(args, dataset):
    domain, base, _, _ = parts(dataset)
    return args.output_dir/domain/'clean'/base


def implementations():
    paths = [Path(__file__), Path(v57.__file__), Path(v56.__file__), Path(v54.__file__),
             Path(v57.v55.__file__), ROOT/'analysis/run_v50_reoptimized_sanity.py',
             ROOT/'src/lipid_ista.py', ROOT/'src/run_758_ista.py', ROOT/'src/config_758.py',
             ROOT/'src/utils.py', ROOT/'src/rho_zero.py']
    # Git checkouts may use CRLF on Windows and LF on AutoDL. Normalize only
    # source line endings, never numerical assets or result artifacts.
    return {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in paths}


def training_contract():
    from config_758 import Cfg
    for key, value in {**v54.v50.CFG_DEFAULTS, 'parent_channel_weight_multiplier': 1.0}.items():
        require(getattr(Cfg, key) == value, 'PRODUCTION_CONFIG_CHANGED:'+key)
    require(v54.MAX_EPOCH == 3000 and list(v54.CHECKPOINT_EPOCHS) == CONTRACT['checkpoint_epochs'], 'TRAINING_POLICY_CHANGED')
    require(not any(os.environ.get(k, '').strip() for k in ('ISTA_WEIGHT_REFERENCE_A_PATH', 'ISTA_WEIGHT_REFERENCE_META_PATH')), 'EXTERNAL_WEIGHT_REFERENCE_FORBIDDEN')
    return canonical({**v54.v50.training_contract(), 'epoch_ceiling':3000,
                      'checkpoint_loads':'same dataset runtime resume only; no production or cross-dataset checkpoint transfer',
                      'runtime_checkpoint_epochs':CONTRACT['checkpoint_epochs']})


def split_indices(metadata, domain):
    counts = Counter(m['lipid_name'] for m in metadata)
    pool = sorted((i for i, m in enumerate(metadata) if counts[m['lipid_name']] == 1),
                  key=lambda i: (metadata[i]['lipid_name'], metadata[i]['candidate_id']))
    require(len(pool) >= 2*K, 'INSUFFICIENT_SINGLETON_IDENTITIES_FOR_K125_CAL_HOLD')
    ordered = np.random.default_rng(CONTRACT['split_seed_by_domain'][domain]).permutation(pool).tolist()
    return {'CAL': ordered[:K], 'HOLD': ordered[K:2*K]}


def load_library(args, domain):
    name, low, high = DOMAINS[domain]
    folder = args.inventory_dir/'libraries'/name
    metadata = read(folder/'metadata.json')
    stamp = read(folder/'construction.json')
    with np.load(folder/'arrays.npz') as data:
        raw, a, axis = [data[k].copy() for k in ('raw', 'A_std', 'axis')]
    require([ahash(x) for x in (raw, a, axis)] == stamp['array_hashes'], 'B1_LIBRARY_ARRAY_CHANGED')
    require(a.shape == (len(axis), len(metadata)) and raw.shape == a.shape, 'LIBRARY_SHAPE_MISMATCH')
    require(np.isfinite(a).all() and (a >= 0).all() and np.all(np.linalg.norm(a, axis=0)>0), 'INVALID_LIBRARY')
    require(np.array_equal(inventory.normalize(raw), a), 'STANDARDIZED_NORMALIZATION_CHANGED')
    require(all(low <= m['mz'] <= high for m in metadata), 'WINDOW_MEMBERSHIP_CHANGED')
    return {'A_raw': raw, 'A_solver': a, 'axis': axis, 'rows': metadata,
            'metadata': {k: np.array([m[k] for m in metadata]) for k in ('candidate_id', 'lipid_name', 'lipid_class')}}


def parent_inputs(args):
    parent = read(args.v57_output/'design.json')
    s = parent['scientific']
    require(parent['status']=='DESIGN_FROZEN_BEFORE_TRAINING' and v57.fingerprint(s)==parent['design_fingerprint'], 'V57_PARENT_NOT_VALID_FROZEN')
    current=implementations()
    for name in ('lipid_ista.py','run_758_ista.py','config_758.py','rho_zero.py','run_v54_complexity_calibration.py',
                 'run_v57_spectral_spatial_identity_confidence_benchmark.py','run_v50_reoptimized_sanity.py'):
        require(current[name]==s['implementation_hashes'][name], 'V57_PRODUCTION_IMPLEMENTATION_CHANGED:'+name)
    audit=read(args.v57_output/'design_audit.json')
    require(audit.get('status')=='PASS' and audit.get('design_fingerprint')==parent['design_fingerprint'], 'V57_PARENT_AUDIT_REQUIRED')
    thresholds = read(args.v57_output/'global_frozen_thresholds.json')
    require(thresholds['design_fingerprint']==parent['design_fingerprint'] and thresholds['status']=='FROZEN_FROM_CAL_ONLY', 'V57_THRESHOLD_PROVENANCE')
    require({k:thresholds['rho_tau_'+k] for k in CLEAN_FIXED}==CLEAN_FIXED, 'V57_CLEAN_THRESHOLDS_CHANGED')
    stage0 = v56.stage0
    paths = stage0.resolve_locked_paths(args.asset_root.resolve(), stage0.load_lock())
    for name in ('production_X_hat', 'foreground_mask', 'candidate_metadata'):
        require(digest(paths[name])==s['input_hashes'][name], 'V57_SPATIAL_ASSET_CHANGED:'+name)
    x = np.load(paths['production_X_hat']).astype(np.float32)
    if x.ndim==4:
        require(x.shape[0]==1, 'PARENT_MAP_SHAPE'); x=x[0]
    mask = np.load(paths['foreground_mask']).astype(bool)
    metadata = np.load(paths['candidate_metadata'], allow_pickle=True).item()
    context = {'X_real':x, 'mask':mask, 'metadata':metadata}
    pool, bases = v57.healthy_pool(context)
    require(v57.canonical(pool)==s['healthy_pool'] and sorted(bases)==s['healthy_indices'], 'V57_HEALTHY_POOL_CHANGED')
    units = [u for u in s['matched_units'] if u['block']<=5]
    require(len(units)==K, 'PARENT_K125_SLOTS')
    amplitude = {r['unit_id']:r['abundance_multiplier'] for r in s['abundance_multipliers']}
    banks, assignment = {}, []
    for rep in REPLICATES:
        lookup = {r['unit_id']:r for r in s['spatial_assignments'] if r['replicate']==rep}
        maps = []
        for u in units:
            r = lookup[u['pair_id']]
            spatial = v57.perturb(bases[r['base_template_id']], mask, r['perturbation_seed'])
            require(ahash(spatial)==r['map_sha256'], 'V57_SPATIAL_REALIZATION_CHANGED')
            maps.append(spatial*np.float32(amplitude[u['pair_id']]))
            assignment.append({**r, 'slot':len(maps)-1,'abundance_multiplier':amplitude[u['pair_id']]})
        banks[rep] = np.stack(maps)
    return mask, banks, {'design_fingerprint':parent['design_fingerprint'],
        'file_hashes':{n:digest(args.v57_output/n) for n in ('design.json','design_audit.json','global_frozen_thresholds.json')},
        'spatial_input_hashes':{k:s['input_hashes'][k] for k in ('production_X_hat','foreground_mask','candidate_metadata')},
        'assignments':assignment,'bank_hashes':{k:ahash(v) for k,v in banks.items()},'mask_sha256':ahash(mask)}


def construct(a, metadata, mask, bank, active, dataset):
    n = a.shape[1]
    require(bank.shape==(len(active),*mask.shape) and len(set(active))==len(active), 'TRUTH_MAPPING_SHAPE')
    x = np.zeros((n,*mask.shape), dtype=np.float32)
    x[active] = bank
    b0 = np.einsum('mc,cyx->myx', a, x, optimize=True)
    p50 = float(np.median(np.linalg.norm(b0[:,mask],axis=0)))
    require(np.isfinite(p50) and p50>0, 'ZERO_SIGNAL')
    scale = CONTRACT['signal_target']/p50
    x *= np.float32(scale)
    b = np.einsum('mc,cyx->myx', a, x, optimize=True)
    require(np.allclose(b,(a@x.reshape(n,-1)).reshape(b.shape),rtol=2e-6,atol=1e-7), 'FORWARD_CONSISTENCY')
    means = x[:,mask].mean(axis=1,dtype=np.float64)
    reportable = [i for i in active if means[i]>CONTRACT['report_gate']]
    counts = Counter(metadata['lipid_name'])
    require(all(counts[metadata['lipid_name'][i]]==1 for i in active), 'NON_SINGLETON_TRUTH')
    return {'case_name':VERSION+'__'+dataset,'X_true':x,'B_sim':b,'foreground_mask':mask,
            'active_indices':active,'reportable_truth_indices':reportable,'global_scale':scale,
            'global_truth_K':len(active),'global_truth_identity_count':len(active),
            'all_truth_molecular_identity_count':len(active),'reportable_truth_molecular_identity_count':len(reportable),
            'K_true_pixel_foreground':v54.numeric_summary(np.sum(x[active][:,mask]>.001,axis=0)),
            'achieved_foreground_B_l2_p50':float(np.median(np.linalg.norm(b[:,mask],axis=0)))}


def signature(case):
    return {'X_true_sha256':ahash(case['X_true']),'B_sha256':ahash(case['B_sim']),
            'X_shape':list(case['X_true'].shape),'B_shape':list(case['B_sim'].shape),
            'global_scale':case['global_scale'],'reportable_truth_count':case['reportable_truth_molecular_identity_count'],
            'K_true_pixel_foreground':case['K_true_pixel_foreground']}


def build_scientific(args):
    # Only explicit B1 assets and V57 design/spatial provenance are read here.
    contract = read(args.inventory_dir/'standardized_library_contract.json')
    require(contract['external_precursor_range']==[700,900], 'SCOPE_NOT_FROZEN_700_900')
    previous = read(args.inventory_dir/'v59_standardized_feasibility_report.json')
    require(previous['GPU_training_executed'] is False and previous['rho_or_FDR_computed'] is False, 'OUTCOME_BLIND_SCOPE_NOT_VERIFIED')
    require(args.v57_spatial_contract=='reuse-frozen-v57', 'SPATIAL_CONTRACT_CONFIRMATION_REQUIRED')
    mask, banks, parent = parent_inputs(args)
    geometry = read(args.inventory_dir/'standardized_window_geometry.json')
    files = ('standardized_library_contract.json','standardized_window_inventory.csv','inventory_records.json',
             'standardized_window_geometry.json','w0_prod_to_std_bridge_audit.json','v59_standardized_feasibility_report.json')
    s = {'contract':CONTRACT,'implementation_hashes':implementations(),'training_contract':training_contract(),
         'B1_hashes':{n:digest(args.inventory_dir/n) for n in files},'parent':parent,'domains':{},'datasets':{},
         'scope_provenance':{'OUTCOME_BLIND_SCOPE_FREEZE':True,'basis':'B1 CPU-only audit and user scope decision before V59 learned/rho/FDR execution',
            'excluded':{'700-750':'W0 overlap','750-800':'W0 overlap','900-950':'outside 700-900 scope, not an outcome exclusion'},
            'cone_selection_used':False,'learned_selection_used':False,'quantile_stratification':'STOPPED'}}
    for domain,(name,lo,hi) in DOMAINS.items():
        lib = load_library(args,domain)
        split = split_indices(lib['rows'],domain)
        require(not {lib['rows'][i]['lipid_name'] for i in split['CAL']} & {lib['rows'][i]['lipid_name'] for i in split['HOLD']}, 'CAL_HOLD_OVERLAP')
        folder = args.inventory_dir/'libraries'/name
        s['domains'][domain] = {'label':('W0_STD' if domain=='D0' else 'E1' if domain=='D1' else 'E2'),
            'range':[lo,hi],'shape':list(lib['A_solver'].shape),'A_std_sha256':ahash(lib['A_solver']),
            'source_file_hashes':{n:digest(folder/n) for n in ('arrays.npz','metadata.json','construction.json')},
            'candidate_order':lib['rows'],'singleton_count':sum(v==1 for v in Counter(m['lipid_name'] for m in lib['rows']).values()),
            'split_indices':split,'geometry':geometry[name]}
        for dataset in ids(domain):
            _,_,sp,rep = parts(dataset)
            case = construct(lib['A_solver'],lib['metadata'],mask,banks[rep],split[sp],dataset)
            require(case['reportable_truth_molecular_identity_count']==K, 'NONREPORTABLE_TRUTH_STOP_NO_REDRAW')
            require(np.isclose(case['achieved_foreground_B_l2_p50'],CONTRACT['signal_target'],rtol=2e-6), 'GLOBAL_SIGNAL_GATE')
            s['datasets'][dataset] = signature(case)
            print('DESIGN',dataset,flush=True)
    require(len(s['datasets'])==18, 'DATASET_COUNT')
    return canonical(s)


def load_design(args, frozen=False):
    d = read(args.output_dir/'design.json')
    require(d['design_fingerprint']==fingerprint(d['scientific']), 'DESIGN_CONTENT_CHANGED')
    require(d['scientific']['contract']==canonical(CONTRACT), 'CONTRACT_CHANGED')
    require(d['scientific']['implementation_hashes']==implementations(), 'IMPLEMENTATION_CHANGED')
    require(d['scientific']['training_contract']==training_contract(), 'TRAINING_CHANGED')
    require(all(digest(args.inventory_dir/name)==h for name,h in d['scientific']['B1_hashes'].items()), 'B1_PROVENANCE_CHANGED')
    require(not (args.output_dir/'process_failure.json').exists(), 'PROCESS_FAILURE_REQUIRES_REVIEW_NO_AUTOMATIC_REDESIGN')
    if frozen:
        seal = read(args.output_dir/'design_freeze.json')
        require(seal=={'design_fingerprint':d['design_fingerprint'],'design_sha256':digest(args.output_dir/'design.json'),
                       'audit_sha256':digest(args.output_dir/'design_audit.json')}, 'FROZEN_DESIGN_CHANGED')
    return d


def context_case(args, design, dataset):
    domain,_,sp,rep = parts(dataset)
    mask,banks,parent = parent_inputs(args)
    require(canonical(parent)==design['scientific']['parent'], 'PARENT_PROVENANCE_CHANGED')
    lib = load_library(args,domain)
    frozen = design['scientific']['domains'][domain]
    folder=args.inventory_dir/'libraries'/DOMAINS[domain][0]
    require(all(digest(folder/name)==h for name,h in frozen['source_file_hashes'].items()),'LIBRARY_SOURCE_FILE_CHANGED')
    require(ahash(lib['A_solver'])==frozen['A_std_sha256'] and lib['rows']==frozen['candidate_order'], 'FROZEN_LIBRARY_CHANGED')
    case = construct(lib['A_solver'],lib['metadata'],mask,banks[rep],frozen['split_indices'][sp],dataset)
    require(canonical(signature(case))==design['scientific']['datasets'][dataset], 'FROZEN_DATASET_CHANGED')
    runtime = args.output_dir/domain/'runtime_assets'
    runtime.mkdir(parents=True,exist_ok=True)
    axis_path = runtime/'axis.npy'
    if axis_path.exists():
        require(np.array_equal(np.load(axis_path),lib['axis']), 'RUNTIME_AXIS_CHANGED')
    else:
        np.save(axis_path,lib['axis'])
    lib.update(mask=mask,paths={'channel_axis':axis_path,'candidate_metadata':args.inventory_dir/'libraries'/DOMAINS[domain][0]/'metadata.json'})
    return lib,case


def failure(args, stage, dataset, exc):
    write(args.output_dir/'process_failure.json',{'stage':stage,'dataset_id':dataset,'error':str(exc),
          'scientific_redesign_performed':False})


def oracle_pass(report):
    return all(report[k]['FP']==report[k]['FN']==0 for k in ('all_truth_metrics','candidate_level_all_truth_metrics')) and np.isfinite(report['reconstruction_relative_residual']) and report['reconstruction_relative_residual']<=CONTRACT['oracle_max_residual']


def oracles(args, design):
    hashes = {}
    for dataset in ids():
        path = directory(args,dataset)/'oracle.json'
        try:
            if path.exists():
                record = read(path)
                require(record['design_fingerprint']==design['design_fingerprint'], 'ORACLE_DESIGN_CHANGED')
            else:
                lib,case = context_case(args,design,dataset)
                result = v54.reporting_gate_oracle(case,lib['A_solver'],lib['metadata'])
                record = {'dataset_id':dataset,'design_fingerprint':design['design_fingerprint'],
                          'oracle':result,'status':'PASS' if oracle_pass(result) else 'FAIL',
                          'uniqueness_proven':False,'truth_used_as_NNLS_initialization':False}
                write(path,record)
            require(record['dataset_id']==dataset and record['status']=='PASS' and oracle_pass(record['oracle']), 'ORACLE_RECOVERY_FAILED_STOP_BEFORE_TRAINING')
            hashes[dataset] = digest(path)
            print('ORACLE PASS',dataset,flush=True)
        except Exception as exc:
            failure(args,'oracle',dataset,exc); raise
    write(args.output_dir/'oracle_all_validity.json',{'status':'PASS','design_fingerprint':design['design_fingerprint'],'hashes':hashes})


def require_oracles(args, design):
    gate = read(args.output_dir/'oracle_all_validity.json')
    require(gate['status']=='PASS' and gate['design_fingerprint']==design['design_fingerprint'] and set(gate['hashes'])==set(ids()), 'ALL_18_ORACLES_REQUIRED')
    for dataset,h in gate['hashes'].items():
        path=directory(args,dataset)/'oracle.json'
        require(digest(path)==h and oracle_pass(read(path)['oracle']), 'ORACLE_EVIDENCE_CHANGED')


def solver_evidence(path):
    names = ['solver_run.json','learned_arrays.npz','runtime_contract.json','training_history.json','latest_model.pth']
    names += [p.name for p in sorted(path.glob('checkpoint_epoch_*.pth'))]
    return {name:digest(path/name) for name in names}


def validity(learned, case, path):
    history = read(path/'training_history.json')
    fields = ('train_total','eval_total','eval_physical_loss','eval_raw_losses','automatic_loss_weights')
    finite = all(np.isfinite(np.asarray(history[k],dtype=float)).all() and len(history[k])>0 for k in fields)
    finite = finite and all(np.isfinite(learned[k]).all() for k in ('X_hat','B_hat','final_raw_losses','final_physical_loss'))
    normal = learned['stop_reason'] in ('converged','max_epochs')
    residual = v57.reconstruction_residual(learned,case)
    return {'status':'PASS' if finite and normal and np.isfinite(residual) and residual<=.10 else 'FAIL',
            'normal_completion':normal,'all_outputs_losses_finite':bool(finite),
            'final_residual':residual if np.isfinite(residual) else None,'stopped_epoch':learned['stopped_epoch'],
            'stop_reason':learned['stop_reason'],'identity_outcomes_used':False}


def train(args, design, dataset, lib, case):
    domain,base,_,_ = parts(dataset)
    require(case['B_sim'].shape==(lib['A_solver'].shape[0],*lib['mask'].shape), 'VARIABLE_M_SHAPE')
    local = SimpleNamespace(output_dir=args.output_dir/domain,device=args.device)
    learned = v57.train_or_resume(local,lib,design,base,case)
    require(learned['X_hat'].shape==case['X_true'].shape and learned['B_hat'].shape==case['B_sim'].shape,'VARIABLE_N_OUTPUT_SHAPE')
    return learned


def sentinel(args, design):
    require_oracles(args,design)
    target=args.output_dir/'sentinel_solver_validity.json'
    results = {}
    for dataset in SENTINELS:
        try:
            write(target,{'status':'IN_PROGRESS','design_fingerprint':design['design_fingerprint'],'datasets':results})
            lib,case=context_case(args,design,dataset)
            learned=train(args,design,dataset,lib,case)
            result=validity(learned,case,directory(args,dataset))
            result['artifact_hashes']=solver_evidence(directory(args,dataset))
            results[dataset]=result
            require(result['status']=='PASS', 'SENTINEL_FAIL_RESIDUAL_OR_COMPLETION')
        except (Exception,KeyboardInterrupt) as exc:
            write(target,{'status':'FAIL','design_fingerprint':design['design_fingerprint'],'datasets':results})
            failure(args,'sentinel',dataset,exc); raise
    write(target,{'status':'PASS','design_fingerprint':design['design_fingerprint'],'datasets':results,
                 'formula':'normal completion AND finite outputs/losses AND final foreground reconstruction relative residual <=0.10',
                 'hard_cap_only':3000,'minimum_epoch_requirement':None,'identity_outcomes_used':False})


def require_sentinels(args, design):
    require_oracles(args,design)
    gate=read(args.output_dir/'sentinel_solver_validity.json')
    require(gate['status']=='PASS' and gate['design_fingerprint']==design['design_fingerprint'] and set(gate['datasets'])==set(SENTINELS), 'THREE_SENTINELS_REQUIRED')
    for dataset,r in gate['datasets'].items():
        require(r['status']=='PASS' and r['normal_completion'] and r['all_outputs_losses_finite'] and r['final_residual'] is not None and r['final_residual']<=.10,'INVALID_SENTINEL')
        require(solver_evidence(directory(args,dataset))==r['artifact_hashes'],'SENTINEL_ARTIFACT_CHANGED')


def result_hashes(args, dataset):
    return {n:digest(directory(args,dataset)/n) for n in ('report.json','reported_identity_records.csv','candidate_false_negative_records.csv','molecular_false_negative_records.csv','solver_run.json')}


def require_threshold(args, design, domain):
    path=args.output_dir/domain/'frozen_thresholds.json'
    seal=read(path.with_suffix('.seal.json'))
    require(seal=={'sha256':digest(path),'design_fingerprint':design['design_fingerprint']},'THRESHOLD_SEAL_CHANGED')
    frozen=read(path)
    require(frozen['domain']==domain and frozen['design_fingerprint']==design['design_fingerprint'] and frozen['status']=='FROZEN_FROM_CAL_ONLY','THRESHOLD_DOMAIN_MISMATCH')
    require(set(frozen['CAL_hashes'])==set(ids(domain,'CAL')), 'CAL_SOURCE_SET_CHANGED')
    require(all(result_hashes(args,d)==h for d,h in frozen['CAL_hashes'].items()),'SEALED_CAL_RESULTS_CHANGED')
    return frozen


def run_dataset(args, design, dataset):
    require_sentinels(args,design)
    domain,base,split,rep=parts(dataset)
    if split=='HOLD':
        require_threshold(args,design,domain)
    out=directory(args,dataset)
    seal=out/'result_seal.json'
    if seal.exists():
        require(read(seal)=={'design_fingerprint':design['design_fingerprint'],'hashes':result_hashes(args,dataset)},'RESULT_CHANGED')
        return
    try:
        lib,case=context_case(args,design,dataset)
        learned=train(args,design,dataset,lib,case)
        status=validity(learned,case,out)
        require(status['normal_completion'] and status['all_outputs_losses_finite'],'NONFINITE_OR_ABNORMAL_TRAINING')
        records,raw=v54.learned_records(base,case,learned,lib,args.rho_workers)
        for r in records:
            r.update(dataset_id=dataset,domain=domain,K=K,reportable_truth=r['lipid_name'] in raw['reportable_truth_lipid_names'])
        report={'status':'COMPLETE','dataset_id':dataset,'domain':domain,'split':split,'replicate':rep,
                'design_fingerprint':design['design_fingerprint'],'all_truth_molecular_identity_count':K,
                'reportable_truth_molecular_identity_count':case['reportable_truth_molecular_identity_count'],
                'learned_raw_identity_performance':raw,'process_diagnostics':status}
        units=v54.molecular_units(records,{dataset:report})
        raw['molecular_level'].update(v56.metrics(units,{dataset:report}))
        reported={r['candidate_index']:r for r in records}
        means=learned['X_hat'][:,lib['mask']].mean(axis=1,dtype=np.float64)
        candidates=[{'dataset_id':dataset,'domain':domain,'K':K,'split':split,'replicate':rep,
            'candidate_index':i,'candidate_id':m['candidate_id'],'lipid_name':m['lipid_name'],'lipid_class':m['lipid_class'],
            'X_hat':float(means[i]),'rho_zero':reported[i]['rho_zero'] if i in reported else None,
            'rho_zero_status':'COMPUTED' if i in reported else 'NOT_COMPUTED','raw_solver_reported':i in reported,
            'molecular_truth':m['lipid_name'] in raw['truth_lipid_names'],
            'candidate_truth':i in case['active_indices'],'reportable_truth':m['lipid_name'] in raw['reportable_truth_lipid_names']}
            for i,m in enumerate(lib['rows'])]
        # V57 helper expects candidate geometry for duplicate identities; grouping here
        # preserves only genuine molecular fields, never inventing group-rho.
        molecular=[]
        lookup={u['lipid_name']:u for u in units}
        for name in sorted({m['lipid_name'] for m in candidates}):
            members=[m for m in candidates if m['lipid_name']==name]
            u=lookup.get(name)
            molecular.append({**{k:v for k,v in members[0].items() if k not in ('candidate_index','candidate_id','candidate_truth')},
                'candidate_indices':[m['candidate_index'] for m in members],
                'raw_solver_reported':u is not None,'rho_zero':u['rho_zero'] if u else None,
                'rho_zero_status':'COMPUTED' if u else 'NOT_COMPUTED',
                'X_hat':u['X_hat'] if u else sum(m['X_hat'] for m in members)})
        v57.write_rows(out/'reported_identity_records.csv',records,None if records else ['dataset_id','lipid_name','rho_zero','X_hat','molecular_truth','complexity_condition'])
        v57.write_rows(out/'candidate_false_negative_records.csv',candidates)
        v57.write_rows(out/'molecular_false_negative_records.csv',molecular)
        write(out/'report.json',report)
        write(seal,{'design_fingerprint':design['design_fingerprint'],'hashes':result_hashes(args,dataset)})
        print('COMPLETE',dataset,flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        failure(args,'formal',dataset,exc); raise


def completed(args, design, domain, split):
    # Guard occurs before any HOLD report/record path is opened.
    if split=='HOLD':
        require_threshold(args,design,domain)
    reports,records,hashes={},[],{}
    for dataset in ids(domain,split):
        out=directory(args,dataset)
        require(read(out/'result_seal.json')=={'design_fingerprint':design['design_fingerprint'],'hashes':result_hashes(args,dataset)},'RESULT_SEAL_CHANGED')
        report=read(out/'report.json')
        require(report['dataset_id']==dataset and report['status']=='COMPLETE' and report['design_fingerprint']==design['design_fingerprint'],'INCOMPLETE_RESULT')
        reports[dataset]=report
        records.extend(v57.rows(out/'reported_identity_records.csv'))
        hashes[dataset]=result_hashes(args,dataset)
    return v54.molecular_units(records,reports),reports,hashes


def freeze_threshold(args, design, domain):
    cal,reports,hashes=completed(args,design,domain,'CAL')
    pack,curves=v56.threshold_pack(cal,reports)
    result={'status':'FROZEN_FROM_CAL_ONLY','domain':domain,'design_fingerprint':design['design_fingerprint'],
            'CAL_hashes':hashes,'thresholds':pack,'selection':CONTRACT['primary']}
    path=args.output_dir/domain/'frozen_thresholds.json'
    if path.exists():
        require(read(path)==canonical(result),'FROZEN_THRESHOLD_CHANGED')
    else:
        write(path,result)
    seal={'sha256':digest(path),'design_fingerprint':design['design_fingerprint']}
    if path.with_suffix('.seal.json').exists():
        require(read(path.with_suffix('.seal.json'))==seal,'THRESHOLD_SEAL_CHANGED')
    else:
        write(path.with_suffix('.seal.json'),seal)
    write(args.output_dir/domain/'CAL_threshold_curves.json',curves)
    return result


def score_summary(units):
    from scipy.stats import rankdata
    truth=np.asarray([u['molecular_truth'] for u in units],dtype=bool)
    scores=np.asarray([u['rho_zero'] for u in units],dtype=float)
    require(np.isfinite(scores).all(),'NONFINITE_RHO')
    p,n=int(truth.sum()),int((~truth).sum())
    auc=float((rankdata(scores)[truth].sum()-p*(p+1)/2)/(p*n)) if p and n else None
    tp=fp=0; ap=0.
    for value in sorted(set(scores),reverse=True):
        tied=scores==value; dt=int(truth[tied].sum()); tp+=dt; fp+=int((~truth[tied]).sum())
        if p: ap+=(dt/p)*tp/(tp+fp)
    return {'AUROC':auc,'AUPRC_average_precision_ties_grouped':ap if p else None,
            'true_rho':v54.numeric_summary(scores[truth]),'false_rho':v54.numeric_summary(scores[~truth]),
            'exact_zero_false_count':int(((scores==0)&~truth).sum()),'population':'raw solver reported molecular units only'}


def aggregate(args, design):
    require_sentinels(args,design)
    summaries={}
    for domain in DOMAINS:
        frozen=require_threshold(args,design,domain)
        cal,cr,_=completed(args,design,domain,'CAL')
        hold,hr,_=completed(args,design,domain,'HOLD')
        pack=frozen['thresholds']; filtered={}
        by_replicate=[]
        for mode,thresholds in [('LOCAL_CAL',pack['rho_zero']),('CLEAN_FIXED',CLEAN_FIXED),('XHAT_BASELINE',pack['X_hat'])]:
            score='X_hat' if mode=='XHAT_BASELINE' else 'rho_zero'
            for label,entry in thresholds.items():
                tau=entry if mode=='CLEAN_FIXED' else (entry['threshold'] if entry else None)
                m=v56.metrics(hold,hr,score,tau)
                m.update(threshold=tau,warning='LOW_OR_ZERO_RETENTION_WARNING' if m['N_retained']==0 or (m['TP_retention'] is not None and m['TP_retention']<.10) else None,
                         role='PRIMARY' if mode=='LOCAL_CAL' else 'SECONDARY_DIAGNOSTIC_ONLY')
                filtered[mode+'_'+label]=m
                for rep in REPLICATES:
                    ds={d for d in hr if parts(d)[3]==rep}
                    by_replicate.append({'replicate':rep,'protocol':mode,'target':label,'threshold':tau,
                        **v56.metrics([u for u in hold if u['dataset_id'] in ds],{d:hr[d] for d in ds},score,tau)})
        curves=[]
        for split,us,rs in [('CAL',cal,cr),('HOLD',hold,hr)]:
            for score in ('rho_zero','X_hat'):
                curves.extend({'domain':domain,'split':split,'score':score,'role':'DESCRIPTIVE_NOT_HOLD_THRESHOLD_SELECTION',**r}
                              for r in v56.calibration_curve(us,score,rs))
        write(args.output_dir/domain/'FDR_recall_TP_retention_curves.json',curves)
        v57.write_rows(args.output_dir/domain/'FDR_recall_TP_retention_curves.csv',curves,
                       None if curves else ['domain','split','score','threshold','TP','FP','FDR','precision','all_truth_recall','reportable_truth_recall','TP_retention','coverage'])
        retained=[]; candidate_retained=[]
        for dataset in ids(domain):
            # All HOLD reads below are after the validated domain-local seal.
            ms=v57.decorate_records(v57.rows(directory(args,dataset)/'molecular_false_negative_records.csv'),pack)
            lookup={r['lipid_name']:r for r in ms}
            for row in v57.rows(directory(args,dataset)/'candidate_false_negative_records.csv'):
                mol=lookup[row['lipid_name']]
                row.update({k:v for k,v in mol.items() if k.startswith(('retained_by_','filter_induced_true_loss_')) or k=='solver_false_negative'})
                row['retention_flag_unit']='dataset/lipid_name'
                candidate_retained.append(row)
            retained.extend(ms)
        v57.write_rows(args.output_dir/domain/'molecular_false_negative_records.csv',retained)
        v57.write_rows(args.output_dir/domain/'candidate_false_negative_records.csv',candidate_retained)
        summaries[domain]={'raw_CAL':v56.metrics(cal,cr),'raw_HOLD':v56.metrics(hold,hr),
                           'rho_CAL':score_summary(cal),'rho_HOLD':score_summary(hold),'HOLD_validation':filtered,
                           'HOLD_by_replicate':by_replicate,
                           'geometry':design['scientific']['domains'][domain]['geometry']}
    # Existing V57 summary is descriptive; it never enters selection or thresholds.
    parent_report=read(args.v57_output/'report.json')
    bridge=read(args.inventory_dir/'w0_prod_to_std_bridge_audit.json')
    write(args.output_dir/'report.json',{'status':'PENDING_SCIENTIFIC_REVIEW','design_fingerprint':design['design_fingerprint'],
        'library_construction_bridge':bridge,
        'domains':summaries,'W0_PROD_descriptive_reference':{'report_sha256':digest(args.v57_output/'report.json'),
        'raw_HOLD':parent_report.get('raw_HOLD'),'heldout_global':parent_report.get('heldout_global'),
        'reference_scope':'existing V57 aggregate includes its K ladder and R1-R5; unpaired descriptive reference, not a matched K125 treatment effect',
        'paired_experiment':False},'limitations':[CONTRACT['claim'],'N=3 domains: geometry relationships descriptive only',
        'Zero retention has undefined FDR, not practical success','Replicates are spatial realizations, not independent biological samples']})


def self_test():
    """Small CPU-only model forward and oracle tests; no optimizer steps."""
    import tempfile
    import torch
    from lipid_ista import LipidENNet
    from config_758 import Cfg
    from unittest.mock import patch
    tests={}
    require(len(ids())==len(set(ids()))==18 and set(d.split('__')[0] for d in ids())==set(DOMAINS), 'TEST_DATASETS')
    require(set(SENTINELS)<=set(ids()) and len(SENTINELS)==3, 'TEST_SENTINEL_MEMBERSHIP')
    require({tuple(v[1:]) for v in DOMAINS.values()}=={(748,798),(800,850),(850,900)},'TEST_DOMAIN_RANGE')
    tests.update(dataset_count_18='PASS',domains='PASS',sentinels_in_formal_count='PASS')
    meta=[{'lipid_name':f'name{i:03d}','candidate_id':f'id{i}'} for i in range(270)]
    meta.extend([{'lipid_name':'duplicate','candidate_id':f'dup{i}'} for i in range(2)])
    for domain in DOMAINS:
        a=split_indices(meta,domain); b=split_indices(meta,domain)
        require(a==b and len(a['CAL'])==len(a['HOLD'])==125 and not set(a['CAL'])&set(a['HOLD']),'TEST_SPLIT')
        require(all(i<270 for i in a['CAL']+a['HOLD']),'TEST_SINGLETON')
    tests.update(deterministic_split='PASS',CAL_HOLD_no_lipid_overlap='PASS',singleton_only='PASS')
    mask=np.ones((8,8),dtype=bool)
    bank=np.stack([np.ones((8,8)),np.full((8,8),.75)]).astype(np.float32)
    for n,m in ((5,7),(9,12)):
        a=np.eye(m,n,dtype=np.float32)
        metadata={'lipid_name':np.array([f'truth{i}' for i in range(n)])}
        case=construct(a,metadata,mask,bank,[0,2],'UNIT_TEST')
        require(signature(case)==signature(construct(a,metadata,mask,bank,[0,2],'UNIT_TEST')),'TEST_DETERMINISTIC_CASE')
        net=LipidENNet(torch.tensor(a),K=Cfg.K_layers,clamp_min=Cfg.calib_clamp_min,clamp_max=Cfg.calib_clamp_max).eval()
        with torch.no_grad():
            x,layers,ac,_=net(torch.tensor(case['B_sim'][None]))
        require(tuple(x.shape)==(1,n,8,8) and tuple(ac.shape)==(m,n) and len(layers)==Cfg.K_layers,'TEST_VARIABLE_N_FORWARD')
        result=v54.reporting_gate_oracle(case,a,metadata)
        require(oracle_pass(result),'TEST_INDEPENDENT_NNLS')
    tests.update(variable_N_and_M_model_forward='PASS',production_12_layers='PASS',deterministic_design='PASS',independent_oracle_small='PASS')
    bad={**result,'reconstruction_relative_residual':.1}
    require(not oracle_pass(bad),'TEST_ORACLE_RESIDUAL_GATE')
    with tempfile.TemporaryDirectory(prefix='v59_unit_') as tmp:
        out=Path(tmp)
        args=SimpleNamespace(output_dir=out)
        history={k:[[1.,2.]] if k=='eval_raw_losses' else [1.] for k in ('train_total','eval_total','eval_physical_loss','eval_raw_losses','automatic_loss_weights')}
        write(out/'training_history.json',history)
        learned={'X_hat':case['X_true'],'B_hat':case['B_sim'],'final_raw_losses':[1.],
                 'final_physical_loss':1.,'stop_reason':'converged','stopped_epoch':100}
        require(validity(learned,case,out)['status']=='PASS','TEST_EARLY_STOP_ALLOWED')
        require(validity({**learned,'stopped_epoch':3000,'stop_reason':'max_epochs'},case,out)['status']=='PASS','TEST_CAP_ALLOWED')
        require(validity({**learned,'B_hat':case['B_sim']*2},case,out)['status']=='FAIL','TEST_SENTINEL_RESIDUAL_FAIL')
        require(validity({**learned,'final_physical_loss':float('nan')},case,out)['status']=='FAIL','TEST_SENTINEL_NONFINITE_FAIL')
        history['train_total']=[float('inf')]
        # Bad history is written directly because normal writer rejects nonfinite JSON.
        (out/'training_history.json').write_text(json.dumps(history),encoding='utf-8')
        require(validity(learned,case,out)['status']=='FAIL','TEST_HISTORY_NONFINITE_FAIL')
        tests['sentinel_early_stop_cap_residual_nonfinite']='PASS'
        accesses=[]
        original_read=read
        def tracked(path):
            accesses.append(str(path)); return original_read(path)
        with patch.dict(globals(),read=tracked):
            try:
                completed(args,{'design_fingerprint':'test'},'D0','HOLD')
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('HOLD_WITHOUT_SEAL_ACCEPTED')
        require(accesses and all('HOLD_' not in path for path in accesses),'TEST_HOLD_READ_BEFORE_SEAL')
        tests['HOLD_read_guard']='PASS'
        design={'design_fingerprint':'test'}
        cal_hashes={d:{'test':'hash'} for d in ids('D0','CAL')}
        frozen={'status':'FROZEN_FROM_CAL_ONLY','domain':'D0','design_fingerprint':'test','CAL_hashes':cal_hashes,'thresholds':{}}
        path=out/'D0/frozen_thresholds.json'
        write(path,frozen)
        write(path.with_suffix('.seal.json'),{'sha256':digest(path),'design_fingerprint':'test'})
        with patch.dict(globals(),result_hashes=lambda *a:{'test':'hash'}):
            require(require_threshold(args,design,'D0')==frozen,'TEST_THRESHOLD_SEAL')
            frozen['thresholds']={'rho_zero':'tampered'}; write(path,frozen)
            try:
                require_threshold(args,design,'D0')
            except RuntimeError as exc:
                require(str(exc)=='THRESHOLD_SEAL_CHANGED','TEST_WRONG_TAMPER_FAILURE')
            else:
                raise AssertionError('TAMPERED_THRESHOLD_ACCEPTED')
        tests['threshold_tamper_guard']='PASS'
    # Prove selection consumes metadata only; no outcome loader is callable.
    import ast, inspect, textwrap
    tree=ast.parse(textwrap.dedent(inspect.getsource(split_indices)))
    forbidden={'read','completed','score_summary','open','load','rho_values'}
    calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    require(not calls&forbidden,'TEST_OUTCOME_BLIND_SPLIT')
    tests['outcome_blindness']='PASS'
    units=[{'molecular_truth':True,'reportable_truth':True,'rho_zero':.2,'X_hat':1.},
           {'molecular_truth':False,'reportable_truth':False,'rho_zero':.1,'X_hat':.5}]
    reports={'test':{'all_truth_molecular_identity_count':3,'reportable_truth_molecular_identity_count':3}}
    metrics=v56.metrics(units,reports,'rho_zero',.3)
    require(metrics['raw_solver_FN']==2 and metrics['filter_induced_true_loss']==1 and metrics['filtered_FN']==3 and metrics['FDR'] is None,'TEST_MISSES_SEPARATE')
    tests['solver_FN_filter_loss_and_zero_retention']='PASS'
    curve=v56.calibration_curve(units,'rho_zero',reports)
    require(v56.choose_threshold(curve,.05)['threshold']==.2,'TEST_CAL_THRESHOLD')
    require(v56.choose_threshold([dict(curve[0],threshold=.3),curve[0]],.05)['threshold']==.2,'TEST_THRESHOLD_TIE')
    tests['threshold_max_retained_and_lower_tie']='PASS'
    require(fingerprint({1:[1,2]})==fingerprint(read_json_roundtrip({1:[1,2]})),'TEST_FINGERPRINT_JSON')
    tests['fingerprint_JSON_roundtrip']='PASS'
    print(json.dumps({'status':'PASS','tests':tests},indent=2))
    return tests


def read_json_roundtrip(value):
    return json.loads(json.dumps(value))


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',type=Path,default=ROOT.parent/'decon-lipid')
    parser.add_argument('--inventory-dir',type=Path,default=ROOT/'results/v59_standardized_cross_library_inventory')
    parser.add_argument('--v57-output',type=Path,default=ROOT/'results/v57_spectral_spatial_identity_confidence_benchmark')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results'/VERSION)
    parser.add_argument('--v57-spatial-contract',choices=['reuse-frozen-v57'],help='Explicitly accept frozen V57 shape/abundance construction; no extra V56 normalization')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--rho-workers',type=int,default=1)
    parser.add_argument('--domain',choices=list(DOMAINS))
    actions=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare-design','audit-design','freeze-design','oracle-all','sentinel','formal-all','freeze-thresholds','aggregate','self-test'):
        actions.add_argument('--'+name,action='store_true')
    actions.add_argument('--dataset',choices=ids())
    return parser.parse_args()


def main():
    args=parse_args()
    require(args.output_dir.resolve().name==VERSION and args.output_dir.resolve() not in (args.inventory_dir.resolve(),args.v57_output.resolve()),'INDEPENDENT_V59_OUTPUT_REQUIRED')
    deps()
    if args.self_test:
        return self_test()
    if args.prepare_design:
        require(not (args.output_dir/'design.json').exists(),'DESIGN_EXISTS_NO_REDRAW')
        require(not any(args.output_dir.glob('D*/clean/*/solver_run.json')),'PREEXISTING_LEARNED_OUTCOME')
        s=build_scientific(args)
        write(args.output_dir/'design.json',{'scientific':s,'design_fingerprint':fingerprint(s)})
        write(args.output_dir/'dataset_manifest.json',ids())
        print('PREPARED',fingerprint(s)); return
    design=load_design(args)
    if args.audit_design:
        require(not (args.output_dir/'design_freeze.json').exists(),'AUDIT_ALREADY_FROZEN')
        repeated=build_scientific(args)
        require(repeated==design['scientific'],'INDEPENDENT_DESIGN_RECONSTRUCTION_CHANGED')
        write(args.output_dir/'design_audit.json',{'status':'PASS','design_fingerprint':design['design_fingerprint'],
              'dataset_count':18,'deterministic_reconstruction':True,'singleton_only':True,'CAL_HOLD_overlap':0})
        return
    if args.freeze_design:
        audit=read(args.output_dir/'design_audit.json')
        require(audit['status']=='PASS' and audit['design_fingerprint']==design['design_fingerprint'],'AUDIT_REQUIRED')
        write(args.output_dir/'design_freeze.json',{'design_fingerprint':design['design_fingerprint'],
              'design_sha256':digest(args.output_dir/'design.json'),'audit_sha256':digest(args.output_dir/'design_audit.json')})
        return
    design=load_design(args,frozen=True)
    if args.oracle_all: return oracles(args,design)
    if args.sentinel: return sentinel(args,design)
    if args.dataset: return run_dataset(args,design,args.dataset)
    if args.freeze_thresholds:
        require_sentinels(args,design)
        for domain in ([args.domain] if args.domain else DOMAINS): freeze_threshold(args,design,domain)
        return
    if args.formal_all:
        require_sentinels(args,design)
        for domain in DOMAINS:
            for dataset in ids(domain,'CAL'): run_dataset(args,design,dataset)
            freeze_threshold(args,design,domain)
            for dataset in ids(domain,'HOLD'): run_dataset(args,design,dataset)
        return
    if args.aggregate: return aggregate(args,design)


if __name__=='__main__':
    main()
