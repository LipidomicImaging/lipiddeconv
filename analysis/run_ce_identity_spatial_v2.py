"""Frozen spatial-conditioning pilot; reuse production training and CE-v1 LPs."""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

GAMMA = 1e-6
KINDS = ('MILD', 'close_neighbor', 'relatively_isolated')
METHODS = ('global', 'local')
SEEDS = {'R61': 6201, 'R62': 6202}
CONTRACT = dict(version='CE_IDENTITY_SPATIAL_V2', gamma_num=GAMMA,
    mapping_seeds=SEEDS, K=125, new_fits=6, methods=METHODS,
    development_CAL='R61', development_EVAL='R62',
    weights='sum ALL same-name production X_hat channels in original foreground; no threshold',
    reporting='any same-name candidate foreground mean >0.001, unchanged',
    evidence='weighted B mean; L1 normalization as CE v1; fixed across full/delete',
    uncertainty='exact CE133 v1; no new peaks, disappearance, parent or mass changes',
    calibration='each method: maximize CAL TP at aggregate FDP<=0.01; ties fewer FP then larger epsilon',
    selection='full_upper<=epsilon-gamma AND deleted_lower>epsilon+gamma',
    status_priority=['NO_SPATIAL_EVIDENCE', 'NUMERICALLY_UNRESOLVED',
        'FULL_MODEL_INCOMPATIBLE', 'THRESHOLD_UNRESOLVED_FULL',
        'RETAINED', 'REPLACEABLE', 'THRESHOLD_UNRESOLVED'],
    go=dict(aggregate_FDP_max=.01, aggregate_retention_min=.4,
        EACH_challenge_retention_min=.4, EACH_challenge_nonempty=True),
    strong_aggregate_retention=.6, guard_gap=1e-6, proof_guard=1e-8,
    original_full_cost_cap_seconds=30, individual_LP_cap_seconds=60,
    novelty=dict(local_relative_l2_min=.001, weight_normalized_l1_min=.05,
        minimum_fraction=.5, minimum_common_reported=20,
        comparisons='new CAL vs new EVAL and EACH new vs BOTH old V1 realizations, per challenge',
        failure='STOP_NON_NOVEL_INPUT; no redraw or outcome-dependent relaxation'),
    process='finite normal production completion; 3000 hard cap, production early-stop retained',
    missing_library='existing five omissions per arm, CLEAN omission not joint mismatch+omission',
    old_run04='input parity/debug/novelty reference ONLY; no calibration or performance selection',
    eval_access='input-only training/evidence/novelty before seals; no EVAL LP or outcomes before BOTH seals',
    no_original_rho_rerun=True, no_cleanup=True)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def fp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def ah(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def require(ok, message):
    if not ok: raise RuntimeError(message)


def lp_module():
    # The immutable v1 scoring module disables CUDA on import. Restore caller visibility.
    prior = os.environ.get('CUDA_VISIBLE_DEVICES')
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    module = importlib.import_module('run_ce_uncertainty_identity_pilot')
    require(sha(Path(module.__file__))=='63768f2d37a7cb90da4cb2dcdd0dbd51f5cc71f25d12f3ede5f0b5f3f67a35d9', 'FROZEN_V1_LP_IMPLEMENTATION_CHANGED')
    if prior is None: os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else: os.environ['CUDA_VISIBLE_DEVICES'] = prior
    return module


@contextmanager
def new_mapping_namespace(v57):
    original = v57.MAPPING_SEEDS
    v57.MAPPING_SEEDS = dict(SEEDS)
    try: yield
    finally: v57.MAPPING_SEEDS = original


def membership():
    return [dict(key=f'{kind}__{split}_{rep}_K125', kind=kind, base=f'{split}_{rep}_K125',
                 role=role, replicate=rep) for rep,role in [('R61','CAL'),('R62','EVAL')]
            for kind,split in [('MILD','CAL'),('close_neighbor','HOLD'),('relatively_isolated','HOLD')]]


def sources(args):
    sys.path[:0] = [str(args.root/'analysis'), str(args.root/'src')]
    import numpy as np
    import torch
    import run_v58_spectral_library_mismatch_fdr_recalibration as v
    import run_missing_library_challenge as missing
    from config_758 import Cfg
    v.dependencies(); torch.set_num_threads(1)
    va = v.parse_args(['--audit-design','--v57-output',str(args.root/'results/v57_spectral_spatial_identity_confidence_benchmark'),
                      '--asset-root',str(args.asset_root)])
    parent, _, parent_hashes = v.parent_provenance(va)
    context = v.load_context(va, parent); context['units'] = parent['scientific']['matched_units']
    for key,value in {**v.v54.v50.CFG_DEFAULTS,'parent_channel_weight_multiplier':1.0}.items():
        require(getattr(Cfg,key)==value, 'PRODUCTION_CONFIG_CHANGED:'+key)
    require(v.v54.MAX_EPOCH==3000 and tuple(v.v54.CHECKPOINT_EPOCHS)==(1000,1500,2000,2500,3000), 'TRAINING_CAP_CHANGED')
    for key in ('ISTA_WEIGHT_REFERENCE_A_PATH','ISTA_WEIGHT_REFERENCE_META_PATH'):
        require(not os.environ.get(key), 'EXTERNAL_WEIGHT_REFERENCE_FORBIDDEN')
    target, _, library_hashes = v.build_targets(context)
    old_missing = args.root/'results/missing_library_challenge_k125'
    md = read(old_missing/'execution_design.json'); ms = read(old_missing/'execution_freeze.json')
    require(fp(md['scientific'])==md['fingerprint']=='b51137f3c6f12cf9bdf0e3e9f1c00c1fff53f9e4d62dc1c54de0b04a7910d9ef', 'OMISSION_DESIGN_CHANGED')
    require(ms['design_sha256']==sha(old_missing/'execution_design.json'), 'OMISSION_SEAL_CHANGED')
    uncertainty = read(args.uncertainty/'provenance.json')
    for name,h in uncertainty['outputs'].items(): require(sha(args.uncertainty/name)==h, 'U_CHANGED:'+name)
    implementation = {**v.implementation_hashes(), Path(__file__).name:sha(Path(__file__)),
                      'run_ce_uncertainty_identity_pilot.py':sha(Path(lp_module().__file__)),
                      'run_missing_library_challenge.py':sha(Path(missing.__file__))}
    binding = dict(parent_fingerprint=parent['design_fingerprint'], parent_files=parent_hashes,
                   old_input_output_manifest_sha256=sha(args.old_v1/'output_manifest.json'),
                   omission_design_sha256=sha(old_missing/'execution_design.json'),
                   uncertainty_manifest_sha256=sha(args.uncertainty/'provenance.json'),
                   libraries=library_hashes, implementation=implementation,
                   training_config={k:getattr(Cfg,k) for k in v.v54.v50.CFG_DEFAULTS},
                   production_training=v.v54.v50.training_contract())
    return np, v, missing, context, parent, target, md['scientific']['arms'], binding


def case_for(pack, scientific, entry):
    np,v,missing,context,parent,targets,arms,_ = pack
    with new_mapping_namespace(v.v57):
        case = v.v57.construct_case(context, scientific, entry['base'])
    if entry['kind']=='MILD':
        case = dict(case, B_sim=np.einsum('mc,cyx->myx',targets['MILD'],case['X_true'],optimize=True))
        reduced, kept = context, np.arange(391)
    else:
        reduced, kept = missing.reduced_context(context, arms[entry['kind']], np)
        require(set(arms[entry['kind']]['removed_original_indices'])<=set(case['active_indices']), 'OMISSION_NOT_TRUE')
    case['case_name'] = 'CE_SPATIAL_V2__'+entry['key']
    return case, reduced, kept


def validate_design(args, pack):
    d = read(args.output/'design.json'); seal = read(args.output/'design_seal.json')
    require(d['fingerprint']==fp(d['scientific']) and seal['design_sha256']==sha(args.output/'design.json'), 'DESIGN_CHANGED')
    require(d['scientific']['contract']==json.loads(json.dumps(CONTRACT)), 'CONTRACT_CHANGED')
    require(d['scientific']['source_binding']==pack[-1], 'SOURCE_IMPLEMENTATION_OR_INPUT_CHANGED')
    require(d['scientific']['membership']==membership(), 'MEMBERSHIP_CHANGED')
    return d


def prepare(args):
    np,v,missing,context,parent,targets,arms,binding = pack = sources(args)
    require(not args.output.exists(), 'OUTPUT_EXISTS_NO_REDRAW')
    scientific = copy.deepcopy(parent['scientific'])
    pool,bases = v.v57.healthy_pool(context)
    require(sorted(bases)==scientific['healthy_indices'], 'HEALTHY_POOL_CHANGED')
    context['bases']=bases
    with new_mapping_namespace(v.v57):
        assignments=v.v57.spatial_assignments(context,scientific['spectral_pair_design'],bases,
                    scientific['healthy_pair_correlations'],scientific['spatial_low_q25'])
    scientific['spatial_assignments']=assignments
    scientific['parent_contract']=scientific.pop('contract')
    scientific['version']='V2_NEW_REALIZATIONS_USING_FROZEN_V57_RECIPE'
    scientific['realization_mapping_seeds']=dict(SEEDS)
    require(not ({r['map_sha256'] for r in assignments}&{r['map_sha256'] for r in parent['scientific']['spatial_assignments']}), 'OLD_MAP_REUSED')
    frozen=dict(contract=CONTRACT, source_binding=binding, membership=membership(), spatial_design=scientific, cases={})
    args.output.mkdir(parents=True)
    shutil.copytree(args.uncertainty,args.output/'uncertainty')
    for entry in membership():
        case,reduced,kept=case_for(pack,scientific,entry);dest=args.output/'cases'/entry['key'];dest.mkdir(parents=True)
        require(case['global_truth_K']==125 and case['reportable_truth_molecular_identity_count']==125,'TRUTH_COUNT_CHANGED')
        np.savez_compressed(dest/'observation_truth.npz',B=case['B_sim'],X_true=case['X_true'],mask=case['foreground_mask'])
        info=dict(**entry,B_sha256=ah(case['B_sim']),X_true_sha256=ah(case['X_true']),A_sha256=ah(reduced['A_solver']),
                  observation_file_sha256=sha(dest/'observation_truth.npz'),kept=kept.tolist(),global_scale=case['global_scale'],
                  truth_indices=case['active_indices'],reportable_truth_indices=case['reportable_truth_indices'])
        write(dest/'input.json',info);frozen['cases'][entry['key']]=info
    d=dict(fingerprint=fp(frozen),scientific=frozen);write(args.output/'design.json',d)
    write(args.output/'design_seal.json',dict(status='FROZEN_BEFORE_NEW_TRAINING_AND_SCORES',design_sha256=sha(args.output/'design.json'),fingerprint=d['fingerprint']))
    print('DESIGN_FROZEN',d['fingerprint'],flush=True)


def spatial_evidence(np, B, X, mask, metadata, kept, truth_names, reportable_names):
    require(np.isfinite(B).all() and np.isfinite(X).all() and (X>=0).all(), 'NONFINITE_OR_NEGATIVE_INPUT')
    names=list(dict.fromkeys(str(n) for n in metadata['lipid_name']))
    solver_names=[str(metadata['lipid_name'][i]) for i in kept]
    xf=X[:,mask].astype(float); bf=B[:,mask].astype(float)
    means=xf.mean(axis=1); weights=np.zeros((len(names),xf.shape[1]));records=[]
    for i,name in enumerate(names):
        columns=[j for j,n in enumerate(solver_names) if n==name]
        weights[i]=xf[columns].sum(axis=0) if columns else 0
        records.append(dict(lipid_name=name,evidence_index=i,removed=columns,
            raw_solver_reported=bool(columns and np.any(means[columns]>.001)),
            X_hat=float(means[columns].sum()) if columns else 0.,
            molecular_truth=name in truth_names,reportable_truth=name in reportable_names))
    sums=weights.sum(axis=1);active=sums>0;weights[active]/=sums[active,None]
    local=weights@bf.T;global_b=bf.mean(axis=1)
    require(np.isfinite(local).all() and np.all(local.sum(axis=1)[active]>0), 'INVALID_LOCAL_OBSERVATION')
    return dict(weights=weights,weight_sums=sums,local=local,global_b=global_b),records


def train_case(args,pack,d,entry):
    np,v,missing,context,_,_,_,_=pack
    case,reduced,kept=case_for(pack,d['scientific']['spatial_design'],entry)
    dest=args.output/'cases'/entry['key'];info=d['scientific']['cases'][entry['key']]
    require(ah(case['B_sim'])==info['B_sha256'] and ah(case['X_true'])==info['X_true_sha256'] and ah(reduced['A_solver'])==info['A_sha256'], 'CASE_CHANGED')
    require(sha(dest/'observation_truth.npz')==info['observation_file_sha256'], 'OBSERVATION_ARCHIVE_CHANGED')
    training=dest/'training';bound=dict(fingerprint=d['fingerprint'],case=entry['key'],input=info)
    missing.bind_runtime(training,bound)
    completion=dest/'training_complete.json'
    if completion.exists():
        verify_files(dest,read(completion)['files']);return
    require(shutil.disk_usage(args.output).free>1400*1024**2, 'STOP_STORAGE_BEFORE_TRAINING')
    learned=v.v54.train_resumable({k:case[k] for k in ('case_name','B_sim','foreground_mask')},v.source_training_context(reduced),training,device_name=args.device)
    require(all(v.solver_validity(learned)) and v.finite_loss_history(read(training/'training_history.json')), 'ABNORMAL_OR_NONFINITE_TRAINING')
    require(learned['X_hat'].shape==(len(kept),*case['foreground_mask'].shape),'LEARNED_SHAPE_CHANGED')
    np.savez_compressed(training/'learned_arrays.npz',X_hat=learned['X_hat'],B_hat=learned['B_hat'])
    write(training/'solver_run.json',{k:value for k,value in learned.items() if k not in ('X_hat','B_hat')})
    truth={str(context['metadata']['lipid_name'][i]) for i in info['truth_indices']}
    reportable={str(context['metadata']['lipid_name'][i]) for i in info['reportable_truth_indices']}
    evidence,records=spatial_evidence(np,case['B_sim'],learned['X_hat'],case['foreground_mask'],context['metadata'],kept,truth,reportable)
    np.savez_compressed(dest/'evidence.npz',**evidence,A=reduced['A_solver'].astype(float),kept=kept)
    write(dest/'molecular_records.json',records)
    files={str(p.relative_to(dest)):sha(p) for p in training.iterdir() if p.is_file()}
    files.update({n:sha(dest/n) for n in ('input.json','observation_truth.npz','evidence.npz','molecular_records.json')})
    write(completion,dict(status='NORMAL_FINITE_COMPLETE',case=entry['key'],fingerprint=d['fingerprint'],files=files,
        weights_frozen_before_LPs=True,training_uses_no_truth_or_target_library=True,
        stopped_epoch=learned['stopped_epoch'],stop_reason=learned['stop_reason']))
    print('TRAINING_AND_EVIDENCE_COMPLETE',entry['key'],flush=True)


def verify_files(root,files):
    for n,h in files.items(): require(sha(root/n)==h,'ARTIFACT_CHANGED:'+n)


def summarize(values):
    import numpy as np
    finite=[float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:return dict(n=len(values),finite_n=0)
    return dict(n=len(values),finite_n=len(finite),quantiles=dict(zip(['min','q25','median','q75','max'],
                [float(v) for v in np.quantile(finite,[0,.25,.5,.75,1])])))


def compare_evidence(np,left,left_rows,right,right_rows):
    ri={r['lipid_name']:r for r in right_rows};rows=[]
    for l in left_rows:
        r=ri.get(l['lipid_name'])
        if not r or not l['raw_solver_reported'] or not r['raw_solver_reported']:continue
        i,j=l['evidence_index'],r['evidence_index']
        if left['weight_sums'][i]<=0 or right['weight_sums'][j]<=0:continue
        a,b=left['local'][i],right['local'][j];a=a/a.sum();b=b/b.sum()
        w,z=left['weights'][i],right['weights'][j]
        wc,zc=w-w.mean(),z-z.mean();den=np.linalg.norm(wc)*np.linalg.norm(zc)
        rows.append(dict(lipid_name=l['lipid_name'],local_l2=float(np.linalg.norm(a-b)),
            local_relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
            weight_overlap=float(np.minimum(w,z).sum()),weight_l1=float(abs(w-z).sum()),
            weight_correlation=float(wc@zc/den) if den>0 else None))
    threshold=CONTRACT['novelty'];novel=sum(r['local_relative_l2']>threshold['local_relative_l2_min'] and r['weight_l1']>threshold['weight_normalized_l1_min'] for r in rows)
    a,b=left['global_b'],right['global_b'];an,bn=a/a.sum(),b/b.sum()
    return dict(rows=rows,distributions={k:summarize([r[k] for r in rows]) for k in ['local_l2','local_relative_l2','weight_overlap','weight_l1','weight_correlation']},
        global_relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(a)),normalized_global_relative_l2=float(np.linalg.norm(an-bn)/np.linalg.norm(an)),
        novel_fraction=novel/len(rows) if rows else 0.,pass_novelty=len(rows)>=threshold['minimum_common_reported'] and novel/len(rows)>=threshold['minimum_fraction'])


def novelty(args,pack,d):
    np,v,_,context,parent,targets,arms,_=pack
    old_manifest=read(args.old_v1/'output_manifest.json');comparisons={};reference=args.output/'old_input_references';reference.mkdir(exist_ok=True)
    for kind in KINDS:
        entries=[e for e in membership() if e['kind']==kind]
        new=[]
        for e in entries:
            p=args.output/'cases'/e['key'];verify_files(p,read(p/'training_complete.json')['files'])
            with np.load(p/'evidence.npz') as z: arrays={k:z[k] for k in z.files}
            new.append((arrays,read(p/'molecular_records.json')))
        comparisons[kind+'__new_CAL_vs_EVAL']=compare_evidence(np,*new[0],*new[1])
        for rep in ('R1','R2'):
            split='CAL' if kind=='MILD' else 'HOLD';key=f'{kind}__{split}_{rep}_K125';rel='inputs/'+key+'/pilot_input.json'
            require(sha(args.old_v1/rel)==old_manifest[rel],'OLD_REFERENCE_BINDING_CHANGED')
            old=read(args.old_v1/rel);src=Path(old['source_dir'])
            verify_files(src,old['original_artifacts'])
            expected=read(src/'solver_run.json')['arrays_sha256'] if kind=='MILD' else read(src/'result.json')['artifact_hashes']['learned_arrays.npz']
            require(sha(src/'learned_arrays.npz')==expected,'OLD_REFERENCE_LEARNED_CHANGED')
            c=v.v57.construct_case(context,parent['scientific'],f'{split}_{rep}_K125')
            if kind=='MILD':c['B_sim']=np.einsum('mc,cyx->myx',targets['MILD'],c['X_true'],optimize=True)
            require(ah(c['B_sim'])==old['B_cube_sha256'],'OLD_REFERENCE_B_CHANGED')
            kept=list(range(391)) if kind=='MILD' else arms[kind]['reduced_to_original']
            with np.load(src/'learned_arrays.npz') as z: x=z['X_hat']
            ev,rows=spatial_evidence(np,c['B_sim'],x,c['foreground_mask'],context['metadata'],kept,set(),set())
            # Old truth flags and old scores are deliberately not read for this input comparison.
            write(reference/(key+'.json'),dict(source_dir=str(src),arrays_sha256=expected,B_sha256=ah(c['B_sim']),used_for='INPUT_NOVELTY_ONLY'))
            for index,label in enumerate(('CAL','EVAL')):
                comparisons[f'{kind}__new_{label}_vs_old_{rep}']=compare_evidence(np,*new[index],ev,rows)
    result=dict(status='PASS' if all(v['pass_novelty'] for v in comparisons.values()) else 'STOP_NON_NOVEL_INPUT',comparisons=comparisons,
                no_performance_scores_used=True,fingerprint=d['fingerprint'])
    write(args.output/'spatial_novelty.json',result)
    require(result['status']=='PASS','STOP_NON_NOVEL_INPUT_NO_REDRAW')


def classify(row,epsilon):
    if row.get('input_status')=='NO_SPATIAL_EVIDENCE':return 'NO_SPATIAL_EVIDENCE'
    if row.get('numerical_error'):return 'NUMERICALLY_UNRESOLVED'
    full,deleted=row['full'],row['deleted']
    if full['lower']>epsilon+GAMMA:return 'FULL_MODEL_INCOMPATIBLE'
    if full['upper']>epsilon-GAMMA:return 'THRESHOLD_UNRESOLVED'
    if deleted['lower']>epsilon+GAMMA:return 'RETAINED'
    if deleted['upper']<=epsilon-GAMMA:return 'REPLACEABLE'
    return 'THRESHOLD_UNRESOLVED'


def metrics(cases,epsilon):
    from collections import Counter
    rows=[r for c in cases for r in c['records']];selected=[r for r in rows if classify(r,epsilon)=='RETAINED']
    truth=sum(c['truth_count'] for c in cases);reportable=sum(c['reportable_truth_count'] for c in cases)
    raw_tp=sum(r['molecular_truth'] for r in rows);tp=sum(r['molecular_truth'] for r in selected);false=len(selected)-tp
    return dict(TP=tp,FP=false,FN=truth-tp,raw_solver_TP=raw_tp,raw_solver_FP=len(rows)-raw_tp,raw_solver_FN=truth-raw_tp,
        filtered_TP=tp,filtered_FP=false,filtered_FN=truth-tp,filter_induced_true_loss=raw_tp-tp,
        filter_induced_true_loss_fraction=(raw_tp-tp)/raw_tp if raw_tp else None,
        TP_retention=tp/raw_tp if raw_tp else None,all_truth_recall=tp/truth if truth else None,
        reportable_truth_recall=sum(r['reportable_truth'] for r in selected)/reportable if reportable else None,
        FDP=false/len(selected) if selected else 0.,retained_count=len(selected),
        distinct_identities=len({r['lipid_name'] for r in selected}),case_count=len(cases),
        status_counts=dict(Counter(classify(r,epsilon) for r in rows)))


def calibrated(cases):
    import numpy as np
    events={0.,1.+GAMMA}
    for c in cases:
        for r in c['records']:
            if 'full' not in r or 'deleted' not in r or r.get('numerical_error'):continue
            for t in (r['full']['upper']+GAMMA,r['deleted']['lower']-GAMMA):
                if t>=0:
                    events.update([float(t),float(np.nextafter(t,-np.inf)),float(np.nextafter(t,np.inf))])
    options=[dict(epsilon=e,**metrics(cases,e)) for e in sorted(events) if e>=0]
    eligible=[v for v in options if v['retained_count'] and v['FDP']<=.01]
    best=max(eligible,key=lambda v:(v['TP'],-v['FP'],v['epsilon'])) if eligible else dict(epsilon=1.+GAMMA,**metrics(cases,1.+GAMMA))
    return dict(epsilon=best['epsilon'],chosen=best,curve=options,status='CALIBRATED' if eligible else 'EMPTY_CALIBRATION')


def go_rule(cases,epsilon):
    total=metrics(cases,epsilon);by={k:metrics([c for c in cases if c['kind']==k],epsilon) for k in KINDS}
    go=bool(total['retained_count'] and total['FDP']<=.01 and total['TP_retention'] is not None and total['TP_retention']>=.4 and
        all(v['retained_count'] and v['TP_retention'] is not None and v['TP_retention']>=.4 for v in by.values()))
    return dict(aggregate=total,by_challenge=by,go=go,strong_success=bool(go and total['TP_retention']>=.6))


def init_worker(A,kept,fractions,components):
    global WORKER_MODEL
    WORKER_MODEL=lp_module().model(A,fractions,components,kept)


def solve_identity(task):
    import numpy as np
    index,b,removed,global_full=task;proofs={};result={}
    try:
        problem=lp_module().Problem(*WORKER_MODEL,b)
        if global_full is None:
            full,px,dy=problem.solve();require(full['seconds']<=30,'FULL_LP_COST_CAP_EXCEEDED')
        else:full,px,dy=global_full
        result['full']=full;proofs.update(full_primal=px,full_dual=dy)
        if np.all(px[removed]==0):
            deleted=dict(full,seconds=0.,proof='FULL_ZERO_IDENTITY_FEASIBLE_WITNESS');dx,dy=px,dy
        else:deleted,dx,dy=problem.solve(removed)
        require(deleted['upper']>=full['lower']-1e-8,'DELETION_BOUND_INCONSISTENT')
        result['deleted']=deleted;proofs.update(deleted_primal=dx,deleted_dual=dy)
    except (RuntimeError,AssertionError) as exc:
        result['numerical_error']=repr(exc)
    return index,result,proofs


def score_case(args,entry,method):
    import numpy as np
    dest=args.output/'scores'/method/entry['key'];dest.mkdir(parents=True,exist_ok=True)
    source=args.output/'cases'/entry['key'];complete=read(source/'training_complete.json')
    # Training weights remain on disk and are rechecked at final review; scoring inputs always checked here.
    for n in ('evidence.npz','molecular_records.json'):require(sha(source/n)==complete['files'][n],'EVIDENCE_CHANGED')
    binding=dict(evidence_sha256=sha(source/'evidence.npz'),records_sha256=sha(source/'molecular_records.json'),method=method,
                 input_evidence_seal_sha256=sha(args.output/'evidence_seal.json'),
                 contract_fp=fp(CONTRACT),fingerprint=complete['fingerprint'])
    path=dest/'scores.json'
    if path.exists():
        saved=read(path);require(saved['binding']==binding,'SCORE_BINDING_CHANGED');verify_files(dest,saved['proof_files']);return saved
    all_rows=read(source/'molecular_records.json');reported=[r for r in all_rows if r['raw_solver_reported']]
    with np.load(source/'evidence.npz') as z: evidence={k:z[k] for k in z.files}
    with np.load(args.output/'uncertainty/component_fractions.npz') as z:fractions=z['fractions']
    components=read(args.output/'uncertainty/components.json');initargs=(evidence['A'],evidence['kept'].tolist(),fractions,components)
    init_worker(*initargs);global_full=None;global_error=None
    if method=='global':
        try:
            global_full=lp_module().Problem(*WORKER_MODEL,evidence['global_b']).solve()
            require(global_full[0]['seconds']<=30,'GLOBAL_FULL_LP_COST_CAP_EXCEEDED')
        except (RuntimeError,AssertionError) as exc:global_error=repr(exc)
    output=[];proof_files={};jobs=[]
    for row in reported:
        i=row['evidence_index'];row=dict(row)
        checkpoint=dest/f'record_{i:04d}.json'
        if checkpoint.exists():
            saved=read(checkpoint);require(saved['binding']==binding,'PARTIAL_SCORE_BINDING_CHANGED')
            for k,value in row.items():require(saved['record'][k]==value,'PARTIAL_RECORD_CHANGED')
            verify_files(dest,saved['proof_files']);proof_files.update(saved['proof_files']);output.append(saved['record'])
        elif method=='local' and evidence['weight_sums'][i]==0:
            row['input_status']='NO_SPATIAL_EVIDENCE';output.append(row)
        elif global_error:
            row['numerical_error']=global_error;output.append(row)
        else:
            b=evidence['local'][i] if method=='local' else evidence['global_b']
            jobs.append((row,(i,b,row['removed'],global_full)))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn'),initializer=init_worker,initargs=initargs) as ex:
        futures=[(r,ex.submit(solve_identity,t)) for r,t in jobs]
        for count,(row,future) in enumerate(futures,1):
            i,result,proofs=future.result();row=dict(row,**result)
            if proofs:
                name=f'proof_{i:04d}.npz';np.savez_compressed(dest/name,**proofs);proof_files[name]=sha(dest/name);row['proof_file']=name
            write(dest/f'record_{i:04d}.json',dict(binding=binding,record=row,
                proof_files={row['proof_file']:proof_files[row['proof_file']]} if 'proof_file' in row else {}))
            output.append(row)
            if count%25==0:
                write(dest/'progress.json',dict(completed=count,total=len(jobs),numerical_failures=sum(bool(r.get('numerical_error')) for r in output)))
                print('SCORE_PROGRESS',entry['key'],method,count,len(jobs),flush=True)
    result=dict(**entry,method=method,binding=binding,truth_count=sum(r['molecular_truth'] for r in all_rows),
                reportable_truth_count=sum(r['reportable_truth'] for r in all_rows),records=sorted(output,key=lambda r:r['evidence_index']),proof_files=proof_files)
    write(path,result);return result


def run(args):
    pack=sources(args);d=validate_design(args,pack)
    for entry in membership():
        write(args.output/'status.json',dict(status='TRAINING_OR_REUSING_VERIFIED_NEW_CASE',case=entry['key']))
        train_case(args,pack,d,entry)
    if not (args.output/'spatial_novelty.json').exists():novelty(args,pack,d)
    require(read(args.output/'spatial_novelty.json')['status']=='PASS','NOVELTY_FAILED')
    input_seal=dict(design_fingerprint=d['fingerprint'],novelty_sha256=sha(args.output/'spatial_novelty.json'),
        cases={e['key']:{n:sha(args.output/'cases'/e['key']/n) for n in ('training_complete.json','evidence.npz','molecular_records.json')} for e in membership()})
    if (args.output/'evidence_seal.json').exists():require(read(args.output/'evidence_seal.json')==input_seal,'INPUT_SEAL_CHANGED')
    else:write(args.output/'evidence_seal.json',input_seal)
    # All new inputs are constructed before scoring; EVAL outcomes remain unused until both seals exist.
    cal={};seals={}
    for method in METHODS:
        cal[method]=[score_case(args,e,method) for e in membership() if e['role']=='CAL']
        seal=calibrated(cal[method]);seal.update(method=method,contract_fingerprint=fp(CONTRACT),design_fingerprint=d['fingerprint'],
            CAL_source_hashes={c['key']:sha(args.output/'scores'/method/c['key']/'scores.json') for c in cal[method]})
        p=args.output/f'{method}_calibration_seal.json'
        if p.exists():require(read(p)==seal,'CALIBRATION_SEAL_CHANGED')
        else:write(p,seal)
        seals[method]=sha(p)
    write(args.output/'both_calibration_seals.json',seals)
    evaluated={};summary={}
    for method in METHODS:
        require(all(sha(args.output/f'{m}_calibration_seal.json')==h for m,h in seals.items()),'SEAL_CHANGED_BEFORE_EVAL')
        evaluated[method]=[score_case(args,e,method) for e in membership() if e['role']=='EVAL']
        epsilon=read(args.output/f'{method}_calibration_seal.json')['epsilon']
        summary[method]=go_rule(evaluated[method],epsilon)
        summary[method]['calibration']=metrics(cal[method],epsilon)
        summary[method]['by_case']={c['key']:metrics([c],epsilon) for c in cal[method]+evaluated[method]}
        for c in cal[method]+evaluated[method]:
            write(args.output/'scores'/method/c['key']/'classified_records.json',[dict(r,status=classify(r,epsilon)) for r in c['records']])
    require(all(sha(args.output/f'{m}_calibration_seal.json')==h for m,h in seals.items()),'SEAL_CHANGED_AFTER_EVAL')
    write(args.output/'summary.json',dict(status='COMPLETE_AWAITING_INDEPENDENT_REVIEW',methods=summary,calibration_seals=seals,
        main_go=summary['local']['go'],not_formal_independent_validation=True))
    write(args.output/'status.json',dict(status='COMPLETE_AWAITING_INDEPENDENT_REVIEW'))
    compact={};retained={}
    for p in sorted(args.output.rglob('*')):
        if not p.is_file() or p.name in ('output_manifest.json','retained_large_artifacts.json'):continue
        name=p.relative_to(args.output).as_posix()
        if p.suffix=='.pth' or p.name in ('learned_arrays.npz','observation_truth.npz'):
            retained[name]=dict(sha256=sha(p),bytes=p.stat().st_size)
        else:compact[name]=sha(p)
    write(args.output/'retained_large_artifacts.json',retained);compact['retained_large_artifacts.json']=sha(args.output/'retained_large_artifacts.json')
    write(args.output/'output_manifest.json',compact)
    print('PILOT_COMPLETE',json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run'))
    p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/lipiddeconv'))
    p.add_argument('--asset-root',type=Path,default=Path('/root/autodl-tmp/decon-lipid'))
    p.add_argument('--uncertainty',type=Path,default=Path('/root/v58_jobs/ce133_uncertainty_v1_ready'))
    p.add_argument('--old-v1',type=Path,default=Path('/root/v58_jobs/ce_uncertainty_identity_pilot_run04'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,choices=(1,4),default=4)
    args=p.parse_args()
    if args.action=='prepare':os.environ['CUDA_VISIBLE_DEVICES']=''
    try:
        prepare(args) if args.action=='prepare' else run(args)
    except BaseException:
        if args.output.exists():write(args.output/'failure.json',dict(error=traceback.format_exc(),time_utc=time.time()))
        raise


if __name__=='__main__':main()
