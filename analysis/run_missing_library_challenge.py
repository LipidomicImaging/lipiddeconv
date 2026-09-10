"""Frozen six-fit missing-library challenge; prepare/check are CPU-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'


def require(ok,message):
    if not ok: raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def array_hash(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def reduced_context(context,arm,np):
    kept=np.asarray(arm['reduced_to_original'],dtype=int)
    removed=set(arm['removed_original_indices'])
    require(len(kept)==386 and len(set(kept))==386 and set(kept)==set(range(391))-removed,'BAD_INDEX_MAP')
    names=context['metadata']['lipid_name']
    omitted_names={r['lipid_name'] for r in arm['omitted']}
    require({i for i,n in enumerate(names) if str(n) in omitted_names}==removed,'INCOMPLETE_MOLECULAR_OMISSION')
    metadata={}
    for key,value in context['metadata'].items():
        a=np.asarray(value)
        metadata[key]=a[kept].copy() if a.ndim and a.shape[0]==391 else value
    for key in ('lipid_name','lipid_class','candidate_id'):
        require(np.array_equal(metadata[key],np.asarray(context['metadata'][key])[kept]),'METADATA_MISALIGNED')
    return {**context,'A_raw':context['A_raw'][:,kept].copy(),
            'A_solver':context['A_solver'][:,kept].copy(),'metadata':metadata},kept


def bind_runtime(directory,binding):
    path=directory/'runtime_contract.json'
    if path.exists(): require(read(path)==binding,'CROSS_DESIGN_OR_CROSS_DATASET_CACHE')
    else:
        require(not directory.exists() or not any(directory.iterdir()),'UNBOUND_OUTPUT_OR_CHECKPOINT')
        write(path,binding)


def molecular_accounting(context,case,kept,means,rhos,np):
    truth=set(case['active_indices']);reportable=set(case['reportable_truth_indices'])
    truth_names={str(context['metadata']['lipid_name'][i]) for i in truth}
    reportable_names={str(context['metadata']['lipid_name'][i]) for i in reportable}
    inverse={int(original):i for i,original in enumerate(kept)}
    candidates=[];grouped={}
    for original,name in enumerate(context['metadata']['lipid_name']):
        name=str(name); reduced=inverse.get(original)
        value=float(means[reduced]) if reduced is not None else 0.
        reported=reduced is not None and value>0.001
        row=dict(candidate_index=original,reduced_index=reduced,lipid_name=name,
                 candidate_id=str(context['metadata']['candidate_id'][original]),
                 X_hat=value,raw_solver_reported=reported,omitted_from_solver=reduced is None,
                 molecular_truth=name in truth_names,reportable_truth=name in reportable_names,
                 rho_zero=float(rhos[reduced]) if reported else None)
        candidates.append(row)
        if reported: grouped.setdefault(name,[]).append(row)
    units=[dict(lipid_name=n,molecular_truth=n in truth_names,reportable_truth=n in reportable_names,
                X_hat=sum(r['X_hat'] for r in rs),rho_zero=max(r['rho_zero'] for r in rs)) for n,rs in grouped.items()]
    report=dict(all_truth_molecular_identity_count=len(truth_names),reportable_truth_molecular_identity_count=len(reportable_names))
    return candidates,units,report


def load_inputs(args):
    sys.path.insert(0,str(args.root/'analysis'))
    import numpy as np
    import torch
    import run_v57_spectral_spatial_identity_confidence_benchmark as v57
    from config_758 import Cfg
    torch.set_num_threads(1)
    parent=args.root/'results/v57_spectral_spatial_identity_confidence_benchmark'
    design=read(parent/'design.json');s=design['scientific']
    require(v57.fingerprint(s)==design['design_fingerprint'],'PARENT_FINGERPRINT')
    require(v57.source_implementations()==s['implementation_hashes'],'PRODUCTION_IMPLEMENTATION_CHANGED')
    for key,value in {**v57.v54.v50.CFG_DEFAULTS,'parent_channel_weight_multiplier':1.0}.items():
        require(getattr(Cfg,key)==value,'PRODUCTION_CONFIG_CHANGED:'+key)
    for key in ('ISTA_WEIGHT_REFERENCE_A_PATH','ISTA_WEIGHT_REFERENCE_META_PATH'):
        require(not os.environ.get(key),'EXTERNAL_WEIGHT_REFERENCE_FORBIDDEN')
    selection=read(args.assets/'selection_manifest.json');content=dict(selection);fp=content.pop('selection_fingerprint')
    require(v57.fingerprint(content)==fp,'SELECTION_CHANGED')
    require(selection['parent_fingerprint']==design['design_fingerprint'],'SELECTION_PARENT_CHANGED')
    context=v57.v56.load_context(args.asset_root)
    require(not context.get('missing_dependencies'),str(context.get('missing_dependencies')))
    require(context['validation']['hashes_sha256']==s['input_hashes'],'INPUT_ASSET_CHANGED')
    context['units']=s['matched_units']
    return np,torch,v57,Cfg,parent,design,selection,context


def build_manifest(args,loaded):
    np,torch,v57,Cfg,parent,design,selection,context=loaded
    from lipid_ista import LipidENNet
    from run_758_ista import build_channel_weights
    audit=read(args.assets/'control_reconstruction_audit.json')
    require(audit['status']=='PASS' and audit['parent_fingerprint']==design['design_fingerprint'],'CONTROL_AUDIT_INVALID')
    controls={r['dataset_id']:r for r in audit['datasets']}
    manifest=dict(version='missing_library_six_fit_v1',selection_fingerprint=selection['selection_fingerprint'],
                  parent_fingerprint=design['design_fingerprint'],script_sha256=sha(Path(__file__)),
                  production_hashes=v57.source_implementations(),production_config=audit['production_config'],
                  thresholds=read(parent/'global_frozen_thresholds.json'),threshold_file_sha256=sha(parent/'global_frozen_thresholds.json'),
                  control_audit_sha256=sha(args.assets/'control_reconstruction_audit.json'),
                  training='unchanged v54.train_resumable;3000 hard cap, production early-stop; fresh reduced-N fits',
                  process_gate='normal completion and all outputs/losses finite; residual is outcome, not gate',
                  rho='unchanged production rho with reduced A and original mean B, W=I',
                  measurement_noise=False,original_truth_denominator=125,omitted_truth_per_case=5,
                  planned_new_fits=6,full_library_reused_controls=3,arms=selection['arms'],cases={},controls={})
    for dataset in selection['datasets']:
        case=v57.construct_case(context,design['scientific'],dataset)
        base=parent/'clean'/dataset;record=controls[dataset]
        require(array_hash(case['B_sim'])==record['B_sha256']==read(base/'runtime_contract.json')['B_sha256'],'B_CHANGED')
        require(array_hash(case['X_true'])==record['X_true_reconstructed_sha256'],'X_TRUE_CHANGED')
        require(case['global_scale']==record['global_scale'],'SCALE_CHANGED')
        require(sha(base/'latest_model.pth')==record['latest_sha256'],'CONTROL_CHECKPOINT_CHANGED')
        for ck in record['checkpoints']: require(sha(base/ck['file'])==ck['sha256'],'CONTROL_CHECKPOINT_CHANGED')
        solver=read(base/'solver_run.json')
        require(sha(base/'learned_arrays.npz')==solver['arrays_sha256'],'CONTROL_ARRAYS_CHANGED')
        manifest['controls'][dataset]={n:sha(base/n) for n in ['solver_run.json','runtime_contract.json','training_history.json','learned_arrays.npz','latest_model.pth']}
        for arm_name,arm in selection['arms'].items():
            reduced,kept=reduced_context(context,arm,np)
            require(set(arm['removed_original_indices'])<=set(case['active_indices']),'OMISSION_NOT_TRUTH')
            A=torch.tensor(reduced['A_raw'],dtype=torch.float32)
            A=A/(torch.linalg.vector_norm(A,dim=0,keepdim=True)+1e-8)
            require(np.array_equal(A.numpy(),reduced['A_solver']),'NORMALIZATION_MISMATCH')
            ns=v57.v54.v50.production_weight_namespace(A,reduced['metadata'],torch,torch.device('cpu'))
            weights,wreport=build_channel_weights(ns,A,v57.v54.v50.runtime_cfg(reduced['paths'],Cfg))
            # Same unmodified network, tiny observed spatial patch, no optimizer or training.
            torch.manual_seed(Cfg.seed)
            model=LipidENNet(A_init=A,K=Cfg.K_layers,clamp_min=Cfg.calib_clamp_min,clamp_max=Cfg.calib_clamp_max)
            with torch.no_grad(): output=model(torch.tensor(case['B_sim'][None,:,:2,:2]))
            require(tuple(output[0].shape)==(1,386,2,2) and torch.isfinite(output[0]).all(),'CPU_FORWARD_SHAPE_OR_FINITE')
            require(tuple(output[2].shape)==(1084,386),'CALIBRATED_LIBRARY_SHAPE')
            key=arm_name+'__'+dataset
            manifest['cases'][key]=dict(base_dataset=dataset,arm=arm_name,B_sha256=array_hash(case['B_sim']),
                X_true_sha256=array_hash(case['X_true']),A_raw_sha256=array_hash(reduced['A_raw']),
                A_solver_sha256=array_hash(reduced['A_solver']),N_solver=386,
                candidate_ids=[str(x) for x in reduced['metadata']['candidate_id']],
                global_scale=case['global_scale'],truth_count=case['global_truth_K'],
                reportable_truth_count=case['reportable_truth_molecular_identity_count'],
                weights_sha256=array_hash(weights.numpy()),weight_report=wreport)
            del model,output
        print(dataset,'INPUT_AND_CPU_FORWARD_PASS',flush=True)
    return manifest


def self_test(loaded):
    np,_,v57,_,_,_,selection,context=loaded
    arm=selection['arms']['close_neighbor'];_,kept=reduced_context(context,arm,np)
    truth=set(arm['removed_original_indices'])|set(int(i) for i in kept[:120])
    require(len(truth)==125,'TEST_FIXTURE')
    # Use synthetic unique names: independently exercise removed FN and one spurious report.
    meta={'lipid_name':np.array([str(i) for i in range(391)]),'candidate_id':np.array([str(i) for i in range(391)])}
    means=np.zeros(386);means[:120]=1.;means[120]=1.
    rows,units,report=molecular_accounting({'metadata':meta},{'active_indices':truth,'reportable_truth_indices':truth},kept,means,{i:1. for i in range(121)},np)
    metric=v57.v56.metrics(units,{'test':report})
    require((metric['TP'],metric['FP'],metric['FN'])==(120,1,5),'OMITTED_TRUTH_DENOMINATOR')
    require(sum(r['omitted_from_solver'] for r in rows)==5,'OMITTED_ROWS_LOST')
    filtered=v57.v56.metrics(units,{'test':report},'rho_zero',2.)
    require(filtered['FN']==125 and filtered['filter_induced_true_loss']==120,'FN_ACCOUNTING')
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'case';bind_runtime(path,{'fp':'new','dataset':'a'})
        rejected=False
        try: bind_runtime(path,{'fp':'old','dataset':'a'})
        except RuntimeError: rejected=True
        require(rejected,'CROSS_DESIGN_NOT_REJECTED')
        other=Path(folder)/'old';other.mkdir();(other/'latest_model.pth').write_bytes(b'test')
        rejected=False
        try: bind_runtime(other,{'fp':'new'})
        except RuntimeError: rejected=True
        require(rejected,'UNBOUND_CHECKPOINT_NOT_REJECTED')
    return dict(status='PASS',removed_FN_and_spurious_FP='PASS',filter_loss_separate='PASS',
                cross_design_cache_rejected='PASS',unbound_checkpoint_rejected='PASS',metadata_mapping='PASS')


def run_one(args,loaded,manifest,fp,key):
    np,_,v57,_,_,design,selection,context=loaded
    entry=manifest['cases'][key];base=entry['base_dataset'];arm=selection['arms'][entry['arm']]
    case=v57.construct_case(context,design['scientific'],base)
    reduced,kept=reduced_context(context,arm,np)
    require(array_hash(case['B_sim'])==entry['B_sha256'] and array_hash(reduced['A_solver'])==entry['A_solver_sha256'],'RUN_INPUT_CHANGED')
    out=args.output/'runs'/key
    bind_runtime(out,dict(fingerprint=fp,case_id=key,B_sha256=entry['B_sha256'],A_solver_sha256=entry['A_solver_sha256'],reduced_to_original=kept.tolist()))
    require(not (out/'failure.json').exists(),'PRIOR_PROCESS_FAILURE_REQUIRES_REVIEW')
    if (out/'result.json').exists():
        result=read(out/'result.json');require(result['fingerprint']==fp,'RESULT_BINDING_CHANGED')
        for n,h in result['artifact_hashes'].items(): require(sha(out/n)==h,'RESULT_ARTIFACT_CHANGED')
        return
    # Only observation and mask enter the unchanged production training driver.
    learned=v57.v54.train_resumable(dict(case_name='MISSING_LIBRARY__'+key,B_sim=case['B_sim'],foreground_mask=case['foreground_mask']),reduced,out,device_name=args.device)
    require(learned['stop_reason'] in ('max_epochs','converged'),'ABNORMAL_COMPLETION')
    require(all(np.isfinite(learned[k]).all() for k in ['X_hat','B_hat','final_raw_losses','final_physical_loss']),'NONFINITE_OUTPUT')
    require(learned['X_hat'].shape==(386,*case['foreground_mask'].shape) and learned['B_hat'].shape==case['B_sim'].shape,'OUTPUT_DIMENSIONS')
    mask=case['foreground_mask'];means=learned['X_hat'][:,mask].mean(axis=1,dtype=np.float64)
    reported=np.flatnonzero(means>.001).tolist()
    rhos=v57.v54.rho_values(reduced['A_solver'],case['B_sim'][:,mask].mean(axis=1,dtype=np.float64),reported,args.rho_workers)
    candidates,units,report=molecular_accounting(context,case,kept,means,rhos,np)
    metrics={'raw':v57.v56.metrics(units,{key:report})}
    for score,targets in manifest['thresholds']['global_thresholds'].items():
        for target,threshold in targets.items():
            metrics[score+'_'+target]=v57.v56.metrics(units,{key:report},score,threshold['threshold'] if threshold else None)
    np.savez_compressed(out/'learned_arrays.npz',X_hat=learned['X_hat'],B_hat=learned['B_hat'],reduced_to_original=kept)
    write(out/'candidate_records.json',candidates);write(out/'molecular_units.json',units)
    write(out/'solver_run.json',{k:v for k,v in learned.items() if k not in ('X_hat','B_hat')})
    residual=float(np.linalg.norm(learned['B_hat'][:,mask]-case['B_sim'][:,mask])/np.linalg.norm(case['B_sim'][:,mask]))
    write(out/'result.json',dict(status='COMPLETE',fingerprint=fp,case_id=key,metrics=metrics,
          structural_omission_FN=5,additional_raw_solver_FN=metrics['raw']['FN']-5,reconstruction_residual=residual,
          artifact_hashes={n:sha(out/n) for n in ['runtime_contract.json','solver_run.json','training_history.json','latest_model.pth','learned_arrays.npz','candidate_records.json','molecular_units.json']}))


def main(args):
    if args.action in ('prepare','check'): os.environ['CUDA_VISIBLE_DEVICES']=''
    loaded=load_inputs(args)
    manifest=build_manifest(args,loaded);tests=self_test(loaded);fp=fingerprint(manifest)
    frozen=args.output/'execution_design.json';seal=args.output/'execution_freeze.json'
    if args.action=='prepare':
        require(not (args.output/'runs').exists(),'PREEXISTING_SOLVER_OUTPUT')
        if frozen.exists(): require(read(frozen)==dict(fingerprint=fp,scientific=manifest),'FROZEN_DESIGN_CHANGED')
        else: write(frozen,dict(fingerprint=fp,scientific=manifest))
        write(args.output/'validation.json',dict(**tests,case_count=len(manifest['cases']),cpu_forward_all_six='PASS',training_executed=False))
        content=dict(status='FROZEN_BEFORE_TRAINING',fingerprint=fp,design_sha256=sha(frozen),validation_sha256=sha(args.output/'validation.json'))
        if seal.exists(): require(read(seal)==content,'FREEZE_CHANGED')
        else: write(seal,content)
    else:
        require(read(frozen)==dict(fingerprint=fp,scientific=manifest),'EXECUTION_DESIGN_CHANGED')
        require(read(seal)['design_sha256']==sha(frozen) and read(seal)['validation_sha256']==sha(args.output/'validation.json'),'FREEZE_SEAL_CHANGED')
        if args.action=='run':
            keys=[args.case] if args.case else list(manifest['cases'])
            require(all(k in manifest['cases'] for k in keys),'UNKNOWN_CASE')
            for key in keys:
                write(args.output/'status.json',dict(status='RUNNING',case_id=key))
                try: run_one(args,loaded,manifest,fp,key)
                except BaseException:
                    write(args.output/'runs'/key/'failure.json',dict(status='FAIL',error=traceback.format_exc()));raise
            write(args.output/'status.json',dict(status='REQUESTED_CASES_COMPLETE',cases=keys))
    print(args.action,fp,'PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','check','run'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case');p.add_argument('--device',default='cuda:0')
    p.add_argument('--rho-workers',type=int,default=1)
    main(p.parse_args())
