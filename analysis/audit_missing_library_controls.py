"""CPU-only reconstruction and checkpoint audit of three existing V57 controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    sys.path.insert(0,str(args.root/'analysis'))
    import numpy as np
    import torch
    import run_v57_spectral_spatial_identity_confidence_benchmark as v57
    from config_758 import Cfg
    torch.set_num_threads(1)
    parent=args.root/'results/v57_spectral_spatial_identity_confidence_benchmark'
    design=json.loads((parent/'design.json').read_text())
    s=design['scientific']
    assert v57.fingerprint(s)==design['design_fingerprint']
    implementations=v57.source_implementations()
    assert implementations==s['implementation_hashes'],'IMPLEMENTATION_CHANGED'
    config={**v57.v54.v50.CFG_DEFAULTS,'parent_channel_weight_multiplier':1.0}
    for k,value in config.items():
        assert getattr(Cfg,k)==value, f'CONFIG_CHANGED:{k}'
    context=v57.v56.load_context(args.asset_root)
    assert not context.get('missing_dependencies'),context.get('missing_dependencies')
    assert context['validation']['hashes_sha256']==s['input_hashes']
    context['units']=s['matched_units']
    result=[]
    for replicate in range(1,4):
        dataset=f'HOLD_R{replicate}_K125'
        directory=parent/'clean'/dataset
        case=v57.construct_case(context,s,dataset)
        bound=json.loads((directory/'runtime_contract.json').read_text())
        b_hash=hashlib.sha256(case['B_sim'].tobytes()).hexdigest()
        assert bound==dict(dataset_id=dataset,design_fingerprint=design['design_fingerprint'],B_sha256=b_hash)
        solver=json.loads((directory/'solver_run.json').read_text())
        assert solver['dataset_id']==dataset and solver['design_fingerprint']==design['design_fingerprint']
        assert solver['arrays_sha256']==sha(directory/'learned_arrays.npz')
        training=solver['training']
        assert training['stop_reason'] in ('converged','max_epochs')
        assert np.isfinite(training['final_raw_losses']).all() and np.isfinite(training['final_physical_loss'])
        with np.load(directory/'learned_arrays.npz') as a:
            assert np.isfinite(a['X_hat']).all() and np.isfinite(a['B_hat']).all()
        checkpoint_rows=[]
        for epoch in v57.v54.CHECKPOINT_EPOCHS:
            if epoch>training['stopped_epoch']:
                continue
            path=directory/f'checkpoint_epoch_{epoch}.pth'
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            assert checkpoint['dataset_id']==case['case_name'] and checkpoint['epoch']==epoch
            assert checkpoint['X_true_used_in_training_or_stopping'] is False
            assert checkpoint['scheduler']['T_max']==Cfg.n_epochs
            assert checkpoint['history']['epoch'][-1]==epoch
            checkpoint_rows.append(dict(file=path.name,sha256=sha(path),epoch=epoch))
            del checkpoint
        latest=torch.load(directory/'latest_model.pth',map_location='cpu',weights_only=False)
        assert latest['dataset_id']==case['case_name'] and latest['epoch']==training['stopped_epoch']
        assert latest['terminal_stop_reason']==training['stop_reason']
        history=json.loads((directory/'training_history.json').read_text())
        assert latest['history']==history
        result.append(dict(dataset_id=dataset,status='PASS',B_sha256=b_hash,
                           X_true_reconstructed_sha256=hashlib.sha256(case['X_true'].tobytes()).hexdigest(),
                           A_solver_sha256=hashlib.sha256(context['A_solver'].tobytes()).hexdigest(),
                           global_scale=case['global_scale'],stopped_epoch=training['stopped_epoch'],
                           checkpoints=checkpoint_rows,latest_sha256=sha(directory/'latest_model.pth')))
        del latest,case
        print(dataset,'PASS',flush=True)
    report=dict(status='PASS',parent_fingerprint=design['design_fingerprint'],
                production_implementation_hashes=implementations,production_config=config,
                hard_epoch_cap=v57.v54.MAX_EPOCH,checkpoint_epochs=list(v57.v54.CHECKPOINT_EPOCHS),
                datasets=result,script_sha256=sha(Path(__file__)),
                limitations=['X_true hashes are reconstructed from the frozen design; no independent historical X_true byte hash was found in solver artifacts.',
                             'Checkpoint hashes are current audit snapshots, not claims of historical hash sealing.',
                             'This validates full-library controls; reduced-library execution adapter is not yet validated.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
