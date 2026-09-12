"""Analytic contract tests, without production assets or outcome-dependent tuning."""
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import run_ce_identity_spatial_v2 as v


def row(full=(0.,0.),deleted=(.1,.1),truth=True,name='x'):
    return dict(lipid_name=name,molecular_truth=truth,reportable_truth=truth,
                full=dict(lower=full[0],upper=full[1]),deleted=dict(lower=deleted[0],upper=deleted[1]))


class ContractTests(unittest.TestCase):
    def test_membership(self):
        entries=v.membership()
        self.assertEqual(len(entries),6)
        self.assertEqual(len({e['key'] for e in entries}),6)
        self.assertEqual([e['role'] for e in entries],['CAL']*3+['EVAL']*3)

    def test_weights_all_same_name_no_threshold_and_FG_only(self):
        B=np.array([[[1.,4.,999.]],[[3.,2.,999.]]])
        X=np.array([[[2.,0.,300.]],[[0.,.0001,600.]],[[0.,0.,400.]]])
        mask=np.array([[True,True,False]])
        e,r=v.spatial_evidence(np,B,X,mask,{'lipid_name':['a','a','b','absent']},[0,1,2],{'a','absent'},{'a','absent'})
        np.testing.assert_allclose(e['local'][0],(2*B[:,0,0]+.0001*B[:,0,1])/2.0001)
        self.assertEqual(r[0]['removed'],[0,1]);self.assertTrue(r[0]['raw_solver_reported'])
        self.assertEqual(e['weight_sums'][1],0.)
        self.assertEqual(e['weight_sums'][2],0.)
        self.assertTrue(r[2]['molecular_truth']);self.assertFalse(r[2]['raw_solver_reported'])

    def test_uniform_weights_equal_global(self):
        B=np.array([[[1.,3.]],[[2.,4.]]]);X=np.ones((1,1,2));mask=np.ones((1,2),dtype=bool)
        e,_=v.spatial_evidence(np,B,X,mask,{'lipid_name':['a']},[0],set(),set())
        np.testing.assert_array_equal(e['local'][0],e['global_b'])

    def test_six_statuses_and_priority(self):
        eps=.02;g=v.GAMMA
        self.assertEqual(v.classify(dict(input_status='NO_SPATIAL_EVIDENCE'),eps),'NO_SPATIAL_EVIDENCE')
        self.assertEqual(v.classify(dict(numerical_error='failed'),eps),'NUMERICALLY_UNRESOLVED')
        self.assertEqual(v.classify(row(full=(eps+2*g,eps+3*g)),eps),'FULL_MODEL_INCOMPATIBLE')
        self.assertEqual(v.classify(row(full=(eps-g/2,eps+g/2)),eps),'THRESHOLD_UNRESOLVED')
        self.assertEqual(v.classify(row(deleted=(eps-3*g,eps-2*g)),eps),'REPLACEABLE')
        self.assertEqual(v.classify(row(deleted=(eps-g/2,eps+g/2)),eps),'THRESHOLD_UNRESOLVED')
        self.assertEqual(v.classify(row(deleted=(eps+2*g,eps+3*g)),eps),'RETAINED')

    def test_original_boundary_false_not_retained(self):
        eps=.019717481319156117
        self.assertEqual(v.classify(row(full=(.0196197,.0196198),deleted=(eps+5.1e-12,eps+3e-8)),eps),'THRESHOLD_UNRESOLVED')

    def test_all_misses_keep_denominators(self):
        records=[dict(row(name='a'),numerical_error='proof'),dict(row(name='b'),input_status='NO_SPATIAL_EVIDENCE'),row(name='c')]
        c=dict(kind='MILD',records=records,truth_count=5,reportable_truth_count=5)
        m=v.metrics([c],.02)
        self.assertEqual((m['TP'],m['raw_solver_FN'],m['filter_induced_true_loss'],m['FN']),(1,2,2,4))
        self.assertEqual(m['TP_retention'],1/3)
        self.assertEqual(m['all_truth_recall'],.2)

    def test_pooled_success_cannot_hide_empty_arm(self):
        cases=[dict(kind=k,truth_count=100,reportable_truth_count=100,
                    records=[row(name=f'{k}{i}',deleted=(.1,.1) if k!='relatively_isolated' else (0.,0.)) for i in range(100)]) for k in v.KINDS]
        result=v.go_rule(cases,.02)
        self.assertGreater(result['aggregate']['TP_retention'],.6);self.assertFalse(result['go'])

    def test_calibration_events_with_gamma(self):
        c=dict(kind='MILD',truth_count=2,reportable_truth_count=2,
               records=[row(deleted=(.03,.0300001),name='a'),row(deleted=(.04,.0400001),name='b'),row(deleted=(.02,.0200001),truth=False,name='f')])
        seal=v.calibrated([c]);self.assertEqual(seal['chosen']['TP'],2);self.assertEqual(seal['chosen']['FP'],0)
        self.assertEqual(v.metrics([c],seal['epsilon'])['TP'],2)

    def test_novelty_detects_scale_only(self):
        e=dict(weights=np.array([[.2,.8]]),weight_sums=np.array([1.]),local=np.array([[1.,2.]]),global_b=np.array([1.,2.]))
        r=[dict(lipid_name='x',evidence_index=0,raw_solver_reported=True)]
        f={k:value.copy() for k,value in e.items()};f['local']*=3;f['global_b']*=3
        result=v.compare_evidence(np,e,r,f,r)
        self.assertFalse(result['pass_novelty']);self.assertEqual(result['rows'][0]['local_relative_l2'],0.)

    def test_exact_LP_reuse_and_fixed_observation_deletion(self):
        lp=v.lp_module();before=os.environ.get('CUDA_VISIBLE_DEVICES')
        lp=v.lp_module();self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'),before)
        A=np.array([[1.,0.],[1.,1.]])
        p=lp.Problem(A,np.zeros((2,0)),np.array([],int),np.array([]),np.array([]),np.array([1.,1.2]))
        frozen=p.b.copy();full,_,_=p.solve();deleted,_,_=p.solve([1])
        np.testing.assert_array_equal(p.b,frozen)
        self.assertLess(full['upper'],1e-7);self.assertAlmostEqual(deleted['upper']-1e-8,1/11,places=7)

    def test_local_and_global_scoring_fixed_spatial_fixture(self):
        from types import SimpleNamespace
        from review_ce_uncertainty_identity_pilot import proof_bounds
        B=np.array([[[1.,0.]],[[0.,1.]]]);X=B.copy();mask=np.ones((1,2),bool)
        e,records=v.spatial_evidence(np,B,X,mask,{'lipid_name':['a','b']},[0,1],{'a','b'},{'a','b'})
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'cases'/'toy';source.mkdir(parents=True)
            (root/'uncertainty').mkdir()
            np.savez_compressed(source/'evidence.npz',**e,A=np.eye(2),kept=[0,1])
            np.savez_compressed(root/'uncertainty/component_fractions.npz',fractions=np.zeros((2,0)))
            v.write(root/'uncertainty/components.json',[]);v.write(root/'evidence_seal.json',{'fixture':True})
            v.write(source/'molecular_records.json',records)
            v.write(source/'training_complete.json',dict(fingerprint='fixture',files={n:v.sha(source/n) for n in ('evidence.npz','molecular_records.json')}))
            args=SimpleNamespace(output=root,workers=1);entry=dict(key='toy',role='CAL',kind='MILD')
            global_case=v.score_case(args,entry,'global');local_case=v.score_case(args,entry,'local')
            self.assertEqual(v.metrics([global_case],.6)['TP'],0)
            self.assertEqual(v.metrics([local_case],.6)['TP'],2)
            for r in local_case['records']:
                with np.load(root/'scores/local/toy'/r['proof_file']) as z:
                    for label,removed in [('full',[]),('deleted',r['removed'])]:
                        lo,hi=proof_bounds(np.eye(2),np.zeros((2,0)),[],np.array([0,1]),e['local'][r['evidence_index']],z[label+'_primal'],z[label+'_dual'],removed)
                        self.assertAlmostEqual(lo,r[label]['lower'],places=12)
                        self.assertAlmostEqual(hi,r[label]['upper'],places=12)
            self.assertEqual(v.score_case(args,entry,'local'),local_case)

    def test_independent_evidence_reconstruction(self):
        import review_ce_identity_spatial_v2 as independent
        B=np.array([[[1.,2.,99.]],[[3.,4.,99.]]],dtype=np.float32)
        X=np.array([[[1.,0.,20.]],[[0.,.0001,20.]],[[0.,0.,20.]]],dtype=np.float32)
        mask=np.array([[True,True,False]]);names=['a','a','b'];A=np.ones((2,3),np.float32)
        e,records=v.spatial_evidence(np,B,X,mask,{'lipid_name':names},[0,1,2],{'a'},{'a'})
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'new';case=root/'cases/toy';training=case/'training';training.mkdir(parents=True)
            old=Path(folder)/'old';rel='inputs/MILD__CAL_R1_K125/pilot_input.json'
            v.write(old/rel,{'names':names});v.write(old/'output_manifest.json',{rel:v.sha(old/rel)})
            truth=np.zeros((3,1,3),np.float32);truth[0]=1
            np.savez_compressed(case/'observation_truth.npz',B=B,mask=mask,X_true=truth)
            np.savez_compressed(training/'learned_arrays.npz',X_hat=X,B_hat=B)
            np.savez_compressed(case/'evidence.npz',**e,A=A.astype(float),kept=[0,1,2])
            info=dict(kept=[0,1,2],B_sha256=v.ah(B),X_true_sha256=v.ah(truth),A_sha256=v.ah(A),truth_indices=[0],reportable_truth_indices=[0])
            v.write(case/'input.json',info);v.write(case/'molecular_records.json',records)
            v.write(training/'solver_run.json',dict(stop_reason='converged',stopped_epoch=200))
            v.write(training/'training_history.json',dict(epoch=[200],**{k:[1.] for k in ('train_total','eval_total','eval_physical_loss','eval_raw_losses','automatic_loss_weights')}))
            v.write(root/'design.json',dict(fingerprint='toy',scientific=dict(cases={'toy':info},source_binding={'old_input_output_manifest_sha256':v.sha(old/'output_manifest.json')})))
            files={p.relative_to(case).as_posix():v.sha(p) for p in case.rglob('*') if p.is_file()}
            v.write(case/'training_complete.json',dict(fingerprint='toy',files=files))
            independent.evidence_review(root,'toy',old)
            self.assertEqual(v.read(case/'independent_evidence_review.json')['status'],'PASS')


if __name__=='__main__':unittest.main()
