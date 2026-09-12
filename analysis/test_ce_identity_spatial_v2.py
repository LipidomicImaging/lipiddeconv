"""Analytic contract tests, without production assets or outcome-dependent tuning."""
import os
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


if __name__=='__main__':unittest.main()
