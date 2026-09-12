"""Analytic accounting, threshold-tie and stopping checks; no experimental fit."""
import math
import unittest
import run_identity_confidence_joint_validation as v


def case_rows(extra_tied_false=False):
    rows=[dict(lipid_name=f't{i}',molecular_truth=True,reportable_truth=True,raw_solver_reported=True,
               joint_score=.9 if i<100 else .1) for i in range(125)]
    rows += [dict(lipid_name=f'f{i}',molecular_truth=False,reportable_truth=False,raw_solver_reported=True,
                  joint_score=.95 if i==0 else .9 if i==1 and extra_tied_false else .5) for i in range(9)]
    return rows


class ValidationTests(unittest.TestCase):
    def test_one_error_among_101_and_80pct(self):
        point,_=v.choose([case_rows()])
        self.assertEqual(point['threshold'],.9)
        self.assertEqual((point['pooled']['TP'],point['pooled']['FP']),(100,1))
        self.assertTrue(v.target(point['pooled']))

    def test_tied_false_identity_cannot_be_dropped(self):
        point,_=v.choose([case_rows(True)])
        self.assertIsNone(point)

    def test_no_pooled_rescue_of_failing_case(self):
        point,_=v.choose([case_rows(),case_rows(True)])
        self.assertIsNone(point)

    def test_solver_misses_remain_in_denominator(self):
        rows=case_rows();rows[0]['raw_solver_reported']=False;rows[0]['joint_score']=None
        m=v.counts(rows,.9)
        self.assertEqual(m['raw_FN'],1)
        self.assertEqual(m['FN'],26)
        self.assertEqual(m['filter_induced_true_loss'],25)
        self.assertAlmostEqual(m['all_truth_recall'],99/125)
        self.assertFalse(v.target(m))

    def test_inference_ignores_labels_and_unreported(self):
        model=dict(mean=[0,0],scale=[1,1],coef=[1,0,0,0,0,0],intercept=0)
        r=dict(X_hat=10.,rho_zero=.1,raw_solver_reported=True,molecular_truth=False)
        self.assertAlmostEqual(v.predict(r,model),1/(1+math.exp(-1)))
        r['molecular_truth']=True
        self.assertAlmostEqual(v.predict(r,model),1/(1+math.exp(-1)))
        r['raw_solver_reported']=False
        self.assertIsNone(v.predict(r,model))


if __name__=='__main__':
    unittest.main()
