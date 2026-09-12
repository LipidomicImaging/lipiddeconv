"""Small analytic checks of the new observable features; no scientific case fit."""
import importlib.util
import math
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('auxiliary', Path(__file__).with_name('run_small_mismatch_auxiliary_confidence.py'))
aux = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aux)


class ObservableFeatureTests(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[1.,1.,0.], [0.,0.,1.]])
        self.X = np.array([[1.,1.,0.,0.], [0.,1.,0.,0.], [1.,1.,1.,1.]])
        self.coef = {0:.5, 1:.25, 2:1.}

    def test_known_geometry_competition_and_effective_pixels(self):
        r = aux.observable_features(self.A,self.X,['a','b','c'],self.coef,[0,1,2])
        self.assertEqual(r['a']['library_separation'],0.)
        self.assertEqual(r['c']['library_separation'],1.)
        self.assertAlmostEqual(r['a']['competition_log_ratio'], math.log10(.5/(.25+1e-12)))
        self.assertEqual(r['a']['global_mean_log_disagreement'],0.)
        self.assertAlmostEqual(r['a']['log_effective_pixel_fraction'],math.log10(.5))
        self.assertAlmostEqual(r['b']['log_effective_pixel_fraction'],math.log10(.25))
        self.assertEqual(r['c']['log_effective_pixel_fraction'],0.)

    def test_same_name_aliases_excluded_from_all_competition(self):
        r=aux.observable_features(self.A,self.X,['a','a','c'],self.coef,[0,1,2])
        self.assertEqual(r['a']['library_separation'],1.)
        self.assertEqual(r['a']['details']['cosine_weighted_competitor_abundance'],0.)
        self.assertAlmostEqual(r['a']['log_effective_pixel_fraction'],math.log10(.45))
        self.assertEqual(r['a']['global_mean_log_disagreement'],0.)

    def test_unreported_competitors_still_participate(self):
        r=aux.observable_features(self.A,self.X,['a','b','c'],self.coef,[0,2])
        self.assertEqual(set(r),{'a','c'})
        self.assertEqual(r['a']['library_separation'],0.)
        self.assertAlmostEqual(r['a']['details']['cosine_weighted_competitor_abundance'],.25)

    def test_invalid_inputs_fail(self):
        X=self.X.copy();X[0,0]=float('nan')
        with self.assertRaises(AssertionError):aux.observable_features(self.A,X,['a','b','c'],self.coef,[0,1,2])

    def test_calibration_ties_and_empty_selection(self):
        self.assertIsNone(aux.cutoff([{'molecular_truth':True},{'molecular_truth':False}],[.5,.5]))
        self.assertEqual(aux.cutoff([{'molecular_truth':True},{'molecular_truth':False}],[.6,.5]),.6)


if __name__ == '__main__':
    unittest.main()
