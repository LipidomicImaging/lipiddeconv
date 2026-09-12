"""Isolated three-channel integration checks, with no production assets or training."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    from . import physical_identity_confidence_core as core
    from . import run_physical_identity_mainline_pilot as runner
except ImportError:
    import physical_identity_confidence_core as core
    import run_physical_identity_mainline_pilot as runner


def _pattern(pattern_id, multiplier):
    return dict(pattern_id=pattern_id, donor_identity=[pattern_id, '[M-H]-', 'TEST'],
                source_files=[pattern_id + '_low', pattern_id + '_high'], ce_pair=[30., 35.],
                channels=['f1', 'f2'], centered_log_change=np.log(multiplier).tolist(),
                multiplier=list(multiplier))


def _bound_row(name, selected=True, supported=False, truth=True, index=0):
    return dict(lipid_name=name, evidence_index=index, molecular_truth=truth,
                reportable_truth=truth, supported_perturbed_truth=supported,
                full=dict(status='BOUNDS_VALID', lower=0., upper=.001, numerical_error=None),
                deleted=dict(status='BOUNDS_VALID', lower=.1 if selected else 0.,
                             upper=.1 + 2e-8 if selected else 2e-8, numerical_error=None))


def _metric_case(kind, selected=(0, 1, 5, 6), supported=(0, 1, 2, 3, 4)):
    return dict(kind=kind, truth_count=10, reportable_truth_count=10,
                supported_perturbed_truth_count=len(supported),
                records=[_bound_row(kind + str(i), i in selected, i in supported, index=i)
                         for i in range(10)])


class MainlineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='physical-mainline-analytic-')
        base = Path(self.temp.name)
        self.args = SimpleNamespace(output=base/'output', snapshot=base/'snapshot', workers=1)
        self.entry = runner.membership()[0]
        self.source = self.args.output/'cases'/self.entry['key']
        self.dest = self.args.output/'scores'/self.entry['key']
        self.source.mkdir(parents=True)
        fraction_dir = self.args.snapshot/'results/ce133_uncertainty_v1_ready'
        fraction_dir.mkdir(parents=True)
        physical_dir = self.args.snapshot/'results/physical_perturbation_contract_v1'
        physical_dir.mkdir(parents=True)
        patterns = [_pattern('MODEL', (2., .5)), _pattern('CAL', (3., 1/3)),
                    _pattern('EVAL', (4., .25))]
        runner.write(self.args.output/'model_patterns.json', patterns[:1])
        mapping = [dict(candidate_index=0, lipid_name='TRUE', rule='TEST', channels=['f1', 'f2'],
                        component_indices=[0, 1], allowed_pattern_ids=['MODEL', 'CAL', 'EVAL'])]
        runner.write(physical_dir/'candidate_mapping.json', mapping)
        np.savez_compressed(fraction_dir/'component_fractions.npz',
                            fractions=np.array([[1., 0.], [0., 1.], [0., 0.]]))
        self.b = np.array([2.357, .58925, 0.])  # Sum is noninteger; original b crosses spawn.
        A = np.array([[1., 0., 1.], [1., 1., 0.], [0., 1., 1.]])
        np.savez_compressed(self.source/'evidence.npz', A=A, kept=np.arange(3), global_b=self.b)
        runner.write(self.source/'metadata.json', dict(lipid_name=['TRUE', 'FALSE', 'MISSED']))
        self.rows = [dict(lipid_name=name, evidence_index=i, X_hat=1. if i == 0 else .002 if i == 1 else 0.,
                         raw_solver_reported=i != 2, molecular_truth=i != 1, reportable_truth=i != 1,
                         supported_perturbed_truth=i == 0, removed=[i])
                     for i, name in enumerate(('TRUE', 'FALSE', 'MISSED'))]
        runner.write(self.source/'molecular_records.json', self.rows)
        files = {name: runner.sha(self.source/name)
                 for name in ('evidence.npz', 'metadata.json', 'molecular_records.json')}
        runner.write(self.source/'training_complete.json', dict(files=files))
        runner.write(self.args.output/'evidence_seal.json', dict(fixture='ANALYTIC_ONLY'))
        self.design = dict(fingerprint='ANALYTIC_ONLY', scientific=dict(model_pattern_ids=['MODEL']))

    def tearDown(self):
        self.temp.cleanup()

    def score(self):
        with patch.object(runner, 'sources', side_effect=AssertionError('NO_PRODUCTION_SOURCE_ACCESS')):
            return runner.score_case(self.args, self.design, self.entry)

    def test_end_to_end_real_lp_bounds_membership_and_denominators(self):
        result = self.score()
        self.assertEqual(result['truth_count'], 2)
        self.assertEqual(result['reportable_truth_count'], 2)
        self.assertEqual(result['supported_perturbed_truth_count'], 1)
        self.assertEqual([r['lipid_name'] for r in result['records']], ['TRUE', 'FALSE'])
        full = runner.read(self.dest/'full.json')['result']
        self.assertEqual(full['status'], 'BOUNDS_VALID')
        for row in result['records']:
            self.assertEqual(row['full'], runner.short_bounds(full))
            saved = runner.read(self.dest/f"record_{row['evidence_index']:04d}.json")
            self.assertEqual(row['deleted'], runner.short_bounds(saved['deletion_result']))
        values = runner.metrics([result], .01)
        self.assertEqual((values['raw_solver_TP'], values['raw_solver_FP'], values['raw_solver_FN']), (1, 1, 1))
        self.assertEqual((values['filtered_TP'], values['filtered_FP'], values['filtered_FN']), (1, 0, 1))
        self.assertEqual(values['filter_induced_true_loss'], 0)
        self.assertEqual(values['all_truth_recall'], .5)
        self.assertEqual(values['reportable_truth_recall'], .5)
        self.assertEqual(values['TP_retention'], 1.)

    def test_resume_verifies_proofs_without_repeating_any_lp(self):
        original = self.score()
        with patch.object(core.ModelBank, 'score_full', side_effect=AssertionError('FULL_MUST_NOT_REPEAT')):
            with patch.object(core.ModelBank, 'score_delete', side_effect=AssertionError('DELETE_MUST_NOT_REPEAT')):
                with patch.object(core._problem_class(), 'solve', side_effect=AssertionError('NO_LP_ON_RESUME')):
                    resumed = self.score()
        self.assertEqual(original, resumed)

    def test_cached_summary_cannot_disagree_with_verified_proof_or_input(self):
        self.score()
        checkpoint = self.dest/'record_0001.json'
        original = runner.read(checkpoint)
        mutations = [lambda row: row['record']['deleted'].update(lower=.9, upper=.9),
                     lambda row: row['record'].update(molecular_truth=True),
                     lambda row: row['record']['full'].update(upper=0.)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(original); mutate(changed); runner.write(checkpoint, changed)
                with self.assertRaises(RuntimeError):
                    self.score()
                runner.write(checkpoint, original)

    def test_cached_bank_manifest_must_match_rebuilt_model(self):
        self.score()
        manifest = runner.read(self.dest/'bank_manifest.json')
        manifest['model_pattern_ids'].append('CAL')
        runner.write(self.dest/'bank_manifest.json', manifest)
        with self.assertRaises(RuntimeError):
            self.score()

    def test_full_npz_hash_and_actual_numerical_proof_both_checked(self):
        self.score()
        proof_file = self.dest/'full_proof.npz'
        original_bytes = proof_file.read_bytes()
        proof_file.write_bytes(original_bytes + b'TAMPER')
        with self.assertRaisesRegex(RuntimeError, 'FULL_PROOF_CHANGED'):
            self.score()
        proof_file.write_bytes(original_bytes)
        with np.load(proof_file) as data:
            arrays = {k: data[k].copy() for k in data.files}
        arrays['all_relaxed__dual'][0] = 1.
        np.savez_compressed(proof_file, **arrays)
        full_file = self.dest/'full.json'; item = runner.read(full_file)
        item['proof_sha256'] = runner.sha(proof_file); runner.write(full_file, item)
        with self.assertRaises((AssertionError, ValueError)):
            self.score()

    def test_inference_receives_model_only_and_never_reads_target(self):
        self.args.output.joinpath('A_target_CAL.npy').write_bytes(b'INVALID_MUST_NOT_BE_READ')
        self.design['scientific']['targets'] = {'CAL': 'MUST_NOT_BE_PASSED_TO_CONFIDENCE'}
        build = core.build_model_bank
        captured = []
        def checked_build(A, fractions, patterns, mapping, ids, metadata, kept):
            self.assertEqual({p['pattern_id'] for p in patterns}, {'MODEL'})
            self.assertEqual(ids, ['MODEL'])
            self.assertTrue(all(set(row) == {'lipid_name'} for row in metadata))
            bank = build(A, fractions, patterns, mapping, ids, metadata, kept)
            captured.append(bank.fingerprint)
            return bank
        with patch.object(core, 'build_model_bank', side_effect=checked_build):
            result = self.score()
        self.assertEqual(captured, [result['binding']['bank_fingerprint']])
        manifest = runner.read(self.dest/'bank_manifest.json')
        self.assertEqual(manifest['model_pattern_ids'], ['MODEL'])
        self.assertEqual({a['pattern_id'] for a in manifest['atoms']}, {'MODEL', core.NOMINAL})


class MainlineDecisionTests(unittest.TestCase):
    def test_gamma_closed_full_and_open_deletion_edges(self):
        epsilon = .01
        row = _bound_row('TRUE', supported=True)
        row['full']['upper'] = epsilon - runner.GAMMA
        row['deleted']['lower'] = epsilon + runner.GAMMA
        self.assertEqual(runner.classify(row, epsilon), 'THRESHOLD_UNRESOLVED')
        row['deleted']['lower'] = float(np.nextafter(epsilon + runner.GAMMA, np.inf))
        self.assertEqual(runner.classify(row, epsilon), 'RETAINED')
        row['full']['upper'] = float(np.nextafter(epsilon - runner.GAMMA, np.inf))
        self.assertEqual(runner.classify(row, epsilon), 'THRESHOLD_UNRESOLVED_FULL')

    def test_calibration_event_search_retains_truth_and_rejects_false(self):
        truth = _bound_row('TRUE', supported=True)
        false = _bound_row('FALSE', truth=False, index=1)
        truth['full']['upper'] = false['full']['upper'] = .005
        truth['deleted'].update(lower=.015, upper=.015 + 2e-8)
        false['deleted'].update(lower=.006, upper=.006 + 2e-8)
        case = dict(kind=runner.KINDS[0], truth_count=2, reportable_truth_count=2,
                    supported_perturbed_truth_count=1, records=[truth, false])
        calibrated = runner.calibrated([case])
        self.assertEqual(calibrated['status'], 'CALIBRATED')
        self.assertEqual((calibrated['chosen']['filtered_TP'], calibrated['chosen']['filtered_FP']), (1, 0))
        self.assertEqual(runner.classify(truth, calibrated['epsilon']), 'RETAINED')
        self.assertNotEqual(runner.classify(false, calibrated['epsilon']), 'RETAINED')
        self.assertEqual(calibrated['chosen']['raw_solver_FN'], 1)
        acceptable = [r for r in calibrated['curve'] if r['retained_count'] and r['FDP'] <= .01]
        self.assertEqual(calibrated['epsilon'], max(r['epsilon'] for r in acceptable))

    def test_exact_forty_percent_each_arm_and_supported_is_go_not_strong(self):
        cases = [_metric_case(kind) for kind in runner.KINDS]
        decision = runner.go_rule(cases, .01)
        self.assertTrue(decision['go'])
        self.assertFalse(decision['strong_success'])
        self.assertEqual(decision['aggregate']['TP_retention'], .4)
        self.assertEqual(decision['supported_perturbed_truth']['TP_retention'], .4)
        self.assertEqual(decision['aggregate']['filter_induced_true_loss'], 18)
        self.assertEqual(decision['aggregate']['filtered_FN'], 18)

    def test_pooled_success_cannot_hide_missing_or_supported_failure(self):
        for problem in ('empty_arm', 'unsupported_only', 'no_supported_solver_truth', 'absent_arm'):
            with self.subTest(problem=problem):
                cases = [_metric_case(kind, selected=tuple(range(10))) for kind in runner.KINDS]
                if problem == 'empty_arm':
                    cases[1] = _metric_case(runner.KINDS[1], selected=())
                elif problem == 'unsupported_only':
                    cases[1] = _metric_case(runner.KINDS[1], selected=(5, 6, 7, 8, 9))
                elif problem == 'no_supported_solver_truth':
                    cases[1]['records'] = [r for r in cases[1]['records'] if not r['supported_perturbed_truth']]
                else:
                    cases = cases[:2]
                self.assertGreaterEqual(runner.metrics(cases, .01)['TP_retention'], .4)
                self.assertFalse(runner.go_rule(cases, .01)['go'])

    def test_risk_empty_sets_and_numerical_abstention(self):
        cases = [_metric_case(kind, selected=tuple(range(10))) for kind in runner.KINDS]
        cases[0]['records'].append(_bound_row('FALSE', truth=False, index=10))
        self.assertGreater(runner.metrics(cases, .01)['FDP'], .01)
        self.assertFalse(runner.go_rule(cases, .01)['go'])
        self.assertFalse(runner.go_rule([], .01)['go'])
        self.assertIsNone(runner.metrics([], .01)['FDP'])
        for case in cases:
            for row in case['records']:
                row['deleted'].update(status='NUMERICALLY_UNRESOLVED', lower=None, upper=None)
        decision = runner.go_rule(cases, .01)
        self.assertFalse(decision['go'])
        self.assertEqual(decision['aggregate']['retained_count'], 0)
        self.assertEqual(decision['aggregate']['filtered_FN'], 30)
        self.assertEqual(decision['aggregate']['raw_solver_TP'], 30)
        self.assertEqual(decision['aggregate']['TP_retention'], 0.)
        self.assertEqual(runner.calibrated(cases)['status'], 'EMPTY_CALIBRATION')


if __name__ == '__main__':
    unittest.main()
