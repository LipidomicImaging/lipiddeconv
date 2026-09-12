"""MODEL-only finite physical endpoints and conservative identity-deletion proofs.

The expanded endpoint cone is a relaxation, not the physical uncertainty model.
Only a refitted, common-CE-pair, one-endpoint physical library supplies the full
model upper bound. The existing L1 LP implementation supplies numerical proofs;
the discrete witness proposal makes no global optimality claim.
"""
from collections import defaultdict
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


GAMMA_NUM = 1e-6
BOUND_GUARD = 1e-8
PROOF_VERSION = "PHYSICAL_IDENTITY_ENDPOINT_PROOF_V1"
NOMINAL = "NOMINAL"


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _array_hash(value):
    value = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _problem_class():
    """Import the original CPU LP without changing the caller's GPU environment."""
    source = Path(__file__).with_name("run_ce_uncertainty_identity_pilot.py")
    spec = importlib.util.spec_from_file_location("_physical_identity_original_lp", source)
    module = importlib.util.module_from_spec(spec)
    keys = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        spec.loader.exec_module(module)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return module.Problem


def _normalize_b(b, channels):
    b = np.asarray(b, dtype=np.float64)
    if b.shape != (channels,) or not np.isfinite(b).all() or np.any(b < 0) or b.sum() <= 0:
        raise ValueError("INVALID_GLOBAL_FOREGROUND_SPECTRUM")
    return b / b.sum()


def _metadata_row(metadata, original, local, kept):
    if isinstance(metadata, dict):
        return metadata[original] if original in metadata else metadata[str(original)]
    if len(metadata) > max(kept):
        return metadata[original]
    if len(metadata) == len(kept):
        return metadata[local]
    raise ValueError("METADATA_DOES_NOT_MATCH_KEPT_CANDIDATES")


def build_model_bank(A, fractions, physical_contract_patterns, mapping,
                     model_pattern_ids, metadata, kept):
    """Build inference atoms from MODEL IDs only; A columns follow kept order.

    ``fractions`` has the original component ordering used by candidate_mapping.
    Neither target donor IDs, target spectra nor truth labels are accepted.
    Source pattern records may contain the frozen universe; only explicitly
    selected MODEL records are retained in the bank and its fingerprint.
    """
    return ModelBank(A, fractions, physical_contract_patterns, mapping,
                     model_pattern_ids, metadata, kept)


class ModelBank:
    def __init__(self, A, fractions, patterns, mapping, model_pattern_ids, metadata, kept):
        A = np.array(A, dtype=np.float64, copy=True)
        fractions = np.asarray(fractions, dtype=np.float64)
        kept = [int(k) for k in kept]
        if (A.ndim != 2 or A.shape[1] != len(kept) or not kept
                or len(set(kept)) != len(kept) or not np.isfinite(A).all()
                or np.any(A < 0) or np.any(A.sum(axis=0) <= 0)):
            raise ValueError("INVALID_NOMINAL_LIBRARY_OR_KEPT_INDICES")
        if (fractions.ndim != 2 or fractions.shape[0] != A.shape[0]
                or not np.isfinite(fractions).all() or np.any(fractions < 0)):
            raise ValueError("INVALID_COMPONENT_FRACTIONS")
        all_ids = [p["pattern_id"] for p in patterns]
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("DUPLICATE_PHYSICAL_PATTERN_ID")
        requested = list(model_pattern_ids)
        if len(set(requested)) != len(requested) or not set(requested).issubset(all_ids):
            raise ValueError("INVALID_MODEL_PATTERN_MEMBERSHIP")
        # No CAL/EVAL endpoint, identity or source-file data survive this filter.
        self.patterns = {p["pattern_id"]: json.loads(json.dumps(p)) for p in patterns
                         if p["pattern_id"] in set(requested)}
        self.model_pattern_ids = tuple(sorted(requested))
        by_candidate = {int(row["candidate_index"]): row for row in mapping}
        if len(by_candidate) != len(mapping):
            raise ValueError("DUPLICATE_CANDIDATE_MAPPING")
        self.names = tuple(str(_metadata_row(metadata, j, i, kept)["lipid_name"])
                           for i, j in enumerate(kept))
        if any(not name for name in self.names):
            raise ValueError("EMPTY_MOLECULAR_IDENTITY")
        self.kept = tuple(kept)
        self.A = A
        self.physical_groups = defaultdict(list)
        self._candidate_group = {}
        active_mapping = []
        for local, original in enumerate(kept):
            row = by_candidate.get(original)
            if row is None:
                continue
            if row["lipid_name"] != self.names[local]:
                raise ValueError("CANDIDATE_MAPPING_IDENTITY_MISMATCH")
            channels = tuple(row["channels"])
            components = tuple(int(i) for i in row["component_indices"])
            if (len(channels) != len(components) or len(set(channels)) != len(channels)
                    or not components or min(components) < 0 or max(components) >= fractions.shape[1]):
                raise ValueError("INVALID_PHYSICAL_COMPONENT_MAPPING")
            group = (self.names[local], str(row["rule"]), channels)
            self.physical_groups[group].append(local)
            self._candidate_group[local] = group
            active_mapping.append(dict(candidate_index=original, lipid_name=self.names[local],
                                       rule=row["rule"], channels=list(channels),
                                       component_indices=list(components),
                                       allowed_pattern_ids=sorted(set(row["allowed_pattern_ids"])
                                                                  & set(requested))))
        # Nominal atoms come first in exact kept-column order.
        atom_columns = [A[:, local] for local in range(len(kept))]
        self.atoms = [dict(atom_index=local, local_candidate_index=local,
                           candidate_index=original, lipid_name=self.names[local],
                           pattern_id=NOMINAL, ce_pair=None)
                      for local, original in enumerate(kept)]
        self._candidate_atoms = {local: {NOMINAL: local} for local in range(len(kept))}
        for row in active_mapping:
            local = kept.index(row["candidate_index"])
            component = fractions[:, row["component_indices"]] * A[:, local, None]
            fixed = A[:, local] - component.sum(axis=1)
            if np.min(fixed) < -1e-10:
                raise ValueError("COMPONENTS_EXCEED_NOMINAL_LIBRARY")
            # This reproduces the existing component numerical guard only.
            fixed = np.maximum(fixed, 0)
            if np.max(abs(fixed + component.sum(axis=1) - A[:, local])) > 1e-10:
                raise ValueError("NOMINAL_COMPONENT_RECONSTRUCTION_FAILED")
            for pattern_id in row["allowed_pattern_ids"]:
                p = self.patterns[pattern_id]
                pair = tuple(float(v) for v in p["ce_pair"])
                delta = np.asarray(p["centered_log_change"], dtype=np.float64)
                multipliers = np.asarray(p["multiplier"], dtype=np.float64)
                if (list(p["channels"]) != row["channels"] or p["donor_identity"][2] != row["rule"]
                        or len(pair) != 2 or not pair[0] < pair[1]
                        or delta.shape != (len(row["channels"]),)
                        or not np.isfinite(delta).all() or not np.isfinite(multipliers).all()
                        or multipliers.shape != delta.shape or np.any(multipliers <= 0)
                        or not np.allclose(np.exp(delta), multipliers, rtol=1e-13, atol=1e-15)):
                    raise ValueError("INVALID_WHOLE_FORWARD_CE_ENDPOINT")
                target = fixed + component @ multipliers
                target *= np.linalg.norm(A[:, local]) / np.linalg.norm(target)
                if np.any(target < 0) or np.any(target[A[:, local] == 0] != 0):
                    raise ValueError("ENDPOINT_INTRODUCES_NEW_SUPPORT")
                if np.any(target[A[:, local] > 0] <= 0):
                    raise ValueError("ENDPOINT_REMOVES_POSITIVE_SUPPORT")
                atom_index = len(atom_columns)
                atom_columns.append(target)
                self._candidate_atoms[local][pattern_id] = atom_index
                self.atoms.append(dict(atom_index=atom_index, local_candidate_index=local,
                                       candidate_index=kept[local], lipid_name=self.names[local],
                                       pattern_id=pattern_id, ce_pair=list(pair)))
        for group, local_indices in self.physical_groups.items():
            choices = [set(self._candidate_atoms[i]) for i in local_indices]
            if any(value != choices[0] for value in choices[1:]):
                raise ValueError("SAME_NAME_RULE_CHANNEL_ALIASES_HAVE_DIFFERENT_ENDPOINT_SETS")
        self.matrix = np.column_stack(atom_columns)
        self.matrix.setflags(write=False)
        self.A.setflags(write=False)
        self.ce_pairs = tuple(sorted({tuple(a["ce_pair"]) for a in self.atoms if a["ce_pair"] is not None}))
        self._pair_atoms = {pair: tuple(a["atom_index"] for a in self.atoms
                                       if a["pattern_id"] == NOMINAL or tuple(a["ce_pair"]) == pair)
                            for pair in self.ce_pairs}
        self._all_atoms = tuple(range(len(self.atoms)))
        self.manifest = dict(proof_version=PROOF_VERSION, model_pattern_ids=list(self.model_pattern_ids),
                             model_patterns=[self.patterns[k] for k in self.model_pattern_ids],
                             kept=list(kept), lipid_names=list(self.names), mapping=active_mapping,
                             atoms=self.atoms, A_sha256=_array_hash(A),
                             endpoint_matrix_sha256=_array_hash(self.matrix),
                             nominal_component_fraction_sha256=_array_hash(fractions),
                             lp_source_sha256=hashlib.sha256(Path(__file__).with_name(
                                 "run_ce_uncertainty_identity_pilot.py").read_bytes()).hexdigest(),
                             full_witness_search="NOMINAL_AND_ONE_MASS_ROUNDING_REFIT_PER_COMMON_CE_PAIR",
                             deletion_model="ALL_MODEL_ENDPOINT_CONE_RELAXATION",
                             original_physical_model_global_optimality_claimed=False)
        self.fingerprint = _json_hash(self.manifest)
        self._b = None
        self._global_solution = None
        self._global_problem = None

    def __getstate__(self):
        # Spawn workers receive the immutable bank and cached proof, not a
        # dynamically imported Problem object. Sparse constraints rebuild once.
        state = self.__dict__.copy()
        state["_global_problem"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._global_problem = None
        self.matrix.setflags(write=False)
        self.A.setflags(write=False)

    def _problem(self, atom_indices, b):
        matrix = self.matrix[:, list(atom_indices)]
        return _problem_class()(matrix, np.zeros((matrix.shape[0], 0)), np.array([], dtype=int),
                                np.array([]), np.array([]), b)

    def _base_record(self, b):
        return dict(proof_version=PROOF_VERSION, bank_fingerprint=self.fingerprint,
                    observation_sha256=_array_hash(b), bound_guard=BOUND_GUARD,
                    numeric_gap_tolerance=GAMMA_NUM)

    def _run_lp(self, key, atom_indices, kind, b, arrays, removed_atoms=(), ce_pair=None):
        problem = self._problem(atom_indices, b)
        inverse = {atom: i for i, atom in enumerate(atom_indices)}
        removed_local = [inverse[atom] for atom in removed_atoms]
        bounds, primal, dual = problem.solve(removed_local)
        arrays[key + "__primal"] = primal
        arrays[key + "__dual"] = dual
        record = dict(key=key, model_kind=kind, atom_indices=list(atom_indices),
                      removed_atoms=list(removed_atoms), ce_pair=list(ce_pair) if ce_pair else None,
                      **bounds)
        return record, problem, primal, dual

    def _round(self, pair_atoms, pair_primal):
        """One deterministic proposal; endpoint masses are summed across aliases."""
        atom_mass = dict(zip(pair_atoms, pair_primal[:len(pair_atoms)]))
        choices = list(range(len(self.kept)))
        for _, local_indices in sorted(self.physical_groups.items()):
            allowed = [pattern for pattern, atom in self._candidate_atoms[local_indices[0]].items()
                       if atom in atom_mass]
            masses = {p: float(sum(atom_mass[self._candidate_atoms[i][p]] for i in local_indices))
                      for p in allowed}
            # Nominal wins exact ties; otherwise pattern ID order is frozen.
            selected = min(allowed, key=lambda p: (-masses[p], p != NOMINAL, p))
            for i in local_indices:
                choices[i] = self._candidate_atoms[i][selected]
        return choices

    def _validate_physical_selection(self, atom_indices, pair):
        if len(atom_indices) != len(self.kept):
            raise ValueError("PHYSICAL_WITNESS_REQUIRES_ONE_ATOM_PER_CANDIDATE")
        patterns = []
        for local, atom in enumerate(atom_indices):
            if not isinstance(atom, int) or not 0 <= atom < len(self.atoms):
                raise ValueError("INVALID_PHYSICAL_WITNESS_ATOM")
            item = self.atoms[atom]
            if item["local_candidate_index"] != local:
                raise ValueError("PHYSICAL_WITNESS_CANDIDATE_ORDER_CHANGED")
            if item["pattern_id"] != NOMINAL and (pair is None or tuple(item["ce_pair"]) != tuple(pair)):
                raise ValueError("PHYSICAL_WITNESS_MIXES_CE_DIRECTIONS")
            patterns.append(item["pattern_id"])
        for local_indices in self.physical_groups.values():
            if len({patterns[i] for i in local_indices}) != 1:
                raise ValueError("PHYSICAL_WITNESS_MIXES_SAME_IDENTITY_ENDPOINTS")

    def score_full(self, b):
        """Return JSON bounds and NPZ-ready proof arrays; no thresholds or truth."""
        b = _normalize_b(b, self.A.shape[0])
        self._b = b.copy()
        self._global_solution = None
        self._global_problem = None
        arrays = {"normalized_b": b.copy()}
        record = dict(**self._base_record(b), status="NUMERICALLY_UNRESOLVED", lower=None,
                      upper=None, lp_records=[], selected_physical_witness=None,
                      lower_bound_model="ALL_MODEL_ENDPOINT_CONE_RELAXATION",
                      upper_bound_model="PHYSICAL_COMMON_PAIR_SINGLE_ENDPOINT_WITNESS",
                      physical_global_optimum_claimed=False)
        try:
            relaxed, problem, primal, dual = self._run_lp(
                "all_relaxed", self._all_atoms, "ALL_ENDPOINT_RELAXATION", b, arrays)
            record["lp_records"].append(relaxed)
            self._global_problem = problem
            self._global_solution = (relaxed, primal, dual)
            nominal, _, _, _ = self._run_lp("nominal", tuple(range(len(self.kept))),
                                           "PHYSICAL_WITNESS", b, arrays)
            record["lp_records"].append(nominal)
            witnesses = [nominal]
            for number, pair in enumerate(self.ce_pairs):
                key = "pair_%03d" % number
                pair_atoms = self._pair_atoms[pair]
                proposal, _, pair_primal, _ = self._run_lp(
                    key + "_relaxed", pair_atoms, "COMMON_PAIR_RELAXATION", b, arrays, ce_pair=pair)
                record["lp_records"].append(proposal)
                selected = self._round(pair_atoms, pair_primal)
                self._validate_physical_selection(selected, pair)
                witness, _, _, _ = self._run_lp(key + "_physical", selected, "PHYSICAL_WITNESS", b,
                                               arrays, ce_pair=pair)
                record["lp_records"].append(witness)
                witnesses.append(witness)
            best = min(witnesses, key=lambda row: (row["upper"], row["key"]))
            if relaxed["lower"] > best["upper"]:
                raise RuntimeError("RELAXATION_LOWER_EXCEEDS_PHYSICAL_WITNESS")
            record.update(status="BOUNDS_VALID", lower=relaxed["lower"], upper=best["upper"],
                          relaxation_upper=relaxed["upper"], selected_physical_witness=best["key"])
        except (AssertionError, RuntimeError, ValueError) as exc:
            record["numerical_error"] = repr(exc)
            self._global_solution = None
            self._global_problem = None
        return record, arrays

    def score_delete(self, removed_names):
        """Delete every same-name nominal and MODEL atom from the cone relaxation."""
        if self._b is None:
            raise ValueError("SCORE_FULL_MUST_FREEZE_OBSERVATION_BEFORE_DELETION")
        if isinstance(removed_names, str):
            removed_names = [removed_names]
        removed_names = sorted(set(removed_names))
        if not removed_names or not set(removed_names).issubset(self.names):
            raise ValueError("UNKNOWN_OR_EMPTY_MOLECULAR_DELETION")
        removed_atoms = [a["atom_index"] for a in self.atoms if a["lipid_name"] in removed_names]
        arrays = {"normalized_b": self._b.copy()}
        record = dict(**self._base_record(self._b), status="NUMERICALLY_UNRESOLVED", lower=None,
                      upper=None, removed_names=removed_names, removed_atoms=removed_atoms,
                      lower_bound_model="ALL_MODEL_ENDPOINT_CONE_RELAXATION",
                      upper_bound_model="ENDPOINT_CONE_RELAXATION_NOT_PHYSICAL_WITNESS", lp_records=[])
        try:
            if self._global_solution is None:
                raise RuntimeError("FULL_MODEL_NUMERICAL_PROOF_UNAVAILABLE")
            if self._global_problem is None:
                self._global_problem = self._problem(self._all_atoms, self._b)
            previous, primal, dual = self._global_solution
            if np.all(primal[removed_atoms] == 0):
                upper = self._global_problem.upper.copy()
                upper[removed_atoms] = 0
                lower, high = self._global_problem.check_bounds(primal, dual, upper)
                if high - lower > GAMMA_NUM:
                    raise RuntimeError("ZERO_MASS_PROOF_GAP_EXCEEDS_CONTRACT")
                arrays["deleted__primal"] = primal.copy()
                arrays["deleted__dual"] = dual.copy()
                result = dict(key="deleted", model_kind="ALL_ENDPOINT_DELETION_RELAXATION",
                              atom_indices=list(self._all_atoms), removed_atoms=removed_atoms,
                              ce_pair=None, lower=lower, upper=high, seconds=0.,
                              backend_attempts=[], proof=previous["proof"],
                              reuse="FULL_RELAXATION_EXACT_ZERO_MASS")
            else:
                result, _, _, _ = self._run_lp("deleted", self._all_atoms,
                                               "ALL_ENDPOINT_DELETION_RELAXATION", self._b,
                                               arrays, removed_atoms=removed_atoms)
                result["reuse"] = None
            record["lp_records"].append(result)
            record.update(status="BOUNDS_VALID", lower=result["lower"], upper=result["upper"])
        except (AssertionError, RuntimeError, ValueError) as exc:
            record["numerical_error"] = repr(exc)
        return record, arrays

    def _verify_header(self, b, record, arrays):
        b = _normalize_b(b, self.A.shape[0])
        for key, value in self._base_record(b).items():
            if record.get(key) != value:
                raise ValueError("PROOF_HEADER_MISMATCH: " + key)
        if not np.array_equal(np.asarray(arrays["normalized_b"]), b):
            raise ValueError("PROOF_OBSERVATION_CHANGED")
        if record.get("status") not in ("BOUNDS_VALID", "NUMERICALLY_UNRESOLVED"):
            raise ValueError("INVALID_NUMERICAL_PROOF_STATUS")
        return b

    def _verify_lp(self, b, item, arrays):
        indices = item["atom_indices"]
        if len(indices) != len(set(indices)) or any(i not in self._all_atoms for i in indices):
            raise ValueError("INVALID_PROOF_ATOM_MEMBERSHIP")
        problem = self._problem(indices, b)
        inverse = {atom: i for i, atom in enumerate(indices)}
        upper = problem.upper.copy()
        upper[[inverse[i] for i in item["removed_atoms"]]] = 0
        primal = np.asarray(arrays[item["key"] + "__primal"], dtype=np.float64)
        dual = np.asarray(arrays[item["key"] + "__dual"], dtype=np.float64)
        if primal.shape != problem.upper.shape or dual.shape != problem.rhs.shape:
            raise ValueError("PROOF_ARRAY_SHAPE_MISMATCH")
        lower, high = problem.check_bounds(primal, dual, upper)
        if high - lower > GAMMA_NUM:
            raise ValueError("PROOF_GAP_EXCEEDS_CONTRACT")
        if abs(lower - item["lower"]) > 1e-12 or abs(high - item["upper"]) > 1e-12:
            raise ValueError("REPORTED_LP_BOUNDS_CHANGED")
        return dict(lower=lower, upper=high, gap=high - lower)

    def verify_full(self, b, record, arrays):
        """Rebuild all convex proofs and validate each finite physical witness."""
        b = self._verify_header(b, record, arrays)
        if (record["lower_bound_model"] != "ALL_MODEL_ENDPOINT_CONE_RELAXATION"
                or record["upper_bound_model"] != "PHYSICAL_COMMON_PAIR_SINGLE_ENDPOINT_WITNESS"
                or record["physical_global_optimum_claimed"] is not False):
            raise ValueError("FULL_PROOF_MODEL_CLAIM_CHANGED")
        if record["status"] != "BOUNDS_VALID":
            if record.get("lower") is not None or record.get("upper") is not None:
                raise ValueError("UNRESOLVED_PROOF_HAS_SELECTION_BOUNDS")
            return dict(status="ABSTENTION_ONLY", reason="NO_COMPLETE_FULL_NUMERICAL_PROOF")
        expected = ["all_relaxed", "nominal"]
        for number in range(len(self.ce_pairs)):
            expected.extend(["pair_%03d_relaxed" % number, "pair_%03d_physical" % number])
        rows = record["lp_records"]
        if [item["key"] for item in rows] != expected:
            raise ValueError("FULL_PROOF_SEARCH_MEMBERSHIP_CHANGED")
        checked = {}
        witnesses = []
        for item in rows:
            key = item["key"]
            if item["removed_atoms"]:
                raise ValueError("FULL_PROOF_DELETES_CANDIDATES")
            if key == "all_relaxed":
                if tuple(item["atom_indices"]) != self._all_atoms or item["model_kind"] != "ALL_ENDPOINT_RELAXATION":
                    raise ValueError("FULL_RELAXATION_MEMBERSHIP_CHANGED")
            elif key == "nominal":
                if item["atom_indices"] != list(range(len(self.kept))) or item["ce_pair"] is not None:
                    raise ValueError("NOMINAL_WITNESS_CHANGED")
            elif key.endswith("_relaxed"):
                number = int(key.split("_")[1])
                pair = self.ce_pairs[number]
                if (tuple(item["atom_indices"]) != self._pair_atoms[pair] or tuple(item["ce_pair"]) != pair
                        or item["model_kind"] != "COMMON_PAIR_RELAXATION"):
                    raise ValueError("COMMON_PAIR_RELAXATION_CHANGED")
            else:
                number = int(key.split("_")[1])
                pair = self.ce_pairs[number]
                if tuple(item["ce_pair"]) != pair:
                    raise ValueError("PHYSICAL_WITNESS_CE_PAIR_CHANGED")
                proposal_key = "pair_%03d_relaxed" % number
                rounded = self._round(self._pair_atoms[pair], arrays[proposal_key + "__primal"])
                if rounded != item["atom_indices"]:
                    raise ValueError("PHYSICAL_WITNESS_ROUNDING_CHANGED")
            if key == "nominal" or key.endswith("_physical"):
                if item["model_kind"] != "PHYSICAL_WITNESS":
                    raise ValueError("RELAXATION_CANNOT_BE_A_PHYSICAL_WITNESS")
                self._validate_physical_selection(item["atom_indices"], item["ce_pair"])
                witnesses.append(item)
            checked[key] = self._verify_lp(b, item, arrays)
        best = min(witnesses, key=lambda item: (item["upper"], item["key"]))
        if (record["selected_physical_witness"] != best["key"]
                or record["lower"] != rows[0]["lower"] or record["upper"] != best["upper"]
                or record["relaxation_upper"] != rows[0]["upper"]
                or record["physical_global_optimum_claimed"] is not False):
            raise ValueError("FULL_BOUND_SELECTION_CHANGED")
        return dict(status="PASS", lp_proofs=len(checked),
                    max_numerical_gap=max(item["gap"] for item in checked.values()),
                    physical_witness_verified=True, global_physical_optimum_claimed=False)

    def restore_full(self, b, record, arrays):
        """Resume from independently verified saved proofs without any LP solve."""
        self._b = None
        self._global_problem = None
        self._global_solution = None
        verification = self.verify_full(b, record, arrays)
        self._b = _normalize_b(b, self.A.shape[0])
        if verification["status"] == "PASS":
            relaxed = next(row for row in record["lp_records"] if row["key"] == "all_relaxed")
            self._global_solution = (json.loads(json.dumps(relaxed)),
                                     np.array(arrays["all_relaxed__primal"], copy=True),
                                     np.array(arrays["all_relaxed__dual"], copy=True))
            self._global_problem = self._problem(self._all_atoms, self._b)
        return verification

    def verify_delete(self, b, removed_names, record, arrays):
        b = self._verify_header(b, record, arrays)
        if (record["lower_bound_model"] != "ALL_MODEL_ENDPOINT_CONE_RELAXATION"
                or record["upper_bound_model"] != "ENDPOINT_CONE_RELAXATION_NOT_PHYSICAL_WITNESS"):
            raise ValueError("DELETION_PROOF_MODEL_CLAIM_CHANGED")
        if isinstance(removed_names, str):
            removed_names = [removed_names]
        removed_names = sorted(set(removed_names))
        expected = [a["atom_index"] for a in self.atoms if a["lipid_name"] in removed_names]
        if not expected or record["removed_names"] != removed_names or record["removed_atoms"] != expected:
            raise ValueError("MOLECULAR_DELETION_MEMBERSHIP_CHANGED")
        if record["status"] != "BOUNDS_VALID":
            if record.get("lower") is not None or record.get("upper") is not None:
                raise ValueError("UNRESOLVED_PROOF_HAS_SELECTION_BOUNDS")
            return dict(status="ABSTENTION_ONLY", reason="NO_COMPLETE_DELETION_NUMERICAL_PROOF")
        if len(record["lp_records"]) != 1:
            raise ValueError("DELETION_PROOF_COUNT_CHANGED")
        item = record["lp_records"][0]
        if (item["key"] != "deleted" or item["atom_indices"] != list(self._all_atoms)
                or item["removed_atoms"] != expected or item["ce_pair"] is not None
                or item["model_kind"] != "ALL_ENDPOINT_DELETION_RELAXATION"):
            raise ValueError("DELETION_RELAXATION_CHANGED")
        checked = self._verify_lp(b, item, arrays)
        if record["lower"] != item["lower"] or record["upper"] != item["upper"]:
            raise ValueError("DELETION_BOUNDS_CHANGED")
        return dict(status="PASS", lp_proofs=1, max_numerical_gap=checked["gap"],
                    all_same_name_atoms_removed=True, upper_is_only_relaxation=True)


def classify(full, deleted, epsilon):
    """Threshold guard shared by CAL and EVAL; a relaxed fit is not replaceability proof."""
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("INVALID_EPSILON")
    if full.get("status") != "BOUNDS_VALID" or deleted.get("status") != "BOUNDS_VALID":
        return "NUMERICALLY_UNRESOLVED"
    if full["lower"] > epsilon + GAMMA_NUM:
        return "FULL_MODEL_INCOMPATIBLE"
    if full["upper"] > epsilon - GAMMA_NUM:
        return "THRESHOLD_UNRESOLVED"
    if deleted["lower"] > epsilon + GAMMA_NUM:
        return "RETAINED"
    if deleted["upper"] <= epsilon - GAMMA_NUM:
        return "RELAXATION_REPLACEABLE"
    return "THRESHOLD_UNRESOLVED"
