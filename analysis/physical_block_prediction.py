"""Physical-envelope blocks and fixed-library held-block NNLS predictions.

No truth labels, full-data coefficient estimates, scoring, calibration or file
I/O enter these functions. The existing NNLS primitive and KKT checker are
reused unchanged. Parallel callers must use separate processes: the original
NNLS primitive stores its current design matrix in process-global state.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

if __package__:
    from . import run_nnls_solver_baseline as baseline
    from . import run_small_mismatch_nnls_first_case as engine
else:
    import run_nnls_solver_baseline as baseline
    import run_small_mismatch_nnls_first_case as engine


KKT_FIELDS = ("max_dual_violation", "max_complementarity", "max_bound_ratio")


def _molecular_groups(candidate_names: Sequence[str], candidate_count: int):
    names = list(candidate_names)
    if len(names) != candidate_count or not names:
        raise ValueError("one name is required for each nonempty candidate column")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("candidate names must be nonempty strings")
    molecular_names = list(dict.fromkeys(names))
    lookup = {name: g for g, name in enumerate(molecular_names)}
    aliases = [[] for _ in molecular_names]
    for candidate, name in enumerate(names):
        aliases[lookup[name]].append(candidate)
    return names, molecular_names, aliases


def build_blocks(components, owner, candidate_names: Sequence[str]) -> dict:
    """Build positive-support connected components of observed channels.

    ``components`` is channels by physical components; ``owner`` assigns each
    physical component to a candidate column in ``candidate_names``. Every
    strictly positive entry participates, without intensity cutoffs. Envelopes
    that touch even one common channel belong to the same transitive block.
    Blocks and channels are ordered by their minimum/original channel index.
    Uncovered channels remain singleton blocks and are explicitly listed. Empty
    component supports are recorded rather than inventing observed evidence.
    The generic function does not hard-code the real-library block count of 34.
    """
    values = np.asarray(components)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("components must be a nonempty channel-by-component matrix")
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("physical components must be finite and nonnegative")
    names, molecular_names, aliases = _molecular_groups(candidate_names, len(candidate_names))
    owners = np.asarray(owner)
    if owners.ndim != 1 or owners.size != values.shape[1]:
        raise ValueError("one owner index is required for each physical component")
    if owners.size and (owners.dtype.kind not in "iu" or (owners < 0).any() or (owners >= len(names)).any()):
        raise ValueError("owner indices must be integers in the candidate universe")
    owners = owners.astype(np.int64)
    channel_count, component_count = values.shape
    parent = np.arange(channel_count)

    def find(channel):
        while parent[channel] != channel:
            parent[channel] = parent[parent[channel]]
            channel = int(parent[channel])
        return int(channel)

    support = values > 0
    component_channels = []
    for component in range(component_count):
        channels = np.flatnonzero(support[:, component])
        component_channels.append(channels)
        if channels.size:
            for channel in channels[1:]:
                left, right = find(int(channels[0])), find(int(channel))
                if left != right:
                    parent[max(left, right)] = min(left, right)
    groups = {}
    for channel in range(channel_count):
        groups.setdefault(find(channel), []).append(channel)
    channel_groups = sorted(groups.values(), key=lambda channels: channels[0])
    channel_to_block = np.empty(channel_count, dtype=np.int64)
    for block, channels in enumerate(channel_groups):
        channel_to_block[channels] = block
    component_to_block = np.full(component_count, -1, dtype=np.int64)
    candidate_blocks = [set() for _ in names]
    for component, channels in enumerate(component_channels):
        if channels.size:
            blocks = np.unique(channel_to_block[channels])
            if blocks.size != 1:
                raise RuntimeError("physical envelope was split between blocks")
            block = int(blocks[0])
            component_to_block[component] = block
            candidate_blocks[int(owners[component])].add(block)
    molecular_blocks = [sorted(set().union(*(candidate_blocks[i] for i in group))) for group in aliases]
    blocks = []
    for block, channels in enumerate(channel_groups):
        component_indices = np.flatnonzero(component_to_block == block).tolist()
        candidate_indices = sorted({int(owners[i]) for i in component_indices})
        molecular_indices = [g for g, memberships in enumerate(molecular_blocks) if block in memberships]
        blocks.append(dict(block_index=block, channel_indices=channels,
                           component_indices=component_indices, candidate_indices=candidate_indices,
                           molecular_indices=molecular_indices))
    return dict(schema_version=1, channel_count=channel_count, component_count=component_count,
                candidate_count=len(names), molecular_count=len(molecular_names), block_count=len(blocks),
                candidate_names=names, molecular_names=molecular_names, aliases=aliases, blocks=blocks,
                channel_to_block=channel_to_block.tolist(), component_to_block=component_to_block.tolist(),
                candidate_block_indices=[sorted(group) for group in candidate_blocks],
                molecular_block_indices=molecular_blocks,
                uncovered_channel_indices=np.flatnonzero(~support.any(axis=1)).tolist(),
                empty_component_indices=np.flatnonzero(~support.any(axis=0)).tolist(),
                support_rule="strictly positive physical-component entries; transitive channel closure",
                statistical_independence_claim=False)


def solve_block(A, b, names: Sequence[str], held_indices) -> dict:
    """Fit full/delete-all-alias models on training rows and predict held rows.

    Returns ``arrays`` for NPZ storage and JSON-compatible ``diagnostics``.
    Losses are unweighted, unnormalized squared L2 residuals. ``deleted_x`` is
    molecular-by-original-candidate, ordered by first occurrence of each name.
    Unsupported training columns are excluded from numerical solving and filled
    with zero; this convention is explicit and does not identify them as false.
    The original KKT checker is applied after zero filling to all allowed
    training columns, including zero columns. A zero-column allowed model is
    the sole analytic empty case, because the original checker uses max().

    No retry, full-data warm start, column/row normalization or coefficient
    threshold is added. In particular KKT validity does not certify unique
    coefficients, unique identities, or identifiability after withholding rows.
    """
    matrix = np.asarray(A, dtype=np.float64)
    observation = np.asarray(b, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) == 0 or observation.shape != (matrix.shape[0],):
        raise ValueError("A must be channel-by-candidate and b a matching channel vector")
    if not np.isfinite(matrix).all() or not np.isfinite(observation).all() or (matrix < 0).any():
        raise ValueError("A must be finite nonnegative and b finite")
    candidate_names, molecular_names, aliases = _molecular_groups(names, matrix.shape[1])
    held = np.asarray(held_indices)
    if held.ndim != 1 or held.size == 0 or held.dtype.kind not in "iu":
        raise ValueError("held_indices must be a nonempty integer vector")
    if (held < 0).any() or (held >= matrix.shape[0]).any() or len(np.unique(held)) != len(held):
        raise ValueError("held channel indices must be unique and in range")
    held = np.sort(held.astype(np.int64))
    train = np.setdiff1d(np.arange(matrix.shape[0]), held, assume_unique=True)
    if not train.size:
        raise ValueError("at least one training channel must remain")
    train_A, train_b = matrix[train], observation[train]
    held_A, held_b = matrix[held], observation[held]
    candidate_supported = np.any(train_A > 0, axis=0)
    candidate_held_supported = np.any(held_A > 0, axis=0)
    molecular_supported = np.array([candidate_supported[group].any() for group in aliases], dtype=bool)
    molecular_held_supported = np.array([candidate_held_supported[group].any() for group in aliases], dtype=bool)
    candidate_count = matrix.shape[1]
    solver_calls = 0

    def fit(allowed):
        nonlocal solver_calls
        solution = np.zeros(candidate_count, dtype=np.float64)
        eligible = allowed[candidate_supported[allowed]]
        if eligible.size:
            baseline.initialize_worker(train_A[:, eligible])
            fitted = np.asarray(baseline.solve_pixel(train_b), dtype=np.float64)
            solver_calls += 1
            if fitted.shape != (len(eligible),) or not np.isfinite(fitted).all() or (fitted < 0).any():
                raise RuntimeError("original NNLS returned invalid coefficients")
            solution[eligible] = fitted
        if allowed.size:
            kkt = engine.kkt_check(train_A[:, allowed], train_b, solution[allowed], np)
            kkt_mode = "ORIGINAL_CHECKER_ALL_ALLOWED_COLUMNS"
        else:
            kkt = {field: 0.0 for field in KKT_FIELDS}
            kkt_mode = "EMPTY_ALLOWED_MODEL_ANALYTIC"
        train_residual = train_A @ solution - train_b
        held_residual = held_A @ solution - held_b
        train_loss = float(np.dot(train_residual, train_residual))
        held_loss = float(np.dot(held_residual, held_residual))
        if not np.isfinite([train_loss, held_loss, *kkt.values()]).all():
            raise RuntimeError("nonfinite prediction loss or original KKT diagnostic")
        diagnostic = dict(KKT=kkt, KKT_mode=kkt_mode, allowed_candidate_count=int(allowed.size),
                          solved_candidate_count=int(eligible.size),
                          unsupported_allowed_candidate_indices=allowed[~candidate_supported[allowed]].tolist(),
                          numerical_solver_called=bool(eligible.size),
                          empty_supported_model=not bool(eligible.size))
        return solution, train_loss, held_loss, diagnostic

    all_indices = np.arange(candidate_count)
    full_x, full_train_loss, full_held_loss, full_diagnostic = fit(all_indices)
    deleted_x = np.zeros((len(molecular_names), candidate_count), dtype=np.float64)
    deleted_train_loss = np.empty(len(molecular_names), dtype=np.float64)
    deleted_held_loss = np.empty(len(molecular_names), dtype=np.float64)
    deleted_diagnostics = []
    for group, alias in enumerate(aliases):
        allowed = np.setdiff1d(all_indices, alias, assume_unique=True)
        x, training_loss, held_loss, diagnostic = fit(allowed)
        if np.any(x[alias] != 0) or np.any(x[~candidate_supported] != 0):
            raise RuntimeError("deleted aliases or unsupported columns were not zero filled")
        deleted_x[group], deleted_train_loss[group], deleted_held_loss[group] = x, training_loss, held_loss
        deleted_diagnostics.append(dict(molecular_index=group, molecular_name=molecular_names[group],
                                        deleted_candidate_indices=alias, **diagnostic))
    checks = [full_diagnostic["KKT"]] + [d["KKT"] for d in deleted_diagnostics]
    kkt_max = {field: max(d[field] for d in checks) for field in KKT_FIELDS}
    arrays = dict(full_x=full_x, deleted_x=deleted_x, deleted_names=np.asarray(molecular_names, dtype=str),
                  full_train_loss=np.asarray(full_train_loss, dtype=np.float64),
                  deleted_train_loss=deleted_train_loss,
                  full_held_loss=np.asarray(full_held_loss, dtype=np.float64),
                  deleted_held_loss=deleted_held_loss,
                  candidate_train_supported=candidate_supported,
                  candidate_held_supported=candidate_held_supported,
                  molecular_train_supported=molecular_supported,
                  molecular_held_supported=molecular_held_supported,
                  full_kkt=np.array([full_diagnostic["KKT"][field] for field in KKT_FIELDS]),
                  deleted_kkt=np.array([[d["KKT"][field] for field in KKT_FIELDS] for d in deleted_diagnostics]))
    if not all(np.isfinite(value).all() for key, value in arrays.items() if key != "deleted_names"):
        raise RuntimeError("nonfinite output array")
    diagnostics = dict(schema_version=1, candidate_names=candidate_names, molecular_names=molecular_names,
                       aliases=aliases, train_indices=train.tolist(), held_indices=held.tolist(),
                       unsupported_training_candidate_indices=np.flatnonzero(~candidate_supported).tolist(),
                       unsupported_training_molecular_indices=np.flatnonzero(~molecular_supported).tolist(),
                       full=full_diagnostic, deleted=deleted_diagnostics, KKT_fields=list(KKT_FIELDS), KKT_max=kkt_max,
                       numerical_solver_calls=solver_calls, requested_models=1+len(molecular_names),
                       nnls_primitive="run_nnls_solver_baseline.solve_pixel; scipy.nnls maxiter=3910",
                       loss="unweighted unnormalized squared L2; original nominal column values",
                       coefficient_precision="float64", retries=0,
                       no_training_support_policy="zero coefficient / explicit unevaluable flag, not false identity",
                       coefficient_or_identity_uniqueness_claim=False,
                       nonuniqueness_not_certified=True,
                       training_design_shape=list(train_A.shape),
                       supported_training_design_shape=[int(train.size), int(candidate_supported.sum())],
                       rank_not_certified=True,
                       limitation="KKT is the original floating-point engineering check; rank deficiency, aliases and multicolumn substitutes can leave coefficients and identities nonunique")
    return dict(arrays=arrays, diagnostics=diagnostics)
