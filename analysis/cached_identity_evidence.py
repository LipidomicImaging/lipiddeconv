"""Observable identity diagnostics from existing held-block coefficient caches.

This module performs no fitting, file I/O or label access. Direct contribution
and redistribution describe a cached full/delete pair; neither is a proof of
identity correctness or unique causal attribution. Stability concerns the sum
of all same-name coefficients across nominally supported folds, including zero
coefficients. It does not establish that an identity is distinguishable.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _numeric(value, label, shape=None, nonnegative=False, float64=False):
    array = np.asarray(value)
    _require(array.dtype.kind in 'fiu' and (shape is None or array.shape == shape),
             label + ': invalid numeric shape/type')
    _require(not float64 or array.dtype == np.float64, label + ': expected float64 cache')
    _require(np.isfinite(array).all() and (not nonnegative or (array >= 0).all()),
             label + ': nonfinite or negative value')
    return array.astype(np.float64, copy=False)


def _manifest(manifest, shape, blocks):
    channels, candidates = shape
    names = list(manifest['candidate_names'])
    _require(len(names) == candidates and all(isinstance(n, str) and n for n in names),
             'invalid candidate names')
    molecular = list(dict.fromkeys(names))
    aliases = [[i for i, name in enumerate(names) if name == molecule] for molecule in molecular]
    _require(list(manifest['molecular_names']) == molecular, 'molecular order/membership mismatch')
    supplied_aliases = list(manifest['aliases'])
    _require(len(supplied_aliases) == len(aliases), 'alias group count mismatch')
    for supplied, expected in zip(supplied_aliases, aliases):
        supplied = np.asarray(supplied)
        _require(supplied.ndim == 1 and supplied.dtype.kind in 'iu'
                 and supplied.tolist() == expected, 'all-alias order/membership mismatch')
    entries = list(manifest['blocks'])
    _require(len(entries) == len(blocks) and len(entries) >= 2, 'block count mismatch or no training rows')
    for key, expected in (('channel_count', channels), ('candidate_count', candidates),
                          ('molecular_count', len(molecular)), ('block_count', len(entries))):
        if key in manifest:
            _require(manifest[key] == expected, 'manifest ' + key + ' mismatch')
    held_groups = []
    for index, block in enumerate(entries):
        held = np.asarray(block['channel_indices'])
        _require(block['block_index'] == index and held.ndim == 1 and held.dtype.kind in 'iu'
                 and 0 < held.size < channels and np.all(held >= 0) and np.all(held < channels),
                 'invalid held channel indices/order')
        _require(held.tolist() == sorted(set(held.tolist())), 'held channels must be sorted and unique')
        held_groups.append(held.astype(np.int64))
    flattened = np.concatenate(held_groups)
    _require(sorted(flattened.tolist()) == list(range(channels)), 'blocks must partition all channels')
    _require([int(h[0]) for h in held_groups] == sorted(int(h[0]) for h in held_groups),
             'blocks must be ordered by first channel')
    if 'channel_to_block' in manifest:
        mapping = np.empty(channels, dtype=np.int64)
        for index, held in enumerate(held_groups):
            mapping[held] = index
        _require(np.array_equal(np.asarray(manifest['channel_to_block']), mapping), 'channel mapping mismatch')
    return molecular, aliases, held_groups


def _saved_loss(value, expected, shape, signal, label):
    saved = _numeric(value, label, shape=shape, nonnegative=True)
    # Same scale-aware loss tolerance as the original cache reviewer. This is
    # numerical validation only, and is not a score or selection threshold.
    _require(np.allclose(saved, expected, rtol=1e-10, atol=1e-15 * max(signal, 1e-30)),
             label + ': cached loss disagrees with coefficients')
    return saved


def extract(A, b, manifest, blocks):
    """Return ``dict(rows=[...], arrays={...})`` in original molecular order.

    ``blocks`` contains the existing NPZ mappings in manifest block order.
    Required cache keys are full_x[N], deleted_x[G,N], deleted_names[G],
    full_held_loss[] and deleted_held_loss[G]. Coefficients must be float64.
    Optional cached support flags are checked when present. The bound original
    block reviewer remains responsible for the cached solutions' KKT checks.

    Six arrays have shape (block_count, molecular_count): direct,
    redistribution, delta, own_energy, fold_coefficient and boolean support.
    Delta preserves the original saved deleted-minus-full loss after tolerant
    reconstruction checks. Thus trimmed_gain preserves the original delta
    definition, and redistribution = delta - direct includes any roundoff.

    own_signal_energy is the unnormalized sum of own_energy. Its >0 flag is
    descriptive only. tau=128*eps64*max(A.shape)*||b||^2 is solely the additive
    denominator in direct_per_signal, and never a screening gate.
    """
    A = _numeric(A, 'A', nonnegative=True)
    _require(A.ndim == 2 and min(A.shape) > 0, 'A must be a nonempty matrix')
    b = _numeric(b, 'b', shape=(A.shape[0],))
    blocks = list(blocks)
    molecular, aliases, held_groups = _manifest(manifest, A.shape, blocks)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        signal = float(b @ b)
        _require(np.isfinite(signal) and signal > 0, 'positive finite observation energy required')
        tau = 128 * np.finfo(np.float64).eps * max(A.shape) * signal
        _require(np.isfinite(tau) and tau > 0, 'numerical denominator underflow/overflow')
        shape = (len(blocks), len(molecular))
        arrays = {key: np.zeros(shape, dtype=np.float64) for key in
                  ('direct', 'redistribution', 'delta', 'own_energy', 'fold_coefficient')}
        arrays['support'] = np.zeros(shape, dtype=bool)
        for index, (held, cache) in enumerate(zip(held_groups, blocks)):
            saved_names = np.asarray(cache['deleted_names'])
            _require(saved_names.shape == (len(molecular),) and saved_names.dtype.kind in 'US'
                     and saved_names.tolist() == molecular, 'deleted molecular order/membership mismatch')
            full = _numeric(cache['full_x'], 'full_x', shape=(A.shape[1],), nonnegative=True, float64=True)
            deleted = _numeric(cache['deleted_x'], 'deleted_x',
                               shape=(len(molecular), A.shape[1]), nonnegative=True, float64=True)
            train = np.setdiff1d(np.arange(A.shape[0]), held)
            train_supported = np.any(A[train] > 0, axis=0)
            held_supported = np.any(A[held] > 0, axis=0)
            _require(np.all(full[~train_supported] == 0) and np.all(deleted[:, ~train_supported] == 0),
                     'unsupported training column has nonzero coefficient')
            group_train = np.array([np.any(train_supported[group]) for group in aliases], dtype=bool)
            group_held = np.array([np.any(held_supported[group]) for group in aliases], dtype=bool)
            for key, expected in (('candidate_train_supported', train_supported),
                                  ('candidate_held_supported', held_supported),
                                  ('molecular_train_supported', group_train),
                                  ('molecular_held_supported', group_held)):
                if key in cache:
                    saved = np.asarray(cache[key])
                    _require(saved.dtype == bool and np.array_equal(saved, expected), 'cached support mismatch: ' + key)
            H = A[held]
            prediction = H @ full
            residual = b[held] - prediction
            deleted_prediction = H @ deleted.T
            own = np.empty((held.size, len(molecular)), dtype=np.float64)
            for group_index, group in enumerate(aliases):
                _require(np.all(deleted[group_index, group] == 0), 'deleted molecular alias remains nonzero')
                own[:, group_index] = H[:, group] @ full[group]
                arrays['fold_coefficient'][index, group_index] = float(np.sum(full[group], dtype=np.float64))
            change = deleted_prediction - prediction[:, None] + own
            deleted_residual = residual[:, None] + own - change
            full_loss = float(residual @ residual)
            deleted_loss = np.sum(deleted_residual * deleted_residual, axis=0, dtype=np.float64)
            saved_full = _saved_loss(cache['full_held_loss'], full_loss, (), signal, 'full_held_loss')
            saved_deleted = _saved_loss(cache['deleted_held_loss'], deleted_loss, (len(molecular),), signal,
                                        'deleted_held_loss')
            delta = saved_deleted - saved_full
            reconstructed_delta = deleted_loss - full_loss
            # Use loss scale rather than relative error in a potentially tiny
            # difference between two large losses.
            delta_bound = 1e-10 * (np.abs(deleted_loss) + abs(full_loss)) + 2e-15 * max(signal, 1e-30)
            _require(np.all(np.abs(delta - reconstructed_delta) <= delta_bound), 'cached delta reconstruction mismatch')
            own_energy = np.sum(own * own, axis=0, dtype=np.float64)
            direct = own_energy + 2 * (residual @ own) - np.sum(own * change, axis=0, dtype=np.float64)
            arrays['own_energy'][index] = own_energy
            arrays['direct'][index] = direct
            arrays['delta'][index] = delta
            arrays['redistribution'][index] = delta - direct
            arrays['support'][index] = group_train & group_held
        _require(all(np.isfinite(value).all() for value in arrays.values()), 'nonfinite derived array')
        rows = []
        for group_index, name in enumerate(molecular):
            delta = arrays['delta'][:, group_index]
            direct_sum = float(np.sum(arrays['direct'][:, group_index], dtype=np.float64))
            own_sum = float(np.sum(arrays['own_energy'][:, group_index], dtype=np.float64))
            support = arrays['support'][:, group_index]
            t = arrays['fold_coefficient'][support, group_index]
            informative = bool(t.size >= 2 and np.any(t > 0))
            stability = 0.
            if informative:
                # Rescaling t cancels algebraically and avoids squaring huge
                # or tiny coefficients. Supported zero folds remain included.
                scaled = t / np.max(t)
                stability = float(np.mean(scaled) ** 2 / np.mean(scaled * scaled))
                _require(0 <= stability <= 1 + 8 * np.finfo(np.float64).eps, 'stability outside numerical [0,1] range')
                stability = min(stability, 1.)
            row = dict(molecular_name=name,
                trimmed_gain=float((np.sum(delta, dtype=np.float64) - max(0., float(np.max(delta)))) / signal),
                direct_gain=direct_sum / signal,
                direct_per_signal=float(np.arcsinh(direct_sum / (own_sum + tau))),
                stability=stability, own_signal_energy=own_sum, direct_informative=bool(own_sum > 0),
                support_count=int(np.count_nonzero(support)), stability_informative=informative)
            _require(all(np.isfinite(value) for key, value in row.items() if key != 'molecular_name'),
                     'nonfinite derived row')
            rows.append(row)
    return dict(rows=rows, arrays=arrays)
