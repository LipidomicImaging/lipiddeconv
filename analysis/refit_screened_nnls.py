"""Source-bound NNLS refit on selected library columns, without truth or rho."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

if __package__:
    from . import run_nnls_solver_baseline as baseline
    from . import run_small_mismatch_nnls_first_case as engine
else:
    import run_nnls_solver_baseline as baseline
    import run_small_mismatch_nnls_first_case as engine

_worker_limiter = None


def _initialize_worker(A):
    # Load SciPy's BLAS before limiting already-loaded thread pools.
    from scipy import optimize as _optimize  # noqa: F401
    global _worker_limiter
    _worker_limiter = threadpool_limits(limits=1)
    baseline.initialize_worker(A)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def _array_binding(array):
    return dict(shape=list(array.shape), dtype=array.dtype.str,
                sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, value, replace=False):
    data = _canonical(value) + b"\n"
    if path.exists():
        if not replace:
            _require(path.read_bytes() == data, "IMMUTABLE_RECORD_CHANGED:" + path.name)
            return
        _require(_read(path).get("fingerprint") == value.get("fingerprint"), "STATUS_BINDING_CHANGED")
    temporary = path.with_name(path.name + ".tmp")
    _require(not temporary.exists(), "PARTIAL_WRITE_PRESERVED:" + temporary.name)
    with temporary.open("xb") as stream:
        stream.write(data)
    temporary.replace(path)


def _write_npz(path, **arrays):
    _require(not path.exists(), "EXISTING_ARRAY_PRESERVED:" + path.name)
    temporary = path.with_name(path.name + ".tmp")
    _require(not temporary.exists(), "PARTIAL_WRITE_PRESERVED:" + temporary.name)
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _indices(indices, n):
    result = np.asarray(indices)
    if result.size == 0:
        _require(result.ndim == 1, "INVALID_RETAINED_INDICES")
        return np.empty(0, dtype=np.int64)
    _require(result.ndim == 1 and result.dtype.kind in "iu", "INVALID_RETAINED_INDICES")
    _require(np.all(result >= 0) and np.all(result < n), "RETAINED_INDEX_OUT_OF_RANGE")
    result = result.astype(np.int64)
    _require(len(set(result.tolist())) == result.size, "DUPLICATE_RETAINED_INDEX")
    return result


def report_identities(means, names, selected_names, gate=.001):
    """Return reported selected names after the unchanged per-candidate gate."""
    means = np.asarray(means)
    names = list(names)
    selected = set(selected_names)
    _require(means.ndim == 1 and means.size == len(names), "MEAN_NAME_SHAPE_MISMATCH")
    _require(np.isfinite(means).all() and (means >= 0).all(), "INVALID_CANDIDATE_MEANS")
    _require(np.isfinite(gate) and gate >= 0, "INVALID_REPORT_GATE")
    _require(all(isinstance(name, str) for name in names), "INVALID_IDENTITY_NAME")
    _require(selected.issubset(set(names)), "UNKNOWN_SELECTED_IDENTITY")
    groups = {}
    for j, name in enumerate(names):
        if name in selected:
            groups.setdefault(name, []).append(j)
    records = []
    for name, indices in groups.items():
        active = [j for j in indices if means[j] > gate]
        if active:
            records.append(dict(lipid_name=name, raw_solver_reported=True,
                                X_hat=sum(float(means[j]) for j in active),
                                candidate_indices=indices, reported_candidate_indices=active))
    return records


def fit_screened(A, B, mask, retained_indices, output_dir, workers=4,
                 block_size=250, source_binding=None):
    """Refit selected columns and return means, array/receipt paths and binding.

    All original candidate positions are preserved in float32 X_hat. A completed
    block is reused only with the exact input, source, implementation and array
    hashes. Incomplete or conflicting artifacts are preserved and rejected.
    """
    A, B, mask = np.asarray(A), np.asarray(B), np.asarray(mask)
    _require(A.ndim == 2 and A.shape[0] > 0 and A.shape[1] > 0, "INVALID_LIBRARY_SHAPE")
    _require(B.ndim == 3 and mask.ndim == 2 and B.shape == (A.shape[0], *mask.shape),
             "OBSERVATION_MASK_SHAPE_MISMATCH")
    _require(A.dtype.kind in "fi" and B.dtype.kind in "fi" and
             np.isfinite(A).all() and np.isfinite(B).all(), "NONFINITE_OR_NONREAL_INPUT")
    _require(mask.dtype == np.bool_ and mask.any(), "INVALID_OR_EMPTY_FOREGROUND_MASK")
    _require(np.all(B[:, ~mask] == 0), "NONZERO_BACKGROUND_REQUIRES_SOLVE")
    _require(isinstance(workers, int) and not isinstance(workers, bool) and 1 <= workers <= 4,
             "INVALID_WORKER_COUNT")
    _require(isinstance(block_size, int) and not isinstance(block_size, bool) and block_size > 0,
             "INVALID_BLOCK_SIZE")
    indices = _indices(retained_indices, A.shape[1])
    scientific = dict(version=1, A=_array_binding(A), B=_array_binding(B), mask=_array_binding(mask),
                      retained_indices=indices.tolist(), source_binding=source_binding,
                      solver="existing scipy NNLS; float64 solve, float32 storage", maxiter=3910,
                      workers=workers, block_size=block_size, blas_threads=1,
                      implementation={Path(p).name: _sha(p) for p in
                                      (__file__, baseline.__file__, engine.__file__)},
                      numpy=np.__version__, scipy=scipy.__version__)
    # Canonicalization also rejects non-JSON/nonfinite caller provenance.
    scientific = json.loads(_canonical(scientific))
    fingerprint = hashlib.sha256(_canonical(scientific)).hexdigest()
    binding = dict(fingerprint=fingerprint, scientific=scientific)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    binding_path = output / "binding.json"
    if not binding_path.exists():
        _require(not any(output.iterdir()), "UNBOUND_EXISTING_OUTPUT_PRESERVED")
    _write_json(binding_path, binding)
    status_path = output / "status.json"
    if status_path.exists():
        _require(_read(status_path).get("fingerprint") == fingerprint, "STATUS_BINDING_CHANGED")
    _require(not list(output.glob("*.tmp")), "PARTIAL_WRITE_PRESERVED")
    directory = output / "nnls_blocks"
    directory.mkdir(exist_ok=True)
    count = int(mask.sum())
    spans = [(start, min(start + block_size, count)) for start in range(0, count, block_size)]
    expected = {f"block_{start:06d}_{stop:06d}.{ext}" for start, stop in spans for ext in ("npz", "json")}
    _require({p.name for p in directory.iterdir()}.issubset(expected), "UNEXPECTED_BLOCK_MEMBERSHIP")
    arrays_path, receipt_path = output / "learned_arrays.npz", output / "completion.json"
    receipt = _read(receipt_path) if receipt_path.exists() else None
    if receipt is not None:
        _require(receipt.get("fingerprint") == fingerprint and receipt.get("status") == "COMPLETE",
                 "COMPLETION_BINDING_CHANGED")
        _require(arrays_path.is_file() and _sha(arrays_path) == receipt["arrays_sha256"],
                 "COMPLETION_ARRAY_HASH_CHANGED")
        _require(_sha(binding_path) == receipt["binding_sha256"], "COMPLETION_SOURCE_HASH_CHANGED")
        _require({p.name for p in directory.iterdir()} == expected, "INCOMPLETE_COMPLETED_BLOCK_MEMBERSHIP")
    reduced = np.ascontiguousarray(A[:, indices], dtype=np.float64)
    spectra = B[:, mask].astype(np.float64)
    foreground_x = np.zeros((len(indices), count), dtype=np.float32)
    overall = dict(max_dual_violation=0., max_complementarity=0., max_bound_ratio=0.)
    block_hashes = {}
    cached_count = 0
    executor = None
    # SciPy may not yet be loaded in a caller that only inspected cached data.
    from scipy import optimize as _optimize  # noqa: F401
    try:
        with threadpool_limits(limits=1):
            for start, stop in spans:
                stem = f"block_{start:06d}_{stop:06d}"
                array_file, record_file = directory / (stem + ".npz"), directory / (stem + ".json")
                if array_file.exists() or record_file.exists():
                    _require(array_file.is_file() and record_file.is_file(), "PARTIAL_NNLS_BLOCK_PRESERVED")
                    record = _read(record_file)
                    _require(record.get("fingerprint") == fingerprint and record.get("start") == start and
                             record.get("stop") == stop and record.get("retained_indices") == indices.tolist(),
                             "NNLS_BLOCK_BINDING_CHANGED")
                    _require(_sha(array_file) == record["array_sha256"], "NNLS_BLOCK_HASH_CHANGED")
                    with np.load(array_file, allow_pickle=False) as archive:
                        _require(set(archive.files) == {"X_hat"}, "INVALID_BLOCK_ARRAY_MEMBERSHIP")
                        block = archive["X_hat"]
                    cached_count += 1
                else:
                    _require(receipt is None, "MISSING_COMPLETED_BLOCK")
                    checks = dict(overall, **{key: 0. for key in overall})
                    block = np.zeros((len(indices), stop - start), dtype=np.float32)
                    if len(indices):
                        if workers == 1:
                            baseline.initialize_worker(reduced)
                            solutions = map(baseline.solve_pixel, (spectra[:, j] for j in range(start, stop)))
                        else:
                            if executor is None:
                                executor = ProcessPoolExecutor(max_workers=workers,
                                    mp_context=multiprocessing.get_context("spawn"),
                                    initializer=_initialize_worker, initargs=(reduced,))
                            solutions = executor.map(baseline.solve_pixel,
                                (spectra[:, j] for j in range(start, stop)), chunksize=16)
                        for offset, solution in enumerate(solutions):
                            check = engine.kkt_check(reduced, spectra[:, start + offset], solution, np)
                            for key, value in check.items():
                                checks[key] = max(checks[key], value)
                            block[:, offset] = solution
                    _require(np.isfinite(block).all(), "FLOAT32_STORAGE_OVERFLOW")
                    _write_npz(array_file, X_hat=block)
                    record = dict(fingerprint=fingerprint, start=start, stop=stop,
                                  retained_indices=indices.tolist(), array_sha256=_sha(array_file),
                                  KKT=checks, all_pixels_checked_before_float32=True,
                                  empty_design_vacuous_kkt=(len(indices) == 0))
                    _write_json(record_file, record)
                checks = record["KKT"]
                _require(block.shape == (len(indices), stop - start) and block.dtype == np.float32 and
                         np.isfinite(block).all() and (block >= 0).all(), "INVALID_NNLS_CHECKPOINT")
                _require(record.get("all_pixels_checked_before_float32") is True and
                         all(np.isfinite(v) and v >= 0 for v in checks.values()) and
                         set(checks) == set(overall) and checks["max_bound_ratio"] <= 1., "INVALID_KKT_RECEIPT")
                foreground_x[:, start:stop] = block
                for key, value in checks.items():
                    overall[key] = max(overall[key], value)
                block_hashes[array_file.name] = _sha(array_file)
                block_hashes[record_file.name] = _sha(record_file)
                if receipt is None:
                    _write_json(status_path, dict(fingerprint=fingerprint, status="RUNNING",
                                pixels_done=stop, total_pixels=count), replace=True)
            X_hat = np.zeros((A.shape[1], *mask.shape), dtype=np.float32)
            flat = np.zeros((A.shape[1], count), dtype=np.float32)
            flat[indices] = foreground_x
            X_hat[:, mask] = flat
            if receipt is not None:
                _require(block_hashes == receipt["block_hashes"], "COMPLETION_BLOCK_HASH_CHANGED")
                with np.load(arrays_path, allow_pickle=False) as archive:
                    _require(set(archive.files) == {"X_hat", "B_hat"}, "INVALID_FINAL_ARRAY_MEMBERSHIP")
                    _require(archive["X_hat"].dtype == np.float32 and
                             np.array_equal(archive["X_hat"], X_hat), "FINAL_BLOCK_MEMBERSHIP_CHANGED")
                    B_hat = archive["B_hat"]
                    _require(B_hat.shape == B.shape and B_hat.dtype == np.float32 and
                             np.isfinite(B_hat).all() and (B_hat[:, ~mask] == 0).all(), "INVALID_FINAL_RECONSTRUCTION")
            else:
                B_hat = np.zeros(B.shape, dtype=np.float32)
                B_hat[:, mask] = (reduced @ foreground_x.astype(np.float64)).astype(np.float32)
                _require(np.isfinite(B_hat).all(), "FLOAT32_RECONSTRUCTION_OVERFLOW")
                if arrays_path.exists():
                    with np.load(arrays_path, allow_pickle=False) as archive:
                        _require(set(archive.files) == {"X_hat", "B_hat"} and
                                 archive["X_hat"].dtype == archive["B_hat"].dtype == np.float32 and
                                 np.array_equal(archive["X_hat"], X_hat) and
                                 np.array_equal(archive["B_hat"], B_hat), "UNSEALED_FINAL_ARRAY_CHANGED")
                else:
                    _write_npz(arrays_path, X_hat=X_hat, B_hat=B_hat)
                receipt = dict(status="COMPLETE", fingerprint=fingerprint, binding_sha256=_sha(binding_path),
                               arrays_sha256=_sha(arrays_path), block_hashes=block_hashes,
                               retained_indices=indices.tolist(), foreground_pixels=count, KKT=overall,
                               all_pixels_checked_before_float32=True, all_outputs_finite=True,
                               candidate_count=A.shape[1], fitted_column_count=len(indices))
                _write_json(receipt_path, receipt)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    _write_json(status_path, dict(fingerprint=fingerprint, status="COMPLETE", pixels_done=count,
                                 total_pixels=count, completion_sha256=_sha(receipt_path)), replace=True)
    return dict(means=X_hat[:, mask].mean(axis=1, dtype=np.float64), arrays_path=arrays_path,
                receipt_path=receipt_path, fingerprint=fingerprint, resumed=(cached_count > 0))
