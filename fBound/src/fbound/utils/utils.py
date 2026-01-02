"""
Utilities for deterministic experiments and basic validation.

Important note on macOS thread-safety knobs:
- Environment variables affecting BLAS/OpenMP thread pools must be set
  *before* importing numpy / sklearn / torch to be effective.
- Call `apply_macos_thread_safety_knobs(enable=True)` at the very top of
  your entrypoint script (before other imports), mirroring the original
  `fBound.py` behavior.
"""
from __future__ import annotations

from typing import Callable, Tuple
import os
import sys
import math


_MACOS_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def apply_macos_thread_safety_knobs(enable: bool) -> None:
    """
    Set conservative thread limits that can reduce OpenMP/BLAS instability
    and oversubscription on some macOS setups (notably Apple Silicon).

    Notes
    -----
    - To be effective for BLAS libraries, call this before importing numpy/sklearn/torch.
    - If torch is already imported, this function will also set torch's intraop threads to 1.
    """
    if not enable:
        return

    for k in _MACOS_THREAD_ENV_KEYS:
        os.environ.setdefault(k, "1")

    # If torch is already imported, also clamp torch threads.
    if "torch" in sys.modules:
        import torch  # local import by design

        torch.set_num_threads(1)


def set_global_seed(seed: int, deterministic_torch: bool) -> "numpy.random.Generator":
    """
    Set seeds for Python, NumPy, and PyTorch in one place.

    Requirements
    ------------
    - NumPy: use default_rng for local RNG use.
    - PyTorch: torch.manual_seed(seed).
    """
    if seed is None:
        raise ValueError("seed must be an int, got None.")

    import random
    import numpy as np

    random.seed(seed)

    # Global legacy RNGs (helps libraries still relying on np.random.* globals)
    np.random.seed(seed)

    # Preferred local RNG
    rng = np.random.default_rng(seed)

    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            # Determinism knobs (best-effort; some ops may still be nondeterministic on GPU)
            torch.use_deterministic_algorithms(True)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

    return rng


def choose_batch_size(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive.")
    if n <= 1000:
        return 16
    if n <= 5000:
        return 32
    if n <= 10000:
        return 64
    return min(128, int(n**0.5))


def make_domain_penalty_schedule(
    num_epochs: int,
    *,
    rho: float = 0.3,
    w1: float = 1e6,
    w2: float = 1e4,
) -> tuple[int, Callable[[int], float]]:
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must be in [0, 1].")
    stage1_epochs = int(math.ceil(rho * num_epochs))

    def w_dom(epoch: int) -> float:
        return w1 if epoch < stage1_epochs else w2

    return stage1_epochs, w_dom


def check_shapes(
    Y: "numpy.ndarray",
    A: "numpy.ndarray",
    X: "numpy.ndarray",
) -> Tuple["numpy.ndarray", "numpy.ndarray", "numpy.ndarray"]:
    """
    Validate basic shapes and types, and return safely-converted arrays.

    Returns
    -------
    (Y, A, X) converted to:
      - Y: float32 shape (n,)
      - A: int64 shape (n,) with values in {0,1}
      - X: float32 shape (n, d)
    """
    import numpy as np

    Y_arr = np.asarray(Y)
    A_arr = np.asarray(A)
    X_arr = np.asarray(X)

    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D (n,d). Got shape {X_arr.shape}.")

    n = X_arr.shape[0]
    if Y_arr.shape[0] != n or A_arr.shape[0] != n:
        raise ValueError(
            f"Shape mismatch: len(Y)={Y_arr.shape[0]}, len(A)={A_arr.shape[0]}, "
            f"X.shape[0]={n}."
        )

    Y_arr = Y_arr.reshape(-1)
    A_arr = A_arr.reshape(-1)

    # Ensure binary A
    unique = set(np.unique(A_arr).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"A must be binary in {{0,1}}. Found unique values: {sorted(unique)}")

    # Conversions used throughout the codebase
    X_arr = X_arr.astype(np.float32, copy=False)
    Y_arr = Y_arr.astype(np.float32, copy=False)
    A_arr = A_arr.astype(np.int64, copy=False)

    return Y_arr, A_arr, X_arr


def make_kfold_splits(
    n: int,
    n_splits: int,
    seed: int,
    shuffle: bool = True,
) -> list[tuple["numpy.ndarray", "numpy.ndarray"]]:
    """
    Deterministic KFold indices helper.

    Returns
    -------
    List of (train_idx, fold_idx) for k=0..K-1.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if n <= 0:
        raise ValueError("n must be positive.")

    import numpy as np
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, fold_idx in kf.split(np.arange(n)):
        splits.append((train_idx.astype(int), fold_idx.astype(int)))
    return splits
