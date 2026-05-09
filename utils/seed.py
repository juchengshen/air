# utils/seed.py
from __future__ import annotations

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None


def _get_rank_world() -> Tuple[int, int]:
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def seed_everything(
    seed: int,
    *,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    rank_offset: bool = True,
    deterministic: bool = False,
    cudnn_benchmark: bool = False,
    allow_tf32: Optional[bool] = None,
    warn_only_determinism: bool = True,
) -> int:
    """
    Seed as many RNG sources as is reasonable.

    Returns the integer seed actually used on this process.

    Notes on determinism:
    - deterministic=True enables PyTorch deterministic algorithms and CuDNN deterministic mode.
    - Some ops will error (or warn) if they have no deterministic implementation.
    - For full CuBLAS determinism, CUBLAS_WORKSPACE_CONFIG should be set before CUDA context creation.
    """

    base_seed = int(seed)

    # Infer rank/world_size if not provided.
    auto_rank, auto_world = _get_rank_world()
    if rank is None:
        rank = auto_rank
    if world_size is None:
        world_size = auto_world

    # Standard practice: each rank gets a distinct seed derived from the base seed.
    # This makes data shuffling, dropout, augmentations, etc. independent across ranks.
    used_seed = base_seed + (rank if rank_offset else 0)

    # Environment-level seeds
    os.environ["PYTHONHASHSEED"] = str(used_seed)

    # CuBLAS determinism knob (must be set early for best effect).
    # Only set it if user requested deterministic mode and it is not already set.
    if deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        # Two common choices are ":4096:8" and ":16:8". The first is the safer default.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Python, NumPy
    random.seed(used_seed)
    np.random.seed(used_seed)

    # Torch CPU and CUDA
    torch.manual_seed(used_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(used_seed)
        torch.cuda.manual_seed_all(used_seed)

    # Determinism and backend knobs
    if allow_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)

    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # benchmark can introduce nondeterminism

        # Force deterministic SDPA kernel (math) to avoid non-deterministic cuDNN/flash backends.
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

        # PyTorch deterministic algorithms flag.
        # warn_only=True avoids hard crashes but still tells you when determinism is violated.
        try:
            torch.use_deterministic_algorithms(True, warn_only=bool(warn_only_determinism))
        except TypeError:
            # Older torch versions do not have warn_only.
            torch.use_deterministic_algorithms(True)

    return used_seed


def seed_worker(worker_id: int) -> None:
    """
    For DataLoader(worker_init_fn=seed_worker), ensures each worker has a unique, reproducible seed.
    Uses torch.initial_seed(), which is derived from the DataLoader generator.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader_generator(seed: int, *, rank: Optional[int] = None, rank_offset: bool = True) -> torch.Generator:
    """
    Convenience: a torch.Generator to pass into DataLoader(generator=...).
    Combined with seed_worker, this gives reproducible multi-worker loading.
    """
    if rank is None:
        rank, _ = _get_rank_world()
    used_seed = int(seed) + (rank if rank_offset else 0)
    g = torch.Generator()
    g.manual_seed(used_seed)
    return g
