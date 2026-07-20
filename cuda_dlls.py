"""Windows CUDA DLL discovery for the CTranslate2 / faster-whisper backend.

Import this module BEFORE ``faster_whisper`` or ``ctranslate2``: it extends the DLL
search path while those extension modules are still unloaded. Import it afterwards
and ctranslate2 has already failed to locate cuBLAS/cuDNN.

On this project's hardware (Quadro P5000, sm_61) torch is *not* an inference engine.
The pinned torch build targets sm_75+ and cannot launch kernels on Pascal at all --
it is depended on solely for the cuBLAS/cuDNN binaries it ships in ``torch/lib``,
which ctranslate2 loads. Never replace this lookup with hardcoded machine paths.
"""

import importlib.util
import os
import sys

# Module-level so the search path survives for the process lifetime.
# Adjudicated in PR #2: CPython's _AddedDllDirectory has no finalizer, so the path
# would persist even unreferenced. The handle is kept as hardening, not necessity.
_dll_dir = None


def preload_cuda_dlls():
    """Add torch's bundled CUDA runtime dir to the DLL search path.

    Returns the directory that was added, or None on non-Windows platforms and when
    torch's lib directory cannot be located.
    """
    global _dll_dir
    if not hasattr(os, "add_dll_directory"):  # non-Windows: the loader handles this
        return None
    if _dll_dir is not None:  # already registered
        return None

    spec = importlib.util.find_spec("torch")
    lib_dir = os.path.join(os.path.dirname(spec.origin), "lib") if spec and spec.origin else None
    if lib_dir and os.path.isdir(lib_dir):
        _dll_dir = os.add_dll_directory(lib_dir)
        return lib_dir

    print(
        "Warning: torch lib dir not found; ctranslate2 may fail to locate cuBLAS/cuDNN DLLs",
        file=sys.stderr,
    )
    return None


# Registered on import so that `import cuda_dlls` above the faster_whisper import is
# sufficient. Callers that import faster_whisper inside a function should still call
# preload_cuda_dlls() explicitly -- it is idempotent.
preload_cuda_dlls()
