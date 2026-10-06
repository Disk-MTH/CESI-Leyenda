"""Let TensorFlow use the NVIDIA GPU when there is one. Call `setup()` before the first `import tensorflow`.

On Linux, `tensorflow[and-cuda]` installs the CUDA libraries as pip packages (site-packages/nvidia/*/lib) and
TensorFlow finds them through its RPATH. The 2.21.0 wheel forgets cuSOLVER in that RPATH, so TensorFlow gives up on
the GPU and silently falls back to the CPU. Loading cuSOLVER first fixes it. Without the CUDA packages (macOS,
Windows, CPU-only setup) this module does nothing and TensorFlow runs on the CPU as before.
"""
import ctypes
import os
import sysconfig
from pathlib import Path


def setup() -> None:
    # TF 2.21 has no native kernels for recent GPUs (RTX 50xx): the driver compiles them from PTX on the first run
    # and caches them. The default cache is too small to keep them all, so every run would recompile.
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024**3))
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # take GPU memory as needed, not all at start
    lib_dir = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cusolver" / "lib"
    for lib in sorted(lib_dir.glob("libcusolver.so.*")):
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
