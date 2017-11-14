import sys
import core
__all__ = ['proto']
argv = []
if core.is_compile_gpu():
    argv = list(sys.argv) + [
        "--tryfromenv=fraction_of_gpu_memory_to_use,use_pinned_memory"
    ]
else:
    argv = list(sys.argv) + ["--tryfromenv=use_pinned_memory"]
core.init_gflags(argv)
