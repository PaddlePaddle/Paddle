import os
import core
__all__ = ['proto']
argv = ['paddle']
if os.getenv('FLAGS_fraction_of_gpu_memory_to_use'):
    argv.append('--fraction_of_gpu_memory_to_use=' + os.getenv(
        'FLAGS_fraction_of_gpu_memory_to_use'))

if os.getenv('FLAGS_use_pinned_memory'):
    argv.append('--use_pinned_memory=' + os.getenv('FLAGS_use_pinned_memory'))

core.init_gflags(argv)
