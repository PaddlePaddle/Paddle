# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import site
import sys
import os
import warnings
import platform

has_paddle_dy_lib = False

dy_lib_name = 'libpaddle'
dy_lib_suffix = 'so'
if os.name == 'nt':
    dy_lib_suffix = 'pyd'

current_path = os.path.abspath(os.path.dirname(__file__))
if os.path.exists(current_path + os.sep + dy_lib_name + '.' + dy_lib_suffix):
    has_paddle_dy_lib = True

try:
    if os.name == 'nt':
        third_lib_path = current_path + os.sep + '..' + os.sep + 'libs'
        # Will load shared library from 'path' on windows
        os.environ['path'] = (
            current_path + ';' + third_lib_path + ';' + os.environ['path']
        )
        sys.path.insert(0, third_lib_path)
        # Note: from python3.8, PATH will not take effect
        # https://github.com/python/cpython/pull/12302
        # Use add_dll_directory to specify dll resolution path
        if sys.version_info[:2] >= (3, 8):
            os.add_dll_directory(third_lib_path)

except ImportError as e:
    if os.name == 'nt':
        executable_path = os.path.abspath(os.path.dirname(sys.executable))
        raise ImportError(
            """NOTE: You may need to run \"set PATH=%s;%%PATH%%\"
        if you encounters \"DLL load failed\" errors. If you have python
        installed in other directory, replace \"%s\" with your own
        directory. The original error is: \n %s"""
            % (executable_path, executable_path, str(e))
        )
    else:
        raise ImportError(
            """NOTE: You may need to run \"export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH\"
        if you encounters \"libmkldnn.so not found\" errors. If you have python
        installed in other directory, replace \"/usr/local/lib\" with your own
        directory. The original error is: \n"""
            + str(e)
        )
except Exception as e:
    raise e


def avx_supported():
    """
    Whether current system(Linux, MacOS, Windows) is supported with AVX.
    """
    sysstr = platform.system().lower()
    has_avx = False
    if sysstr == 'linux':
        try:
            pipe = os.popen('cat /proc/cpuinfo | grep -i avx')
            has_avx = pipe.read() != ''
            pipe.close()
        except Exception as e:
            sys.stderr.write(
                'Can not get the AVX flag from /proc/cpuinfo.\n'
                'The original error is: %s\n' % str(e)
            )
        return has_avx
    elif sysstr == 'darwin':
        try:
            pipe = os.popen('sysctl machdep.cpu.features | grep -i avx')
            has_avx = pipe.read() != ''
            pipe.close()
        except Exception as e:
            sys.stderr.write(
                'Can not get the AVX flag from machdep.cpu.features.\n'
                'The original error is: %s\n' % str(e)
            )
        if not has_avx:
            import subprocess

            pipe = subprocess.Popen(
                'sysctl machdep.cpu.leaf7_features | grep -i avx',
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _ = pipe.communicate()
            has_avx = True if pipe.returncode == 0 else False
        return has_avx
    elif sysstr == 'windows':
        import ctypes

        ONE_PAGE = ctypes.c_size_t(0x1000)

        def asm_func(code_str, restype=ctypes.c_uint32, argtypes=()):
            # Call the code_str as a function
            # Alloc 1 page to ensure the protection
            pfnVirtualAlloc = ctypes.windll.kernel32.VirtualAlloc
            pfnVirtualAlloc.restype = ctypes.c_void_p
            MEM_COMMIT = ctypes.c_ulong(0x1000)
            PAGE_READWRITE = ctypes.c_ulong(0x4)
            address = pfnVirtualAlloc(
                None, ONE_PAGE, MEM_COMMIT, PAGE_READWRITE
            )
            if not address:
                raise Exception("Failed to VirtualAlloc")

            # Copy the code into the memory segment
            memmove = ctypes.CFUNCTYPE(
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            )(ctypes._memmove_addr)
            if memmove(address, code_str, len(code_str)) < 0:
                raise Exception("Failed to memmove")

            # Enable execute permissions
            PAGE_EXECUTE = ctypes.c_ulong(0x10)
            pfnVirtualProtect = ctypes.windll.kernel32.VirtualProtect
            res = pfnVirtualProtect(
                ctypes.c_void_p(address),
                ONE_PAGE,
                PAGE_EXECUTE,
                ctypes.byref(ctypes.c_ulong(0)),
            )
            if not res:
                raise Exception("Failed VirtualProtect")

            # Flush instruction cache
            pfnGetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
            pfnGetCurrentProcess.restype = ctypes.c_void_p
            prochandle = ctypes.c_void_p(pfnGetCurrentProcess())
            res = ctypes.windll.kernel32.FlushInstructionCache(
                prochandle, ctypes.c_void_p(address), ONE_PAGE
            )
            if not res:
                raise Exception("Failed FlushInstructionCache")

            # Cast the memory to function
            functype = ctypes.CFUNCTYPE(restype, *argtypes)
            func = functype(address)
            return func, address

        # http://en.wikipedia.org/wiki/CPUID#EAX.3D1:_Processor_Info_and_Feature_Bits
        # mov eax,0x1; cpuid; mov cx, ax; ret
        code_str = b"\xB8\x01\x00\x00\x00\x0f\xa2\x89\xC8\xC3"
        avx_bit = 28
        retval = 0
        try:
            # Convert the code_str into a function that returns uint
            func, address = asm_func(code_str)
            retval = func()
            ctypes.windll.kernel32.VirtualFree(
                ctypes.c_void_p(address), ctypes.c_size_t(0), ONE_PAGE
            )
        except Exception as e:
            sys.stderr.write(
                'Failed getting the AVX flag on Windows.\n'
                'The original error is: %s\n' % str(e)
            )
        return (retval & (1 << avx_bit)) > 0
    else:
        sys.stderr.write('Do not get AVX flag on %s\n' % sysstr)
        return False


def run_shell_command(cmd):
    import subprocess

    out, err = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    ).communicate()
    if err:
        return None
    else:
        return out.decode('utf-8').strip()


def get_dso_path(core_so, dso_name):
    if core_so and dso_name:
        return run_shell_command(
            "ldd %s|grep %s|awk '{print $3}'" % (core_so, dso_name)
        )
    else:
        return None


def load_dso(dso_absolute_path):
    if dso_absolute_path:
        try:
            from ctypes import cdll

            cdll.LoadLibrary(dso_absolute_path)
        except:
            warnings.warn("Load {} failed".format(dso_absolute_path))


def pre_load(dso_name):
    if has_paddle_dy_lib:
        core_so = current_path + os.sep + dy_lib_name + '.' + dy_lib_suffix
    else:
        core_so = None
    dso_path = get_dso_path(core_so, dso_name)
    load_dso(dso_path)


def get_libc_ver():
    ldd_glibc = run_shell_command("ldd --version | awk '/ldd/{print $NF}'")
    if ldd_glibc is not None:
        return ("glibc", ldd_glibc)

    ldd_musl = run_shell_command("ldd 2>&1 | awk '/Version/{print $NF}'")
    if ldd_musl is not None:
        return ("musl", ldd_musl)
    return (None, None)


def less_than_ver(a, b):
    if a is None or b is None:
        return False

    import re
    import operator

    def to_list(s):
        s = re.sub(r'(\.0+)+$', '', s)
        return [int(x) for x in s.split('.')]

    return operator.lt(to_list(a), to_list(b))


# NOTE(zhiqiu): An error may occurs when import paddle in linux platform with glibc < 2.22,
# the error message of which is "dlopen: cannot load any more object with static TLS".
# This happens when:
# (1) the number of dynamic shared librarys (DSO) loaded > 14,
# (2) after that, load a dynamic shared library (DSO) with static TLS.
# For paddle, the problem is that 'libgomp' is a DSO with static TLS, and it is loaded after 14 DSOs.
# So, here is a tricky way to solve the problem by pre load 'libgomp' before 'libpaddle.so'.
# The final solution is to upgrade glibc to > 2.22 on the target system.
if platform.system().lower() == 'linux':
    libc_type, libc_ver = get_libc_ver()
    if libc_type == 'glibc' and less_than_ver(libc_ver, '2.23'):
        try:
            pre_load('libgomp')
        except Exception as e:
            # NOTE(zhiqiu): do not abort if failed, since it may success when import libpaddle.so
            sys.stderr.write('Error: Can not preload libgomp.so')

try:
    from . import libpaddle

    if avx_supported() and not libpaddle.is_compiled_with_avx():
        sys.stderr.write(
            "Hint: Your machine support AVX, but the installed paddlepaddle doesn't have avx core. "
            "Hence, no-avx core with worse preformance will be imported.\nIf you like, you could "
            "reinstall paddlepaddle by 'python -m pip install --force-reinstall paddlepaddle-gpu[==version]' "
            "to get better performance.\n"
        )

    # assign tensor alias
    libpaddle.LoDTensor = libpaddle.Tensor

    from .libpaddle import *
    from .libpaddle import __doc__, __file__, __name__, __package__
    from .libpaddle import __unittest_throw_exception__
    from .libpaddle import _append_python_callable_object_and_return_id
    from .libpaddle import _cleanup, _Scope
    from .libpaddle import _get_use_default_grad_op_desc_maker_ops
    from .libpaddle import _get_all_register_op_kernels
    from .libpaddle import _is_program_version_supported
    from .libpaddle import _set_eager_deletion_mode
    from .libpaddle import _get_eager_deletion_vars
    from .libpaddle import _set_fuse_parameter_group_size
    from .libpaddle import _set_fuse_parameter_memory_size
    from .libpaddle import _is_dygraph_debug_enabled
    from .libpaddle import _dygraph_debug_level
    from .libpaddle import _switch_tracer
    from .libpaddle import _set_paddle_lib_path
    from .libpaddle import _create_loaded_parameter
    from .libpaddle import _cuda_synchronize
    from .libpaddle import _is_compiled_with_heterps
    from .libpaddle import _promote_types_if_complex_exists
    from .libpaddle import _set_cached_executor_build_strategy
    from .libpaddle import _device_synchronize
    from .libpaddle import _xpu_device_synchronize
    from .libpaddle import _get_current_stream
    from .libpaddle import _Profiler, _ProfilerResult, _RecordEvent
    from .libpaddle import _set_current_stream
    from .libpaddle import _get_phi_kernel_name

    if sys.platform != 'win32':
        from .libpaddle import _set_process_pids
        from .libpaddle import _erase_process_pids
        from .libpaddle import _set_process_signal_handler
        from .libpaddle import _throw_error_if_process_failed
        from .libpaddle import _convert_to_tensor_list
        from .libpaddle import _array_to_share_memory_tensor
        from .libpaddle import _cleanup_mmap_fds
        from .libpaddle import _remove_tensor_list_mmap_fds
except Exception as e:
    if has_paddle_dy_lib:
        sys.stderr.write(
            'Error: Can not import paddle core while this file exists: '
            + current_path
            + os.sep
            + 'libpaddle.'
            + dy_lib_suffix
            + '\n'
        )
    if not avx_supported() and libpaddle.is_compiled_with_avx():
        sys.stderr.write(
            "Error: Your machine doesn't support AVX, but the installed PaddlePaddle is avx core, "
            "you should reinstall paddlepaddle with no-avx core.\n"
        )
    raise e


def set_paddle_custom_device_lib_path(lib_path):
    if os.environ.get('CUSTOM_DEVICE_ROOT', None) is not None:
        # use setted environment value
        return
    if os.path.exists(lib_path):
        # set CUSTOM_DEVICE_ROOT default path
        os.environ['CUSTOM_DEVICE_ROOT'] = os.path.normpath(lib_path)
    else:
        os.environ['CUSTOM_DEVICE_ROOT'] = ''


# set paddle lib path
def set_paddle_lib_path():
    site_dirs = (
        site.getsitepackages()
        if hasattr(site, 'getsitepackages')
        else [x for x in sys.path if 'site-packages' in x]
    )
    for site_dir in site_dirs:
        lib_dir = os.path.sep.join([site_dir, 'paddle', 'libs'])
        if os.path.exists(lib_dir):
            _set_paddle_lib_path(lib_dir)
            set_paddle_custom_device_lib_path(
                os.path.sep.join([lib_dir, '..', '..', 'paddle-plugins'])
            )
            return
    if hasattr(site, 'USER_SITE'):
        lib_dir = os.path.sep.join([site.USER_SITE, 'paddle', 'libs'])
        if os.path.exists(lib_dir):
            _set_paddle_lib_path(lib_dir)
            set_paddle_custom_device_lib_path(
                os.path.sep.join([lib_dir, '..', '..', 'paddle-plugins'])
            )


set_paddle_lib_path()
