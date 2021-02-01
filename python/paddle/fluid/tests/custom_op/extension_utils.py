# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import six
import sys
import copy
import glob
import warnings
import subprocess

import paddle

IS_WINDOWS = os.name == 'nt'
# TODO(Aurelius84): Need check version of gcc and g++ is same.
# After CI path is fixed, we will modify into cc.
NVCC_COMPILE_FLAGS = [
    '-ccbin', 'gcc', '-DPADDLE_WITH_CUDA', '-DEIGEN_USE_GPU',
    '-DPADDLE_USE_DSO', '-Xcompiler', '-fPIC', '-w', '--expt-relaxed-constexpr',
    '-O3', '-DNVCC'
]


def prepare_unix_cflags(cflags):
    """
    Prepare all necessary compiled flags for nvcc compiling CUDA files.
    """
    cflags = NVCC_COMPILE_FLAGS + cflags + get_cuda_arch_flags(cflags)

    return cflags


def add_std_without_repeat(cflags, compiler_type, use_std14=False):
    """
    Append -std=c++11/14 in cflags if without specific it before.
    """
    cpp_flag_prefix = '/std:' if compiler_type == 'msvc' else '-std='
    if not any(cpp_flag_prefix in flag for flag in cflags):
        suffix = 'c++14' if use_std14 else 'c++11'
        cpp_flag = cpp_flag_prefix + suffix
        cflags.append(cpp_flag)


def get_cuda_arch_flags(cflags):
    """
    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.
    """
    # TODO(Aurelius84):
    return []


def normalize_extension_kwargs(kwargs, use_cuda=False):
    """ 
    Normalize include_dirs, library_dir and other attributes in kwargs.
    """
    assert isinstance(kwargs, dict)
    # append necessary include dir path of paddle
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs.extend(find_paddle_includes(use_cuda))
    kwargs['include_dirs'] = include_dirs

    # append necessary lib path of paddle
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs.extend(find_paddle_libraries(use_cuda))
    kwargs['library_dirs'] = library_dirs

    # add runtime library dirs
    runtime_library_dirs = kwargs.get('runtime_library_dirs', [])
    runtime_library_dirs.extend(find_paddle_libraries(use_cuda))
    kwargs['runtime_library_dirs'] = runtime_library_dirs

    # append compile flags
    extra_compile_args = kwargs.get('extra_compile_args', [])
    extra_compile_args.extend(['-g'])
    kwargs['extra_compile_args'] = extra_compile_args

    # append link flags
    extra_link_args = kwargs.get('extra_link_args', [])
    extra_link_args.extend(['-lpaddle_framework', '-lcudart'])
    kwargs['extra_link_args'] = extra_link_args

    kwargs['language'] = 'c++'
    return kwargs


def find_paddle_includes(use_cuda=False):
    """
    Return Paddle necessary include dir path.
    """
    # pythonXX/site-packages/paddle/include
    paddle_include_dir = paddle.sysconfig.get_include()
    third_party_dir = os.path.join(paddle_include_dir, 'third_party')

    include_dirs = [paddle_include_dir, third_party_dir]

    return include_dirs


def find_cuda_includes():

    cuda_home = find_cuda_home()
    if cuda_home is None:
        raise ValueError(
            "Not found CUDA runtime, please use `export CUDA_HOME=XXX` to specific it."
        )

    return [os.path.join(cuda_home, 'lib64')]


def find_cuda_home():
    """
    Use heuristic method to find cuda path
    """
    # step 1. find in $CUDA_HOME or $CUDA_PATH
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')

    # step 2.  find path by `which nvcc`
    if cuda_home is None:
        which_cmd = 'where' if IS_WINDOWS else 'which'
        try:
            with open(os.devnull, 'w') as devnull:
                nvcc_path = subprocess.check_output(
                    [which_cmd, 'nvcc'], stderr=devnull)
                if six.PY3:
                    nvcc_path = nvcc_path.decode()
                nvcc_path = nvcc_path.rstrip('\r\n')
                # for example: /usr/local/cuda/bin/nvcc
                cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        except:
            if IS_WINDOWS:
                # search from default NVIDIA GPU path
                candidate_paths = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(candidate_paths) > 0:
                    cuda_home = candidate_paths[0]
            else:
                cuda_home = "/usr/local/cuda"
    # step 3. check whether path is valid
    if not os.path.exists(cuda_home) and paddle.is_compiled_with_cuda():
        cuda_home = None
        warnings.warn(
            "Not found CUDA runtime, please use `export CUDA_HOME= XXX` to specific it."
        )

    return cuda_home


def find_paddle_libraries(use_cuda=False):
    """
    Return Paddle necessary library dir path.
    """
    # pythonXX/site-packages/paddle/libs
    paddle_lib_dirs = [paddle.sysconfig.get_lib()]
    if use_cuda:
        cuda_dirs = find_cuda_includes()
        paddle_lib_dirs.extend(cuda_dirs)
    return paddle_lib_dirs


def append_necessary_flags(extra_compile_args, use_cuda=False):
    """
    Add necessary compile flags for gcc/nvcc compiler.
    """
    necessary_flags = ['-std=c++11']

    if use_cuda:
        necessary_flags.extend(NVCC_COMPILE_FLAGS)


def add_compile_flag(extension, flag):
    extra_compile_args = copy.deepcopy(extension.extra_compile_args)
    if isinstance(extra_compile_args, dict):
        for args in extra_compile_args.values():
            args.append(flag)
    else:
        extra_compile_args.append(flag)

    extension.extra_compile_args = extra_compile_args


def is_cuda_file(path):

    cuda_suffix = set(['.cu'])
    items = os.path.splitext(path)
    assert len(items) > 1
    return items[-1] in cuda_suffix


def get_build_directory(name):
    """
    Return paddle extension root directory, default specific by `PADDLE_EXTENSION_DIR`
    """
    root_extensions_directory = os.envsiron.get('PADDLE_EXTENSION_DIR')
    if root_extensions_directory is None:
        # TODO(Aurelius84): consider wind32/macOs
        here = os.path.abspath(__file__)
        root_extensions_directory = os.path.realpath(here)
        warnings.warn(
            "$PADDLE_EXTENSION_DIR is not set, using path: {} by default."
            .format(root_extensions_directory))

    return root_extensions_directory
