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
import copy
import setuptools
from setuptools.command.build_ext import build_ext

import paddle

NVCC_COMPILE_FLAGS = [
    '-ccbin cc', '-DNVCC', '-DPADDLE_WITH_CUDA', '-DEIGEN_USE_GPU',
    '-DPADDLE_USE_DSO', '-Xcompiler'
]


def CppExtension(name, sources, *args, **kwargs):
    """
    Returns setuptools.CppExtension instance for setup.py to make it easy
    to specify compile flags while build C++ custommed op kernel.
    """
    kwargs = normalize_extension_kwargs(kwargs, use_cuda=False)

    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(name, sources, *args, **kwargs):
    """
    Returns setuptools.CppExtension instance for setup.py to make it easy
    to specify compile flags while build CUDA custommed op kernel.
    """
    kwargs = normalize_extension_kwargs(kwargs, use_cuda=True)

    return setuptools.Extension(name, sources, *args, **kwargs)


def normalize_extension_kwargs(kwargs, use_cuda=False):
    """ 
    Normalize include_dirs, library_dir and other attributes in kwargs.
    """
    assert isinstance(kwargs, dict)
    # append necessary include dir path of paddle
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs.extend(find_paddle_includes(True))
    kwargs['include_dirs'] = include_dirs

    # append necessary lib path of paddle
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs.extend(find_paddle_libraries(True))
    kwargs['library_dirs'] = library_dirs

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

    if use_cuda:
        cuda_dirs = find_cuda_includes()
        include_dirs.extend(cuda_dirs)

    return include_dirs


def find_cuda_includes():
    # TODO(Aurelius84): Use heuristic method to find cuda path
    include_dirs = ["/usr/local/cuda/lib64"]

    return include_dirs


def find_paddle_libraries(use_cuda=False):
    """
    Return Paddle necessary library dir path.
    """
    # pythonXX/site-packages/paddle/libs
    paddle_lib_dir = paddle.sysconfig.get_lib()
    return [paddle_lib_dir]


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


class BuildExtension(build_ext, object):
    """
    For setuptools.cmd_class.
    """

    @classmethod
    def with_options(cls, **options):
        '''
        Returns a subclass with alternative constructor that extends any original keyword
        arguments to the original constructor with the given options.
        '''

        class cls_with_options(cls):  # type: ignore
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs):
        super(BuildExtension, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

    def initialize_options(self):
        super(BuildExtension, self).initialize_options()
        # update options here.
        self.build_lib = './'

    def finalize_options(self):
        super(BuildExtension, self).finalize_options()

    def build_extensions(self):
        self._check_abi()
        for extension in self.extensions:
            # check settings of compiler
            if isinstance(extension.extra_compile_args, dict):
                for compiler in ['cxx', 'nvcc']:
                    if compiler not in extension.extra_compile_args:
                        extension.extra_compile_args[compiler] = []
            # add determine compile flags
            add_compile_flag(extension, '-std=c++11')
            # add_compile_flag(extension, '-lpaddle_framework')

        # Consider .cu, .cu.cc as valid source extensions.
        self.compiler.src_extensions += ['.cu', '.cu.cc']
        # Save the original _compile method for later.
        if self.compiler.compiler_type == 'msvc':
            raise NotImplementedError("Not support on MSVC currently.")

        build_ext.build_extensions(self)

    def get_ext_filename(self, fullname):
        # for example: custommed_extension.cpython-37m-x86_64-linux-gnu.so
        ext_name = super(BuildExtension, self).get_ext_filename(fullname)
        if self.no_python_abi_suffix:
            split_str = '.'
            name_items = ext_name.split(split_str)
            assert len(
                name_items
            ) > 2, "Expected len(name_items) > 2, but received {}".format(
                len(name_items))
            name_items.pop(-2)
            # custommed_extension.so
            ext_name = split_str.join(name_items)

        return ext_name

    def _check_abi(self):
        pass
