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

import copy
import setuptools
from setuptools.command.build_ext import build_ext

import paddle


def CppExtension(name, sources, *args, **kwargs):
    """
    Returns setuptools.CppExtension instance for setup.py to make it easy
    to specify compile flags while build C++ custommed op kernel.
    """
    # append necessary include dir path of paddle
    include_dirs = kwargs.get('include_dirs', [])
    # TODO(Aurelius84): consider CUDA include path
    include_dirs += paddle.sysconfig.get_include()
    kwargs['include_dirs'] = include_dirs

    # append necessary lib path of paddle
    library_dirs = kwargs.get('library_dirs', [])
    # TODO(Aurelius84): consider CUDA lib path
    library_dirs += paddle.sysconfig.get_lib()
    kwargs['library_dirs'] = library_dirs

    kwargs['language'] = 'c++'
    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(name, sources, *args, **kwargs):
    """
    Returns setuptools.CppExtension instance for setup.py to make it easy
    to specify compile flags while build CUDA custommed op kernel.
    """
    # append necessary include dir path of paddle
    include_dirs = kwargs.get('include_dirs', [])
    # TODO(Aurelius84): consider CUDA include path
    include_dirs += paddle.sysconfig.get_include()
    kwargs['include_dirs'] = include_dirs

    # append necessary lib path of paddle
    library_dirs = kwargs.get('library_dirs', [])
    # TODO(Aurelius84): consider CUDA lib path
    library_dirs += paddle.sysconfig.get_lib()
    kwargs['library_dirs'] = library_dirs

    kwargs['language'] = 'c++'

    return setuptools.Extension(name, sources, *args, **kwargs)


class BuildExtension(build_ext, object):
    """
    For setuptools.cmd_class.
    """

    def __init__(self, *args, **kwargs):
        super(BuildExtension, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

    def finalize_options(self, options):
        super(BuildExtension, self).finalize_options()
        # update options here.

    def build_extensions(self):
        self._check_abi()
        for extension in self.extensions:
            # check settings of compiler
            if isinstance(extension.extra_compile_tag, dict):
                for compiler in ['cxx', 'nvcc']:
                    if compiler not in extension.extra_compile_tag:
                        extension.extra_compile_tag[compiler] = []
            # TODO(Aurelius84): determine compile flag
            self._add_compile_flag(extension, '')

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
            ext_name.pop(-2)
            # custommed_extension.so
            ext_name = split_str.join(ext_name)

        return ext_name

    def _check_abi(self):
        pass

    def _add_compile_flag(extension, flag):
        extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extra_compile_args, dict):
            for args in extra_compile_args.values():
                args.append(flag)
        else:
            extra_compile_args.append(flag)
        # reset into the new extra_compile_args
        extension.extra_compile_tag = extra_compile_args
