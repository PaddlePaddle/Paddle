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
import textwrap
import copy
import setuptools
from setuptools.command.easy_install import easy_install
from setuptools.command.build_ext import build_ext

from .extension_utils import find_cuda_home, normalize_extension_kwargs, add_compile_flag, bootstrap_context
from .extension_utils import is_cuda_file, prepare_unix_cflags, add_std_without_repeat, get_build_directory
from .extension_utils import _import_module_from_library, CustomOpInfo, _write_setup_file, _jit_compile

IS_WINDOWS = os.name == 'nt'
CUDA_HOME = find_cuda_home()


def setup(**attr):
    """
    Wrapper setuptools.setup function to valid `build_ext` command and
    implement generated paddle api code injection by switching `write_stub`
    function in bdist_egg with `custom_write_stub`.
    """
    cmdclass = attr.get('cmdclass', {})
    assert isinstance(cmdclass, dict)
    # if not specific cmdclass in setup, add it automaticaly.
    if 'build_ext' not in cmdclass:
        cmdclass['build_ext'] = BuildExtension.with_options(
            no_python_abi_suffix=True)
        attr['cmdclass'] = cmdclass
    elif not isinstance(cmdclass['build_ext'], BuildExtension):
        raise ValueError(
            "Require paddle.utils.cpp_extension.BuildExtension in setup(cmdclass={'build_ext: ...'}), but received {}".
            format(type(cmdclass['build_ext'])))

    # Add rename .so hook in easy_install
    assert 'easy_install' not in cmdclass
    cmdclass['easy_install'] = EasyInstallCommand

    # switch `write_stub` to inject paddle api in .egg
    with bootstrap_context():
        setuptools.setup(**attr)


def CppExtension(name, sources, *args, **kwargs):
    """
    Returns setuptools.CppExtension instance for setup.py to make it easy
    to specify compile flags while build C++ custommed op kernel.

    Args:
           name(str): The extension name
           sources(list[str]): The C++/CUDA source file names
           args(list[options]): list of config options used to compile shared library
           kwargs(dict[option]): dict of config options used to compile shared library
           
       Returns:
           Extension: instance of setuptools.Extension
    """
    kwargs = normalize_extension_kwargs(kwargs, use_cuda=False)

    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(name, sources, *args, **kwargs):
    """
    Returns setuptools.CppExtension instance for setup.py to make it easy
    to specify compile flags while build CUDA custommed op kernel.

    Args:
           name(str): The extension name
           sources(list[str]): The C++/CUDA source file names
           args(list[options]): list of config options used to compile shared library
           kwargs(dict[option]): dict of config options used to compile shared library
           
       Returns:
           Extension: instance of setuptools.Extension
    """
    kwargs = normalize_extension_kwargs(kwargs, use_cuda=True)

    return setuptools.Extension(name, sources, *args, **kwargs)


class BuildExtension(build_ext, object):
    """
    Inherited from setuptools.command.build_ext to customized how to apply
    compilation process with shared library.
    """

    @classmethod
    def with_options(cls, **options):
        '''
        Returns a BuildExtension subclass that support to specific use-defined options.
        '''

        class cls_with_options(cls):
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                cls.__init__(self, *args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs):
        super(BuildExtension, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", True)
        if 'output_dir' in kwargs:
            self.build_lib = kwargs.get("output_dir")

    def initialize_options(self):
        super(BuildExtension, self).initialize_options()
        # update options here
        # self.build_lib = './'

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
        if self.compiler.compiler_type == 'msvc' or IS_WINDOWS:
            raise NotImplementedError("Not support on MSVC currently.")
        else:
            original_compile = self.compiler._compile

        def unix_custom_single_compiler(obj, src, ext, cc_args, extra_postargs,
                                        pp_opts):
            """
            Monkey patch machanism to replace inner compiler to custom complie process on Unix platform.
            """
            # use abspath to ensure no warning
            src = os.path.abspath(src)
            cflags = copy.deepcopy(extra_postargs)

            try:
                original_compiler = self.compiler.compiler_so
                # ncvv compile CUDA source
                if is_cuda_file(src):
                    assert CUDA_HOME is not None
                    nvcc_cmd = os.path.join(CUDA_HOME, 'bin', 'nvcc')
                    self.compiler.set_executable('compiler_so', nvcc_cmd)
                    # {'nvcc': {}, 'cxx: {}}
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    else:
                        cflags = prepare_unix_cflags(cflags)
                # cxx compile Cpp source
                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']

                add_std_without_repeat(
                    cflags, self.compiler.compiler_type, use_std14=False)
                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # restore original_compiler
                self.compiler.compiler_so = original_compiler

        def object_filenames_with_cuda(origina_func):
            """
            Decorated the function to add customized naming machanism.
            """

            def wrapper(source_filenames, strip_dir=0, output_dir=''):
                try:
                    objects = origina_func(source_filenames, strip_dir,
                                           output_dir)
                    for i, source in enumerate(source_filenames):
                        # modify xx.o -> xx.cu.o
                        if is_cuda_file(source):
                            old_obj = objects[i]
                            objects[i] = old_obj[:-1] + 'cu.o'
                    # ensure to use abspath
                    objects = [os.path.abspath(obj) for obj in objects]
                finally:
                    self.compiler.object_filenames = origina_func

                return objects

            return wrapper

        # customized compile process
        self.compiler._compile = unix_custom_single_compiler
        self.compiler.object_filenames = object_filenames_with_cuda(
            self.compiler.object_filenames)

        print(self.get_outputs())
        build_ext.build_extensions(self)

    def get_ext_filename(self, fullname):
        # for example: custommed_extension.cpython-37m-x86_64-linux-gnu.so
        ext_name = super(BuildExtension, self).get_ext_filename(fullname)
        if self.no_python_abi_suffix and six.PY3:
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


class EasyInstallCommand(easy_install, object):
    """
    Extend easy_intall Command to control the behavior of naming shared library
    file.
    """

    def __init__(self, *args, **kwargs):
        super(EasyInstallCommand, self).__init__(*args, **kwargs)

    def run(self):
        super(EasyInstallCommand, self).run()
        # NOTE: To avoid failing import .so file instead of
        # python file because they have same name, we rename
        # .so shared library to another name.
        for egg_file in self.outputs:
            filename, ext = os.path.splitext(egg_file)
            if ext == '.so':
                new_so_path = filename + "_pd_" + ext
                if not os.path.exists(new_so_path):
                    os.rename(r'%s' % egg_file, r'%s' % new_so_path)
                assert os.path.exists(new_so_path)


def load(name,
         sources,
         extra_cflags=None,
         extra_cuda_cflags=None,
         extra_ldflags=None,
         extra_include_paths=None,
         build_directory=None,
         verbose=False):

    # TODO(Aurelius84): It just contains main logic codes, more details
    # will be added later.
    if build_directory is None:
        build_directory = get_build_directory()
    # ensure to use abs path
    build_directory = os.path.abspath(build_directory)
    file_path = os.path.join(build_directory, "setup.py")

    # TODO(Aurelius84): split cflags and cuda_flags
    if extra_cflags is None: extra_cflags = []
    if extra_cuda_cflags is None: extra_cuda_cflags = []
    compile_flags = extra_cflags + extra_cuda_cflags

    _write_setup_file(name, sources, file_path, extra_include_paths,
                      compile_flags, link_args)
    _jit_compile(file_path)

    custom_op_api = _import_module_from_library(name, build_directory)

    return custom_op_api
