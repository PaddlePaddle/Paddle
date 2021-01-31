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
import setuptools
from setuptools.command.build_ext import build_ext

from .extension_utils import find_cuda_home, normalize_extension_kwargs, add_compile_flag
from .extension_utils import is_cuda_file, prepare_unix_cflags, add_std_without_repeat, get_build_directory

IS_WINDOWS = os.name == 'nt'
CUDA_HOME = find_cuda_home()


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
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)
        if 'output_dir' in kwargs:
            self.build_lib = kwargs.get("output_dir")

    def initialize_options(self):
        super(BuildExtension, self).initialize_options()
        # update options here
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

def load(name,
         sources,
         extra_cflags=None,
         extra_cuda_cflags=None,
         extra_ldflags=None,
         extra_include_paths=None,
         build_directory=None,
         verbose=False):

    # TODO(Aurelius84): Add JIT compile with cmake
    file_dir = os.path.dirname(os.path.abspath(__file__))
    os.system('cd {} && python3.7 setup.py build'.format(file_dir))

    return _import_module_from_library(name, build_directory)

call_once = True

def _import_module_from_library(name, build_directory):
    """
    Load .so shared library and import it as callable python module.
    """
    ext_path = os.path.join(build_directory, name+'.so')
    if not os.path.exists(ext_path):
        raise FileNotFoundError("Extension path: {} does not exist.".format(ext_path))
    
    # load custom op_info and kernels from .so shared library
    global call_once
    import paddle.fluid as fluid
    if call_once:
        fluid.load_op_library(ext_path)
        call_once = False

    # TODO(Aurelius84): need op_type
    op_name = 'relu2'

    # generate Python api in ext_path
    return _generate_python_module(op_name, build_directory)


API_TEMPLATE = """
from paddle.fluid.layer_helper import LayerHelper
# from /workspace/paddle-fork/python/paddle/fluid/tests/custom_op/cpp_extension import parse_op_info

# _, out_infos = parse_op_info('{op_name}')

from importlib import machinery
loader = machinery.SourceFileLoader('x', '/workspace/paddle-fork/python/paddle/fluid/tests/custom_op/cpp_extension.py')
m = loader.load_module()

_, out_infos = m.parse_op_info('relu2')

def {op_name}({inputs}):
    helper = LayerHelper("{op_name}", **locals())

    # prepare inputs
    ins = {ins}

    # prepare outputs
    outs = {{}}
    for out_name in out_infos:
        outs[out_name] =  helper.create_variable(dtype='float32', persistable=False)
    
    helper.append_op(type="{op_name}", inputs=ins, outputs=outs)

    return list(outs.values())[0]
"""

def _generate_python_module(op_name, build_directory):
    """
    Automatically generate python file to allow import or load into as module
    """
    api_file = os.path.join(build_directory, op_name + '.py')
    params_str, ins_str = _get_api_inputs_str(op_name)

    # generate python api file
    api_content = API_TEMPLATE.format(op_name=op_name, inputs=params_str, ins=ins_str)

    
    with open(api_file, 'w') as f:
        f.write(api_content)

    custom_api = _load_module_from_file(op_name, api_file)
    return custom_api

def _load_module_from_file(op_name, api_file_path):
    """
    Load module from python file.
    """
    if not os.path.exists(api_file_path):
        raise FileNotFoundError("File : {} does not exist.".format(api_file_path))
    
    ext_name = "extension"
    if six.PY2:
        import imp
        module = imp.load_source(ext_name, api_file_path)
    else:
        from importlib import machinery
        loader = machinery.SourceFileLoader(ext_name, api_file_path)
        module = loader.load_module()
    
    assert hasattr(module, op_name)
    return getattr(module, op_name)


def _get_api_inputs_str(op_name):
    """
    Returns string of api parameters and inputs dict.
    """
    in_names, _ = parse_op_info(op_name)
    params_str = ','.join([p.lower() for p in in_names])
    ins_str = "{%s}" % ','.join(["'{}' : {}".format(in_name, in_name.lower()) for in_name in in_names])
    return params_str, ins_str
