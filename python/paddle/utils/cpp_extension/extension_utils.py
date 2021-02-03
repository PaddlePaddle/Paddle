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
import re
import six
import sys
import copy
import glob
import collections
import textwrap
import platform
import warnings
import subprocess

from contextlib import contextmanager
from setuptools.command import bdist_egg

from .. import load_op_library
from ...fluid import core
from ...fluid.framework import OpProtoHolder
from ...sysconfig import get_include, get_lib

OS_NAME = platform.system()
IS_WINDOWS = OS_NAME == 'Windows'
NVCC_COMPILE_FLAGS = [
    '-ccbin', 'cc', '-DPADDLE_WITH_CUDA', '-DEIGEN_USE_GPU', '-DPADDLE_USE_DSO',
    '-Xcompiler', '-fPIC', '-w', '--expt-relaxed-constexpr', '-O3', '-DNVCC'
]

USING_NEW_CUSTOM_OP_LOAD_METHOD = True


# NOTE(chenweihang): In order to be compatible with 
# the two custom op define method, after removing 
# old method, we can remove them together
def use_new_custom_op_load_method(*args):
    global USING_NEW_CUSTOM_OP_LOAD_METHOD
    if len(args) == 0:
        return USING_NEW_CUSTOM_OP_LOAD_METHOD
    else:
        assert len(args) == 1 and isinstance(args[0], bool)
        USING_NEW_CUSTOM_OP_LOAD_METHOD = args[0]


@contextmanager
def bootstrap_context():
    """
    Context to manage how to write `__bootstrap__` code in .egg
    """
    origin_write_stub = bdist_egg.write_stub
    bdist_egg.write_stub = custom_write_stub
    yield

    bdist_egg.write_stub = origin_write_stub


def load_and_register_custom_op(lib_filename):
    if USING_NEW_CUSTOM_OP_LOAD_METHOD:
        core.load_and_register_custom_op(lib_filename)
    else:
        print("old branch")
        core.load_op_library(lib_filename)
    return OpProtoHolder.instance().update_op_proto()


def custom_write_stub(resource, pyfile):
    """
    Customized write_stub function to allow us to inject generated python
    api codes into egg python file.
    """
    _stub_template = textwrap.dedent("""
        import os
        import sys
        import paddle
        
        def inject_ext_module(module_name, api_name):
            if module_name in sys.modules:
                return sys.modules[module_name]

            new_module = imp.new_module(module_name)
            setattr(new_module, api_name, eval(api_name))

            return new_module

        def __bootstrap__():
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            so_path = os.path.join(cur_dir, "{resource}")

            assert os.path.exists(so_path)

            # load custom op shared library with abs path
            new_custom_op = paddle.utils.cpp_extension.load_and_register_custom_op(so_path)
            assert len(new_custom_op) == 1
            m = inject_ext_module(__name__, new_custom_op[0])
        
        __bootstrap__()

        {custom_api}
        """).lstrip()

    # Parse registerring op information
    _, op_info = CustomOpInfo.instance().last()
    so_path = op_info.build_directory

    new_custom_op = load_and_register_custom_op(so_path)
    assert len(new_custom_op
               ) == 1, "The number of loaded costom operators is %d" % len(
                   new_custom_op)

    # NOTE: To avoid importing .so file instead of python file because they have same name,
    # we rename .so shared library to another name, see EasyInstallCommand.
    filename, ext = os.path.splitext(resource)
    resource = filename + "_pd_" + ext

    with open(pyfile, 'w') as f:
        f.write(
            _stub_template.format(
                resource=resource,
                custom_api=_custom_api_content(new_custom_op[0])))


OpInfo = collections.namedtuple('OpInfo',
                                ['so_name', 'build_directory', 'out_dtypes'])


class CustomOpInfo:
    """
    A global Singleton map to record all compiled custom ops information.
    """

    @classmethod
    def instance(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        assert not hasattr(
            self.__class__,
            '_instance'), 'Please use `instance()` to get CustomOpInfo object!'
        # NOTE(Aurelius84): Use OrderedDict to save more order information
        self.op_info_map = collections.OrderedDict()

    def add(self, op_name, so_name, build_directory=None, out_dtypes=None):
        self.op_info_map[op_name] = OpInfo(so_name, build_directory, out_dtypes)

    def last(self):
        """
        Return the lastest insert custom op info.
        """
        assert len(self.op_info_map) > 0
        return next(reversed(self.op_info_map.items()))


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
    paddle_include_dir = get_include()
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
    if not os.path.exists(cuda_home) and core.is_compiled_with_cuda():
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
    paddle_lib_dirs = [get_lib()]
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


def get_build_directory():
    """
    Return paddle extension root directory, default specific by `PADDLE_EXTENSION_DIR`
    """
    root_extensions_directory = os.environ.get('PADDLE_EXTENSION_DIR')
    if root_extensions_directory is None:
        dir_name = "paddle_extensions"
        if OS_NAME == 'Linux':
            root_extensions_directory = os.path.join(
                os.path.expanduser('~/.cache'), dir_name)
        else:
            # TODO(Aurelius84): consider wind32/macOs
            raise NotImplementedError("Only support Linux now.")

        warnings.warn(
            "$PADDLE_EXTENSION_DIR is not set, using path: {} by default.".
            format(root_extensions_directory))

    if not os.path.exists(root_extensions_directory):
        os.makedirs(root_extensions_directory)

    return root_extensions_directory


def parse_op_info(op_name):
    """
    Parse input names and outpus detail information from registered custom op
    from OpInfoMap.
    """
    from paddle.fluid.framework import OpProtoHolder
    if op_name not in OpProtoHolder.instance().op_proto_map:
        raise ValueError(
            "Please load {} shared library file firstly by `paddle.utils.cpp_extension.load_and_register_custom_op(...)`".
            format(op_name))
    op_proto = OpProtoHolder.instance().get_op_proto(op_name)

    in_names = [x.name for x in op_proto.inputs]
    assert len(op_proto.outputs) == 1
    out_name = op_proto.outputs[0].name

    # TODO(Aurelius84): parse necessary out_dtype  of custom op
    out_infos = {out_name: ['float32']}
    return in_names, out_infos


def _import_module_from_library(name, build_directory):
    """
    Load .so shared library and import it as callable python module.
    """
    ext_path = os.path.join(build_directory, name + '.so')
    if not os.path.exists(ext_path):
        raise FileNotFoundError("Extension path: {} does not exist.".format(
            ext_path))

    # load custom op_info and kernels from .so shared library
    op_names = load_and_register_custom_op(ext_path)
    assert len(op_names) == 1

    # generate Python api in ext_path
    return _generate_python_module(op_names[0], build_directory)


def _generate_python_module(op_name, build_directory):
    """
    Automatically generate python file to allow import or load into as module
    """
    api_file = os.path.join(build_directory, op_name + '.py')

    # write into .py file
    api_content = _custom_api_content(op_name)
    with open(api_file, 'w') as f:
        f.write(api_content)

    # load module
    custom_api = _load_module_from_file(op_name, api_file)
    return custom_api


def _custom_api_content(op_name):
    params_str, ins_str = _get_api_inputs_str(op_name)

    API_TEMPLATE = textwrap.dedent("""
        from paddle.fluid.layer_helper import LayerHelper
        from paddle.utils.cpp_extension import parse_op_info

        _, _out_infos = parse_op_info('{op_name}')

        def {op_name}({inputs}):
            helper = LayerHelper("{op_name}", **locals())

            # prepare inputs and output 
            ins = {ins}
            outs = {{}}
            for out_name in _out_infos:
                outs[out_name] = [helper.create_variable(dtype=dtype) for dtype in _out_infos[out_name]]
            
            helper.append_op(type="{op_name}", inputs=ins, outputs=outs)

            res = list(outs.values())[0]
            if len(res) == 1:
                return res[0]
            else:
                return res
            """).lstrip()

    # generate python api file
    api_content = API_TEMPLATE.format(
        op_name=op_name, inputs=params_str, ins=ins_str)

    return api_content


def _load_module_from_file(op_name, api_file_path):
    """
    Load module from python file.
    """
    if not os.path.exists(api_file_path):
        raise FileNotFoundError("File : {} does not exist.".format(
            api_file_path))

    # Unique readable module name to place custom api.
    ext_name = "_paddle_cpp_extension_"
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
    # e.g: x, y, z
    params_str = ','.join([p.lower() for p in in_names])
    # e.g: {'X': x, 'Y': y, 'Z': z}
    ins_str = "{%s}" % ','.join(
        ["'{}' : {}".format(in_name, in_name.lower()) for in_name in in_names])
    return params_str, ins_str


def _write_setup_file(name, sources, file_path, include_dirs, compile_flags,
                      link_args):
    """
    Automatically generate setup.py and write it into build directory.
    """
    template = textwrap.dedent("""
    import os
    from paddle.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, setup
    from paddle.utils.cpp_extension import get_build_directory
    setup(
        name='{name}',
        ext_modules=[
            {prefix}Extension(
                name='{name}',
                sources={sources},
                include_dirs={include_dirs},
                extra_compile_args={extra_compile_args},
                extra_link_args={extra_link_args})],
        cmdclass={{"build_ext" : BuildExtension.with_options(
            output_dir=get_build_directory(),
            no_python_abi_suffix=True,
            use_new_method={use_new_method})
        }})""").lstrip()

    with_cuda = False
    if any([is_cuda_file(source) for source in sources]):
        with_cuda = True

    content = template.format(
        name=name,
        prefix='CUDA' if with_cuda else 'Cpp',
        sources=list2str(sources),
        include_dirs=list2str(include_dirs),
        extra_compile_args=list2str(compile_flags),
        extra_link_args=list2str(link_args),
        use_new_method=use_new_custom_op_load_method())
    with open(file_path, 'w') as f:
        f.write(content)


def list2str(args):
    """
    Convert list[str] into string. For example: [x, y] -> "['x', 'y']"
    """
    if args is None: return '[]'
    assert isinstance(args, (list, tuple))
    args = ["'{}'".format(arg) for arg in args]
    return '[' + ','.join(args) + ']'


def _jit_compile(file_path):
    """
    Build shared library in subprocess
    """
    ext_dir = os.path.dirname(file_path)
    setup_file = os.path.basename(file_path)
    compile_cmd = 'cd {} && python {} build'.format(ext_dir, setup_file)
    run_cmd(compile_cmd)


def parse_op_name_from(sources):
    """
    Parse registerring custom op name from sources.
    """

    def regex(content):
        if USING_NEW_CUSTOM_OP_LOAD_METHOD:
            pattern = re.compile(r'BUILD_OPERATOR\(([^,]+),')
        else:
            pattern = re.compile(r'REGISTER_OPERATOR\(([^,]+),')

        content = re.sub(r'\s|\t|\n', '', content)
        op_name = pattern.findall(content)
        op_name = set([re.sub('_grad', '', name) for name in op_name])

        return op_name

    op_names = set()
    for source in sources:
        with open(source, 'r') as f:
            content = f.read()
            op_names |= regex(content)

    # TODO(Aurelius84): Support register more customs op at once
    assert len(
        op_names) == 1, "The number of registered costom operators is %d" % len(
            op_names)
    return list(op_names)[0]


def run_cmd(command, wait=True):
    """
    Execute command with subprocess.
    """
    return subprocess.check_call(command, shell=True)
