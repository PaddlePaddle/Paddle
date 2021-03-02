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
import json
import glob
import hashlib
import logging
import collections
import textwrap
import warnings
import subprocess

from contextlib import contextmanager
from setuptools.command import bdist_egg

from .. import load_op_library
from ...fluid import core
from ...fluid.framework import OpProtoHolder
from ...sysconfig import get_include, get_lib

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("utils.cpp_extension")

OS_NAME = sys.platform
IS_WINDOWS = OS_NAME.startswith('win')

MSVC_COMPILE_FLAGS = [
    '/MT', '/wd4819', '/wd4251', '/wd4244', '/wd4267', '/wd4275', '/wd4018',
    '/wd4190', '/EHsc', '/w', '/DGOOGLE_GLOG_DLL_DECL',
    '/DBOOST_HAS_STATIC_ASSERT', '/DNDEBUG', '/DPADDLE_USE_DSO'
]

MSVC_LINK_FLAGS = ['/MACHINE:X64', 'paddle_custom_op.lib']

COMMON_NVCC_FLAGS = ['-DPADDLE_WITH_CUDA', '-DEIGEN_USE_GPU', '-O3']

GCC_MINI_VERSION = (5, 4, 0)
MSVC_MINI_VERSION = (19, 0, 24215)
# Give warning if using wrong compiler
WRONG_COMPILER_WARNING = '''
                        *************************************
                        *  Compiler Compatibility WARNING   *
                        *************************************

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Found that your compiler ({user_compiler}) is not compatible with the compiler 
built Paddle for this platform, which is {paddle_compiler} on {platform}. Please
use {paddle_compiler} to compile your custom op. Or you may compile Paddle from
source using {user_compiler}, and then also use it compile your custom op.

See https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html
for help with compiling Paddle from source.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
# Give warning if used compiler version is incompatible
ABI_INCOMPATIBILITY_WARNING = '''
                            **********************************
                            *    ABI Compatibility WARNING   *
                            **********************************

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Found that your compiler ({user_compiler} == {version}) may be ABI-incompatible with pre-installed Paddle!
Please use compiler that is ABI-compatible with GCC >= 5.4 (Recommended 8.2).

See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html for ABI Compatibility
information

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
USING_NEW_CUSTOM_OP_LOAD_METHOD = True

DEFAULT_OP_ATTR_NAMES = [
    core.op_proto_and_checker_maker.kOpRoleAttrName(),
    core.op_proto_and_checker_maker.kOpRoleVarAttrName(),
    core.op_proto_and_checker_maker.kOpNameScopeAttrName(),
    core.op_proto_and_checker_maker.kOpCreationCallstackAttrName(),
    core.op_proto_and_checker_maker.kOpDeviceAttrName()
]


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


def load_op_meta_info_and_register_op(lib_filename):
    if USING_NEW_CUSTOM_OP_LOAD_METHOD:
        core.load_op_meta_info_and_register_op(lib_filename)
    else:
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
        import types
        import paddle
        
        def inject_ext_module(module_name, api_names):
            if module_name in sys.modules:
                return sys.modules[module_name]

            new_module = types.ModuleType(module_name)
            for api_name in api_names:
                setattr(new_module, api_name, eval(api_name))

            return new_module

        def __bootstrap__():
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            so_path = os.path.join(cur_dir, "{resource}")

            assert os.path.exists(so_path)

            # load custom op shared library with abs path
            new_custom_ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)
            m = inject_ext_module(__name__, new_custom_ops)
        
        __bootstrap__()

        {custom_api}
        """).lstrip()

    # Parse registerring op information
    _, op_info = CustomOpInfo.instance().last()
    so_path = op_info.so_path

    new_custom_ops = load_op_meta_info_and_register_op(so_path)
    assert len(
        new_custom_ops
    ) > 0, "Required at least one custom operators, but received len(custom_op) =  %d" % len(
        new_custom_ops)

    # NOTE: To avoid importing .so file instead of python file because they have same name,
    # we rename .so shared library to another name, see EasyInstallCommand.
    filename, ext = os.path.splitext(resource)
    resource = filename + "_pd_" + ext

    api_content = []
    for op_name in new_custom_ops:
        api_content.append(_custom_api_content(op_name))

    with open(pyfile, 'w') as f:
        f.write(
            _stub_template.format(
                resource=resource, custom_api='\n\n'.join(api_content)))


OpInfo = collections.namedtuple('OpInfo', ['so_name', 'so_path'])


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

    def add(self, op_name, so_name, so_path=None):
        self.op_info_map[op_name] = OpInfo(so_name, so_path)

    def last(self):
        """
        Return the lastest insert custom op info.
        """
        assert len(self.op_info_map) > 0
        return next(reversed(self.op_info_map.items()))


VersionFields = collections.namedtuple('VersionFields', [
    'sources',
    'extra_compile_args',
    'extra_link_args',
    'library_dirs',
    'runtime_library_dirs',
    'include_dirs',
    'define_macros',
    'undef_macros',
])


class VersionManager:
    def __init__(self, version_field):
        self.version_field = version_field
        self.version = self.hasher(version_field)

    def hasher(self, version_field):
        from paddle.fluid.layers.utils import flatten

        md5 = hashlib.md5()
        for field in version_field._fields:
            elem = getattr(version_field, field)
            if not elem: continue
            if isinstance(elem, (list, tuple, dict)):
                flat_elem = flatten(elem)
                md5 = combine_hash(md5, tuple(flat_elem))
            else:
                raise RuntimeError(
                    "Support types with list, tuple and dict, but received {} with {}.".
                    format(type(elem), elem))

        return md5.hexdigest()

    @property
    def details(self):
        return self.version_field._asdict()


def combine_hash(md5, value):
    """
    Return new hash value.
    DO NOT use `hash()` beacuse it doesn't generate stable value between different process.
    See https://stackoverflow.com/questions/27522626/hash-function-in-python-3-3-returns-different-results-between-sessions
    """
    md5.update(repr(value).encode())
    return md5


def clean_object_if_change_cflags(so_path, extension):
    """
    If already compiling source before, we should check whether cflags 
    have changed and delete the built object to re-compile the source
    even though source file content keeps unchanaged.
    """

    def serialize(path, version_info):
        assert isinstance(version_info, dict)
        with open(path, 'w') as f:
            f.write(json.dumps(version_info, indent=4, sort_keys=True))

    def deserialize(path):
        assert os.path.exists(path)
        with open(path, 'r') as f:
            content = f.read()
            return json.loads(content)

    # version file
    VERSION_FILE = "version.txt"
    base_dir = os.path.dirname(so_path)
    so_name = os.path.basename(so_path)
    version_file = os.path.join(base_dir, VERSION_FILE)

    # version info
    args = [getattr(extension, field, None) for field in VersionFields._fields]
    version_field = VersionFields._make(args)
    versioner = VersionManager(version_field)

    if os.path.exists(so_path) and os.path.exists(version_file):
        old_version_info = deserialize(version_file)
        so_version = old_version_info.get(so_name, None)
        # delete shared library file if versison is changed to re-compile it.
        if so_version is not None and so_version != versioner.version:
            log_v(
                "Re-Compiling {}, because specified cflags have been changed. New signature {} has been saved into {}.".
                format(so_name, versioner.version, version_file))
            os.remove(so_path)
            # upate new version information
            new_version_info = versioner.details
            new_version_info[so_name] = versioner.version
            serialize(version_file, new_version_info)
    else:
        # If compile at first time, save compiling detail information for debug.
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        details = versioner.details
        details[so_name] = versioner.version
        serialize(version_file, details)


def prepare_unix_cudaflags(cflags):
    """
    Prepare all necessary compiled flags for nvcc compiling CUDA files.
    """
    cflags = COMMON_NVCC_FLAGS + [
        '-ccbin', 'cc', '-Xcompiler', '-fPIC', '-w', '--expt-relaxed-constexpr',
        '-DNVCC'
    ] + cflags + get_cuda_arch_flags(cflags)

    return cflags


def prepare_win_cudaflags(cflags):
    """
    Prepare all necessary compiled flags for nvcc compiling CUDA files.
    """
    cflags = COMMON_NVCC_FLAGS + ['-w'] + cflags + get_cuda_arch_flags(cflags)

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

    # append compile flags and check settings of compiler
    extra_compile_args = kwargs.get('extra_compile_args', [])
    if isinstance(extra_compile_args, dict):
        for compiler in ['cxx', 'nvcc']:
            if compiler not in extra_compile_args:
                extra_compile_args[compiler] = []

    if IS_WINDOWS:
        # TODO(zhouwei): may append compile flags in future
        pass
        # append link flags
        extra_link_args = kwargs.get('extra_link_args', [])
        extra_link_args.extend(MSVC_LINK_FLAGS)
        if use_cuda:
            extra_link_args.extend(['cudadevrt.lib', 'cudart_static.lib'])
        kwargs['extra_link_args'] = extra_link_args
    else:
        # append compile flags
        add_compile_flag(extra_compile_args, ['-g', '-w'])  # disable warnings

        # append link flags
        extra_link_args = kwargs.get('extra_link_args', [])
        if use_new_custom_op_load_method():
            extra_link_args.append('-lpaddle_custom_op')
        else:
            extra_link_args.append('-lpaddle_framework')
        if use_cuda:
            extra_link_args.append('-lcudart')

        kwargs['extra_link_args'] = extra_link_args

        # add runtime library dirs
        runtime_library_dirs = kwargs.get('runtime_library_dirs', [])
        runtime_library_dirs.extend(find_paddle_libraries(use_cuda))
        kwargs['runtime_library_dirs'] = runtime_library_dirs

    kwargs['extra_compile_args'] = extra_compile_args

    kwargs['language'] = 'c++'
    return kwargs


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
                    'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*.*'
                )
                if len(candidate_paths) > 0:
                    cuda_home = candidate_paths[0]
            else:
                cuda_home = "/usr/local/cuda"
    # step 3. check whether path is valid
    if cuda_home and not os.path.exists(
            cuda_home) and core.is_compiled_with_cuda():
        cuda_home = None
        warnings.warn(
            "Not found CUDA runtime, please use `export CUDA_HOME= XXX` to specific it."
        )

    return cuda_home


def find_cuda_includes():
    """
    Use heuristic method to find cuda include path
    """
    cuda_home = find_cuda_home()
    if cuda_home is None:
        raise ValueError(
            "Not found CUDA runtime, please use `export CUDA_HOME=XXX` to specific it."
        )

    return [os.path.join(cuda_home, 'include')]


def find_paddle_includes(use_cuda=False):
    """
    Return Paddle necessary include dir path.
    """
    # pythonXX/site-packages/paddle/include
    paddle_include_dir = get_include()
    third_party_dir = os.path.join(paddle_include_dir, 'third_party')
    include_dirs = [paddle_include_dir, third_party_dir]

    if use_cuda:
        cuda_include_dir = find_cuda_includes()
        include_dirs.extend(cuda_include_dir)

    return include_dirs


def find_cuda_libraries():
    """
    Use heuristic method to find cuda static lib path
    """
    cuda_home = find_cuda_home()
    if cuda_home is None:
        raise ValueError(
            "Not found CUDA runtime, please use `export CUDA_HOME=XXX` to specific it."
        )
    if IS_WINDOWS:
        cuda_lib_dir = [os.path.join(cuda_home, 'lib', 'x64')]
    else:
        cuda_lib_dir = [os.path.join(cuda_home, 'lib64')]

    return cuda_lib_dir


def find_paddle_libraries(use_cuda=False):
    """
    Return Paddle necessary library dir path.
    """
    # pythonXX/site-packages/paddle/libs
    paddle_lib_dirs = [get_lib()]

    if use_cuda:
        cuda_lib_dir = find_cuda_libraries()
        paddle_lib_dirs.extend(cuda_lib_dir)

    return paddle_lib_dirs


def add_compile_flag(extra_compile_args, flags):
    assert isinstance(flags, list)
    if isinstance(extra_compile_args, dict):
        for args in extra_compile_args.values():
            args.extend(flags)
    else:
        extra_compile_args.extend(flags)


def is_cuda_file(path):

    cuda_suffix = set(['.cu'])
    items = os.path.splitext(path)
    assert len(items) > 1
    return items[-1] in cuda_suffix


def get_build_directory(verbose=False):
    """
    Return paddle extension root directory to put shared library. It could be specified by
    ``export PADDLE_EXTENSION_DIR=XXX`` . If not set, ``~/.cache/paddle_extension`` will be used
    by default.

    Returns:
        The root directory of compiling customized operators.

    Examples:

    .. code-block:: python

        from paddle.utils.cpp_extension import get_build_directory

        build_dir = get_build_directory()
        print(build_dir)

    """
    root_extensions_directory = os.environ.get('PADDLE_EXTENSION_DIR')
    if root_extensions_directory is None:
        dir_name = "paddle_extensions"
        root_extensions_directory = os.path.join(
            os.path.expanduser('~/.cache'), dir_name)
        if IS_WINDOWS:
            root_extensions_directory = os.path.normpath(
                root_extensions_directory)
        elif OS_NAME.startswith('darwin'):
            # TODO(Aurelius84): consider macOs
            raise NotImplementedError("Not support Mac now.")

        log_v("$PADDLE_EXTENSION_DIR is not set, using path: {} by default.".
              format(root_extensions_directory), verbose)

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
            "Please load {} shared library file firstly by `paddle.utils.cpp_extension.load_op_meta_info_and_register_op(...)`".
            format(op_name))
    op_proto = OpProtoHolder.instance().get_op_proto(op_name)

    in_names = [x.name for x in op_proto.inputs]
    out_names = [x.name for x in op_proto.outputs]
    attr_names = [
        x.name for x in op_proto.attrs if x.name not in DEFAULT_OP_ATTR_NAMES
    ]

    return in_names, out_names, attr_names


def _import_module_from_library(module_name, build_directory, verbose=False):
    """
    Load shared library and import it as callable python module.
    """
    if IS_WINDOWS:
        dynamic_suffix = '.pyd'
    else:
        dynamic_suffix = '.so'
    ext_path = os.path.join(build_directory, module_name + dynamic_suffix)
    if not os.path.exists(ext_path):
        raise FileNotFoundError("Extension path: {} does not exist.".format(
            ext_path))

    # load custom op_info and kernels from .so shared library
    log_v('loading shared library from: {}'.format(ext_path), verbose)
    op_names = load_op_meta_info_and_register_op(ext_path)

    # generate Python api in ext_path
    return _generate_python_module(module_name, op_names, build_directory,
                                   verbose)


def _generate_python_module(module_name,
                            op_names,
                            build_directory,
                            verbose=False):
    """
    Automatically generate python file to allow import or load into as module
    """
    api_file = os.path.join(build_directory, module_name + '.py')
    log_v("generate api file: {}".format(api_file), verbose)

    # write into .py file
    api_content = [_custom_api_content(op_name) for op_name in op_names]
    with open(api_file, 'w') as f:
        f.write('\n\n'.join(api_content))

    # load module
    custom_module = _load_module_from_file(api_file, verbose)
    return custom_module


def _custom_api_content(op_name):
    params_str, ins_str, attrs_str, outs_str = _get_api_inputs_str(op_name)

    API_TEMPLATE = textwrap.dedent("""
        from paddle.fluid.layer_helper import LayerHelper

        def {op_name}({inputs}):
            helper = LayerHelper("{op_name}", **locals())

            # prepare inputs and outputs
            ins = {ins}
            attrs = {attrs}
            outs = {{}}
            out_names = {out_names}
            for out_name in out_names:
                # Set 'float32' temporarily, and the actual dtype of output variable will be inferred
                # in runtime.
                outs[out_name] = helper.create_variable(dtype='float32')
            
            helper.append_op(type="{op_name}", inputs=ins, outputs=outs, attrs=attrs)

            res = [outs[out_name] for out_name in out_names]

            return res[0] if len(res)==1 else res
            """).lstrip()

    # generate python api file
    api_content = API_TEMPLATE.format(
        op_name=op_name,
        inputs=params_str,
        ins=ins_str,
        attrs=attrs_str,
        out_names=outs_str)

    return api_content


def _load_module_from_file(api_file_path, verbose=False):
    """
    Load module from python file.
    """
    if not os.path.exists(api_file_path):
        raise FileNotFoundError("File : {} does not exist.".format(
            api_file_path))

    # Unique readable module name to place custom api.
    log_v('import module from file: {}'.format(api_file_path), verbose)
    ext_name = "_paddle_cpp_extension_"
    if six.PY2:
        import imp
        module = imp.load_source(ext_name, api_file_path)
    else:
        from importlib import machinery
        loader = machinery.SourceFileLoader(ext_name, api_file_path)
        module = loader.load_module()

    return module


def _get_api_inputs_str(op_name):
    """
    Returns string of api parameters and inputs dict.
    """
    in_names, out_names, attr_names = parse_op_info(op_name)
    # e.g: x, y, z
    param_names = in_names + attr_names
    params_str = ','.join([p.lower() for p in param_names])
    # e.g: {'X': x, 'Y': y, 'Z': z}
    ins_str = "{%s}" % ','.join(
        ["'{}' : {}".format(in_name, in_name.lower()) for in_name in in_names])
    # e.g: {'num': n}
    attrs_str = "{%s}" % ",".join([
        "'{}' : {}".format(attr_name, attr_name.lower())
        for attr_name in attr_names
    ])
    # e.g: ['Out', 'Index']
    outs_str = "[%s]" % ','.join(["'{}'".format(name) for name in out_names])
    return params_str, ins_str, attrs_str, outs_str


def _write_setup_file(name,
                      sources,
                      file_path,
                      build_dir,
                      include_dirs,
                      extra_cxx_cflags,
                      extra_cuda_cflags,
                      link_args,
                      verbose=False):
    """
    Automatically generate setup.py and write it into build directory.
    """
    template = textwrap.dedent("""
    import os
    from paddle.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, setup
    from paddle.utils.cpp_extension import get_build_directory
    from paddle.utils.cpp_extension.extension_utils import use_new_custom_op_load_method

    use_new_custom_op_load_method({use_new_method})

    setup(
        name='{name}',
        ext_modules=[
            {prefix}Extension(
                sources={sources},
                include_dirs={include_dirs},
                extra_compile_args={{'cxx':{extra_cxx_cflags}, 'nvcc':{extra_cuda_cflags}}},
                extra_link_args={extra_link_args})],
        cmdclass={{"build_ext" : BuildExtension.with_options(
            output_dir=r'{build_dir}',
            no_python_abi_suffix=True)
        }})""").lstrip()

    with_cuda = False
    if any([is_cuda_file(source) for source in sources]):
        with_cuda = True
    log_v("with_cuda: {}".format(with_cuda), verbose)

    content = template.format(
        name=name,
        prefix='CUDA' if with_cuda else 'Cpp',
        sources=list2str(sources),
        include_dirs=list2str(include_dirs),
        extra_cxx_cflags=list2str(extra_cxx_cflags),
        extra_cuda_cflags=list2str(extra_cuda_cflags),
        extra_link_args=list2str(link_args),
        build_dir=build_dir,
        use_new_method=use_new_custom_op_load_method())

    log_v('write setup.py into {}'.format(file_path), verbose)
    with open(file_path, 'w') as f:
        f.write(content)


def list2str(args):
    """
    Convert list[str] into string. For example: ['x', 'y'] -> "['x', 'y']"
    """
    if args is None: return '[]'
    assert isinstance(args, (list, tuple))
    args = ["{}".format(arg) for arg in args]
    return repr(args)


def _jit_compile(file_path, interpreter=None, verbose=False):
    """
    Build shared library in subprocess
    """
    ext_dir = os.path.dirname(file_path)
    setup_file = os.path.basename(file_path)

    if interpreter is None:
        interpreter = 'python'
    try:
        which = 'where' if IS_WINDOWS else 'which'
        py_path = subprocess.check_output([which, interpreter])
        py_version = subprocess.check_output([interpreter, '-V'])
        if six.PY3:
            py_path = py_path.decode()
            py_version = py_version.decode()
        log_v("Using Python interpreter: {}, version: {}".format(
            py_path.strip(), py_version.strip()), verbose)
    except Exception:
        _, error, _ = sys.exc_info()
        raise RuntimeError(
            'Failed to check Python interpreter with `{}`, errors: {}'.format(
                interpreter, error))

    if IS_WINDOWS:
        compile_cmd = 'cd /d {} && {} {} build'.format(ext_dir, interpreter,
                                                       setup_file)
    else:
        compile_cmd = 'cd {} && {} {} build'.format(ext_dir, interpreter,
                                                    setup_file)

    print("Compiling user custom op, it will cost a few seconds.....")
    run_cmd(compile_cmd, verbose)


def parse_op_name_from(sources):
    """
    Parse registerring custom op name from sources.
    """

    def regex(content):
        if USING_NEW_CUSTOM_OP_LOAD_METHOD:
            pattern = re.compile(r'PD_BUILD_OP\(([^,\)]+)\)')
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

    return list(op_names)


def run_cmd(command, verbose=False):
    """
    Execute command with subprocess.
    """
    # logging
    log_v("execute command: {}".format(command), verbose)
    try:
        from subprocess import DEVNULL  # py3
    except ImportError:
        DEVNULL = open(os.devnull, 'wb')

    # execute command
    try:
        if verbose:
            return subprocess.check_call(
                command, shell=True, stderr=subprocess.STDOUT)
        else:
            return subprocess.check_call(command, shell=True, stdout=DEVNULL)
    except Exception:
        _, error, _ = sys.exc_info()
        raise RuntimeError("Failed to run command: {}, errors: {}".format(
            compile, error))


def check_abi_compatibility(compiler, verbose=False):
    """
    Check whether GCC version on user local machine is compatible with Paddle in
    site-packages.
    """
    if os.environ.get('PADDLE_SKIP_CHECK_ABI') in ['True', 'true', '1']:
        return True

    which = 'where' if IS_WINDOWS else 'which'
    cmd_out = subprocess.check_output(
        [which, compiler], stderr=subprocess.STDOUT)
    compiler_path = os.path.realpath(cmd_out.decode()
                                     if six.PY3 else cmd_out).strip()
    # step 1. if not found any suitable compiler, raise error
    if not any(name in compiler_path
               for name in _expected_compiler_current_platform()):
        warnings.warn(
            WRONG_COMPILER_WARNING.format(
                user_compiler=compiler,
                paddle_compiler=_expected_compiler_current_platform()[0],
                platform=OS_NAME))
        return False

    version = (0, 0, 0)
    # clang++ have no ABI compatibility problem
    if OS_NAME.startswith('darwin'):
        return True
    try:
        if OS_NAME.startswith('linux'):
            mini_required_version = GCC_MINI_VERSION
            version_info = subprocess.check_output(
                [compiler, '-dumpfullversion', '-dumpversion'])
            if six.PY3:
                version_info = version_info.decode()
            version = version_info.strip().split('.')
        elif IS_WINDOWS:
            mini_required_version = MSVC_MINI_VERSION
            compiler_info = subprocess.check_output(
                compiler, stderr=subprocess.STDOUT)
            if six.PY3:
                compiler_info = compiler_info.decode()
            match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.strip())
            if match is not None:
                version = match.groups()
    except Exception:
        # check compiler version failed
        _, error, _ = sys.exc_info()
        warnings.warn('Failed to check compiler version for {}: {}'.format(
            compiler, error))
        return False

    # check version compatibility
    assert len(version) == 3
    if tuple(map(int, version)) >= mini_required_version:
        return True
    warnings.warn(
        ABI_INCOMPATIBILITY_WARNING.format(
            user_compiler=compiler, version='.'.join(version)))
    return False


def _expected_compiler_current_platform():
    """
    Returns supported compiler string on current platform
    """
    if OS_NAME.startswith('darwin'):
        expect_compilers = ['clang', 'clang++']
    elif OS_NAME.startswith('linux'):
        expect_compilers = ['gcc', 'g++', 'gnu-c++', 'gnu-cc']
    elif IS_WINDOWS:
        expect_compilers = ['cl']
    return expect_compilers


def log_v(info, verbose=True):
    """
    Print log information on stdout.
    """
    if verbose:
        logging.info(info)
