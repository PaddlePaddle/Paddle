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

import errno
import fnmatch
import glob
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from distutils.spawn import find_executable
from subprocess import CalledProcessError

from setuptools import Command, Distribution, Extension, setup
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install as InstallCommandBase
from setuptools.command.install_lib import install_lib
from setuptools.dist import Distribution

if sys.version_info < (3, 7):
    raise RuntimeError(
        "Paddle only support Python version>=3.7 now, You are using Python %s"
        % platform.python_version()
    )
else:
    if os.getenv("PY_VERSION") is None:
        print("export PY_VERSION = %s" % platform.python_version())
        python_version = platform.python_version()
        os.environ["PY_VERSION"] = python_version

# check cmake
CMAKE = find_executable('cmake3') or find_executable('cmake')
assert (
    CMAKE
), 'The "cmake" executable is not found. Please check if Cmake is installed.'

# get Top_dir
TOP_DIR = os.path.dirname(os.path.realpath(__file__))

IS_WINDOWS = os.name == 'nt'


# filter args of setup.py


def filter_setup_args(input_args):
    cmake_and_make = True
    only_cmake = False
    rerun_cmake = False
    filter_args_list = []
    for arg in input_args:
        if arg == 'rerun-cmake':
            rerun_cmake = True  # delete Cmakecache.txt and rerun cmake
            continue
        if arg == 'only-cmake':
            only_cmake = True  # only cmake and do not make, leave a chance for users to adjust build options
            continue
        if arg in ['clean', 'egg_info', 'sdist']:
            cmake_and_make = False
        filter_args_list.append(arg)

    return cmake_and_make, only_cmake, rerun_cmake, filter_args_list


cmake_and_make, only_cmake, rerun_cmake, filter_args_list = filter_setup_args(
    sys.argv
)


def parse_input_command(input_parameters):
    dist = Distribution()
    # get script name :setup.py
    sys.argv = input_parameters
    dist.script_name = os.path.basename(sys.argv[0])
    print("Start executing %s" % dist.script_name)
    # get args of setup.py
    dist.script_args = sys.argv[1:]
    print("args of setup.py:%s" % dist.script_args)
    try:
        dist.parse_command_line()
    except:
        print(
            "An error occurred while parsing the parameters, '%s'"
            % dist.script_args
        )
        sys.exit(1)


class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True


RC = 0

ext_suffix = (
    '.dll'
    if os.name == 'nt'
    else ('.dylib' if sys.platform == 'darwin' else '.so')
)


class InstallHeaders(Command):
    """Override how headers are copied."""

    description = 'install C/C++ header files'

    user_options = [
        ('install-dir=', 'd', 'directory to install header files to'),
        ('force', 'f', 'force installation (overwrite existing files)'),
    ]

    boolean_options = ['force']

    def initialize_options(self):
        self.install_dir = None
        self.force = 0
        self.outfiles = []

    def finalize_options(self):
        self.set_undefined_options(
            'install', ('install_headers', 'install_dir'), ('force', 'force')
        )

    def mkdir_and_copy_file(self, header):
        if 'pb.h' in header:
            install_dir = re.sub(
                envir_var.get("PADDLE_BINARY_DIR") + '/', '', header
            )
        elif 'third_party' not in header:
            # paddle headers
            install_dir = re.sub(
                envir_var.get("PADDLE_SOURCE_DIR") + '/', '', header
            )
            print('install_dir: ', install_dir)
            if 'fluid/jit' in install_dir:
                install_dir = re.sub('fluid/jit', 'jit', install_dir)
                print('fluid/jit install_dir: ', install_dir)
            if 'trace_event.h' in install_dir:
                install_dir = re.sub(
                    'fluid/platform/profiler',
                    'phi/backends/custom',
                    install_dir,
                )
                print('trace_event.h install_dir: ', install_dir)
        else:
            # third_party
            install_dir = re.sub(
                envir_var.get("THIRD_PARTY_PATH") + '/', 'third_party', header
            )
            patterns = ['install/mkldnn/include']
            for pattern in patterns:
                install_dir = re.sub(pattern, '', install_dir)
        install_dir = os.path.join(
            self.install_dir, os.path.dirname(install_dir)
        )
        raise RuntimeError(install_dir)
        if not os.path.exists(install_dir):
            self.mkpath(install_dir)

        return self.copy_file(header, install_dir)

    def run(self):
        hdrs = self.distribution.headers
        if not hdrs:
            return
        self.mkpath(self.install_dir)
        for header in hdrs:
            print("Header:", header)
            (out, _) = self.mkdir_and_copy_file(header)
            self.outfiles.append(out)

    def get_inputs(self):
        return self.distribution.headers or []

    def get_outputs(self):
        return self.outfiles


class InstallCommand(InstallCommandBase):
    def finalize_options(self):

        ret = InstallCommandBase.finalize_options(self)
        self.install_lib = self.install_platlib
        print("install_lib:", self.install_platlib)

        self.install_headers = os.path.join(
            self.install_platlib, 'paddle', 'include'
        )
        print("install_headers:", self.install_headers)
        return ret


class EggInfo(egg_info):
    """Copy license file into `.dist-info` folder."""

    def run(self):
        # don't duplicate license into `.dist-info` when building a distribution
        if not self.distribution.have_run.get('install', True):
            self.mkpath(self.egg_info)
            self.copy_file(
                envir_var.get("PADDLE_SOURCE_DIR") + "/LICENSE", self.egg_info
            )

        egg_info.run(self)


# class Installlib is rewrote to add header files to .egg/paddle
class InstallLib(install_lib):
    def copy_file_to_egg_paddle(self, header):
        if 'pb.h' in header:
            install_dir = re.sub(
                envir_var.get("PADDLE_BINARY_DIR") + '/', '', header
            )
        elif 'third_party' not in header:
            # paddle headers
            install_dir = re.sub(
                envir_var.get("PADDLE_SOURCE_DIR") + '/', '', header
            )
            print('install_dir: ', install_dir)
            if 'fluid/jit' in install_dir:
                install_dir = re.sub('fluid/jit', 'jit', install_dir)
                print('fluid/jit install_dir: ', install_dir)
            if 'trace_event.h' in install_dir:
                install_dir = re.sub(
                    'fluid/platform/profiler',
                    'phi/backends/custom',
                    install_dir,
                )
        else:
            # third_party
            install_dir = re.sub(
                envir_var.get("THIRD_PARTY_PATH") + '/', 'third_party', header
            )
            patterns = ['install/mkldnn/include']
            for pattern in patterns:
                install_dir = re.sub(pattern, '', install_dir)
        install_dir = os.path.join(
            self.install_dir, 'paddle/include', os.path.dirname(install_dir)
        )
        if not os.path.exists(install_dir):
            os.makedirs(install_dir)
        return self.copy_file(header, install_dir)

    def run(self):
        self.build()
        outfiles = self.install()
        hrds = self.distribution.headers
        if not hrds:
            return
        for header in hrds:
            self.copy_file_to_egg_paddle(header)
        if outfiles is not None:
            # always compile, in case we have any extension stubs to deal with
            self.byte_compile(outfiles)
            # raise("___")


def git_commit():
    try:
        cmd = ['git', 'rev-parse', 'HEAD']
        git_commit = (
            subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                cwd=envir_var.get("PADDLE_SOURCE_DIR"),
            )
            .communicate()[0]
            .strip()
        )
    except:
        git_commit = 'Unknown'
    git_commit = git_commit.decode('utf-8')
    return str(git_commit)


def _get_version_detail(idx):
    assert (
        idx < 3
    ), "vesion info consists of %(major)d.%(minor)d.%(patch)d, \
        so detail index must less than 3"
    tag_version_regex = envir_var.get("TAG_VERSION_REGEX")
    paddle_version = envir_var.get("PADDLE_VERSION")
    if re.match(tag_version_regex, paddle_version):
        version_details = paddle_version.split('.')
        if len(version_details) >= 3:
            return version_details[idx]
    return 0


def _mkdir_p(dir_str):
    try:
        os.makedirs(dir_str)
    except OSError as e:
        raise RuntimeError("Failed to create folder build/")


def get_major():
    return int(_get_version_detail(0))


def get_minor():
    return int(_get_version_detail(1))


def get_patch():
    return str(_get_version_detail(2))


def get_cuda_version():
    with_gpu = envir_var.get("WITH_GPU")
    if with_gpu == 'ON':
        return envir_var.get("CUDA_VERSION")
    else:
        return 'False'


def get_cudnn_version():
    with_gpu = envir_var.get("WITH_GPU")
    if with_gpu == 'ON':
        temp_cudnn_version = ''
        cudnn_major_version = envir_var.get("CUDNN_MAJOR_VERSION")
        if cudnn_major_version:
            temp_cudnn_version += cudnn_major_version
            cudnn_minor_version = envir_var.get("CUDNN_MINOR_VERSION")
            if cudnn_minor_version:
                temp_cudnn_version = (
                    temp_cudnn_version + '.' + cudnn_minor_version
                )
                cudnn_patchlevel_version = envir_var.get(
                    "CUDNN_PATCHLEVEL_VERSION"
                )
                if cudnn_patchlevel_version:
                    temp_cudnn_version = (
                        temp_cudnn_version + '.' + cudnn_patchlevel_version
                    )
        return temp_cudnn_version
    else:
        return 'False'


def is_taged():
    try:
        cmd = [
            'git',
            'describe',
            '--exact-match',
            '--tags',
            'HEAD',
            '2>/dev/null',
        ]
        git_tag = (
            subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                cwd=envir_var.get("PADDLE_SOURCE_DIR"),
            )
            .communicate()[0]
            .strip()
        )
        git_tag = git_tag.decode()
    except:
        return False
    if str(git_tag).replace('v', '') == envir_var.get("PADDLE_VERSION"):
        return True
    else:
        return False


def write_version_py(filename='paddle/version/__init__.py'):
    cnt = '''# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY
#
full_version    = '%(major)d.%(minor)d.%(patch)s'
major           = '%(major)d'
minor           = '%(minor)d'
patch           = '%(patch)s'
rc              = '%(rc)d'
cuda_version    = '%(cuda)s'
cudnn_version   = '%(cudnn)s'
istaged         = %(istaged)s
commit          = '%(commit)s'
with_mkl        = '%(with_mkl)s'

__all__ = ['cuda', 'cudnn', 'show']

def show():
    """Get the version of paddle if `paddle` package if tagged. Otherwise, output the corresponding commit id.

    Returns:
        If paddle package is not tagged, the commit-id of paddle will be output.
        Otherwise, the following information will be output.

        full_version: version of paddle

        major: the major version of paddle

        minor: the minor version of paddle

        patch: the patch level version of paddle

        rc: whether it's rc version

        cuda: the cuda version of package. It will return `False` if CPU version paddle package is installed

        cudnn: the cudnn version of package. It will return `False` if CPU version paddle package is installed

    Examples:
        .. code-block:: python

            import paddle

            # Case 1: paddle is tagged with 2.2.0
            paddle.version.show()
            # full_version: 2.2.0
            # major: 2
            # minor: 2
            # patch: 0
            # rc: 0
            # cuda: '10.2'
            # cudnn: '7.6.5'

            # Case 2: paddle is not tagged
            paddle.version.show()
            # commit: cfa357e984bfd2ffa16820e354020529df434f7d
            # cuda: '10.2'
            # cudnn: '7.6.5'
    """
    if istaged:
        print('full_version:', full_version)
        print('major:', major)
        print('minor:', minor)
        print('patch:', patch)
        print('rc:', rc)
    else:
        print('commit:', commit)
    print('cuda:', cuda_version)
    print('cudnn:', cudnn_version)

def mkl():
    return with_mkl

def cuda():
    """Get cuda version of paddle package.

    Returns:
        string: Return the version information of cuda. If paddle package is CPU version, it will return False.

    Examples:
        .. code-block:: python

            import paddle

            paddle.version.cuda()
            # '10.2'

    """
    return cuda_version

def cudnn():
    """Get cudnn version of paddle package.

    Returns:
        string: Return the version information of cudnn. If paddle package is CPU version, it will return False.

    Examples:
        .. code-block:: python

            import paddle

            paddle.version.cudnn()
            # '7.6.5'

    """
    return cudnn_version
'''
    commit = git_commit()

    dirname = os.path.dirname(filename)

    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    with open(filename, 'w') as f:
        f.write(
            cnt
            % {
                'major': get_major(),
                'minor': get_minor(),
                'patch': get_patch(),
                'rc': RC,
                'version': envir_var.get("PADDLE_VERSION"),
                'cuda': get_cuda_version(),
                'cudnn': get_cudnn_version(),
                'commit': commit,
                'istaged': is_taged(),
                'with_mkl': envir_var.get("WITH_MKL"),
            }
        )


def write_cuda_env_config_py(filename='paddle/cuda_env.py'):
    cnt = ""
    if envir_var.get("JIT_RELEASE_WHL") == 'ON':
        cnt = '''# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY
#
import os
os.environ['CUDA_CACHE_MAXSIZE'] = '805306368'
'''

    with open(filename, 'w') as f:
        f.write(cnt)


def write_distributed_training_mode_py(
    filename='paddle/fluid/incubate/fleet/parameter_server/version.py',
):
    cnt = '''

# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY

from paddle.fluid.incubate.fleet.base.mode import Mode

BUILD_MODE=Mode.%(mode)s

def is_transpiler():
    return Mode.TRANSPILER == BUILD_MODE

'''

    dirname = os.path.dirname(filename)

    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(filename, 'w') as f:
        f.write(
            cnt
            % {
                'mode': 'PSLIB'
                if envir_var.get("WITH_PSLIB") == 'ON'
                else 'TRANSPILER'
            }
        )


def find_files(pattern, root, recursive=False):
    for dirpath, _, files in os.walk(root):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(dirpath, filename)
        if not recursive:
            break


@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError('Can only cd to absolute path, got: {}'.format(path))
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


def options_process(args, build_options):
    for key, value in sorted(build_options.items()):
        if value is not None:
            args.append("-D{}={}".format(key, value))


def cmake_run(args, build_path):
    with cd(build_path):
        cmake_args = []
        cmake_args.append(CMAKE)
        # cmake_args.append("-GNinja")
        cmake_args += args
        cmake_args.append(TOP_DIR)
        print("cmake_args:", cmake_args)
        subprocess.check_call(cmake_args)


def make_run(args, build_path, envrion_var):
    with cd(build_path):
        cmake_args = []
        cmake_args.append(CMAKE)

        cmake_args += args
        # cmake_args.append(TOP_DIR)
        print(" ".join(cmake_args))
        try:
            # print("cmake_args: ", cmake_args)
            subprocess.check_call(cmake_args, cwd=build_path, env=envrion_var)
        except (CalledProcessError, KeyboardInterrupt) as e:
            sys.exit(1)


def build_steps():
    print('------- Building start ------')

    if not os.path.exists(TOP_DIR + '/build'):
        _mkdir_p(TOP_DIR + '/build')
    build_path = TOP_DIR + '/build'

    # run cmake to generate native build files
    cmake_cache_file_path = os.path.join(build_path, "CMakeCache.txt")

    # if rerun_cmake is True,remove CMakeCache.txt and rerun camke
    if os.path.isfile(cmake_cache_file_path) and rerun_cmake is True:
        os.remove(cmake_cache_file_path)

    if not os.path.exists(cmake_cache_file_path):

        env_var = os.environ.copy()  # get env variables
        paddle_build_options = {}
        other_options = {}
        other_options.update(
            {
                option: option
                for option in (
                    "PYTHON_LIBRARY",
                    "INFERENCE_DEMO_INSTALL_DIR",
                    "ON_INFER",
                    "PYTHON_EXECUTABLE",
                    "TENSORRT_ROOT",
                    "CUDA_ARCH_NAME",
                    "CUDA_ARCH_BIN",
                    "PYTHON_INCLUDE_DIR",
                    "PYTHON_LIBRARIES",
                    "PY_VERSION",
                    "CUB_PATH",
                    "NEW_RELEASE_PYPI",
                    "CUDNN_ROOT",
                    "THIRD_PARTY_PATH",
                    "NOAVX_CORE_FILE",
                    "LITE_GIT_TAG",
                    "CUDA_TOOLKIT_ROOT_DIR",
                    "NEW_RELEASE_JIT",
                    "XPU_SDK_ROOT",
                    "MSVC_STATIC_CRT",
                    "Ninja",
                    "NEW_RELEASE_ALL",
                )
            }
        )

        # if environment variables which start with "WITH_" or "CMAKE_",put it into build_options
        for option_key, option_value in env_var.items():
            if option_key.startswith(("CMAKE_", "WITH_")):
                paddle_build_options[option_key] = option_value
            if option_key in other_options:
                print("type:", type(other_options[option_key]))
                if (
                    option_key == 'PYTHON_EXECUTABLE'
                    or option_key == 'PYTHON_LIBRARY'
                    or option_key == 'PYTHON_LIBRARIES'
                ):
                    key = option_key + ":FILEPATH"
                    print(key)
                elif option_key == 'PYTHON_INCLUDE_DIR':
                    key = key = option_key + ':PATH'
                    print(key)
                else:
                    key = other_options[option_key]
                if key not in paddle_build_options:
                    paddle_build_options[key] = option_value
        args = []
        options_process(args, paddle_build_options)
        print("args:", args)
        cmake_run(args, build_path)
    # make
    if only_cmake:
        print(
            "You have finished running cmake, the program exited,run 'ccmake build' to adjust build options and 'python setup.py install to build'"
        )
        sys.exit()

    build_args = ["--build", ".", "--target", "install", "--config", 'Release']

    max_jobs = os.getenv("MAX_JOBS")
    if max_jobs is not None:
        max_jobs = max_jobs or str(multiprocessing.cpu_count())

        build_args += ["--"]
        if IS_WINDOWS:
            build_args += ["/p:CL_MPCount={}".format(max_jobs)]
        else:
            build_args += ["-j", str(multiprocessing.cpu_count())]
    else:
        build_args += ["-j", str(multiprocessing.cpu_count())]
    environ_var = os.environ.copy()
    make_run(build_args, build_path, environ_var)


def get_setup_requires():
    with open(
        envir_var.get("PADDLE_SOURCE_DIR") + '/python/requirements.txt'
    ) as f:
        setup_requires = (
            f.read().splitlines()
        )  # Specify the dependencies to install
    if sys.version_info >= (3, 7):
        setup_requires_tmp = []
        for setup_requires_i in setup_requires:
            if (
                "<\"3.6\"" in setup_requires_i
                or "<=\"3.6\"" in setup_requires_i
                or "<\"3.5\"" in setup_requires_i
                or "<=\"3.5\"" in setup_requires_i
                or "<\"3.7\"" in setup_requires_i
            ):
                continue
            setup_requires_tmp += [setup_requires_i]
        setup_requires = setup_requires_tmp
        return setup_requires
    else:
        raise RuntimeError(
            "please check your python version,Paddle only support Python version>=3.7 now"
        )


def get_package_data_and_package_dir():

    if os.name != 'nt':
        package_data = {
            'paddle.fluid': [envir_var.get("FLUID_CORE_NAME") + '.so']
        }
    else:
        package_data = {
            'paddle.fluid': [
                envir_var.get("FLUID_CORE_NAME") + '.pyd',
                envir_var.get("FLUID_CORE_NAME") + '.lib',
            ]
        }

    package_data['paddle.fluid'] += [
        paddle_binary_dir + '/python/paddle/cost_model/static_op_benchmark.json'
    ]
    if 'develop' in sys.argv:
        package_dir = {
            '': paddle_binary_dir.split('/')[-1] + '/python',
            # '':'build/python',
            # The paddle.fluid.proto will be generated while compiling.
            # So that package points to other directory.
            'paddle.fluid.proto.profiler': paddle_binary_dir.split('/')[-1]
            + '/paddle/fluid/platform',
            'paddle.fluid.proto': paddle_binary_dir.split('/')[-1]
            + '/paddle/fluid/framework',
            'paddle.fluid': paddle_binary_dir.split('/')[-1]
            + '/python/paddle/fluid',
        }
    else:
        package_dir = {
            '': envir_var.get("PADDLE_BINARY_DIR") + '/python',
            'paddle.fluid.proto.profiler': envir_var.get("PADDLE_BINARY_DIR")
            + '/paddle/fluid/platform',
            'paddle.fluid.proto': envir_var.get("PADDLE_BINARY_DIR")
            + '/paddle/fluid/framework',
            'paddle.fluid': envir_var.get("PADDLE_BINARY_DIR")
            + '/python/paddle/fluid',
        }
    # put all thirdparty libraries in paddle.libs
    libs_path = paddle_binary_dir + '/python/paddle/libs'

    package_data['paddle.libs'] = []
    package_data['paddle.libs'] = [
        ('libwarpctc' if os.name != 'nt' else 'warpctc') + ext_suffix
    ]
    shutil.copy(envir_var.get("WARPCTC_LIBRARIES"), libs_path)

    package_data['paddle.libs'] += [
        os.path.basename(envir_var.get("LAPACK_LIB")),
        os.path.basename(envir_var.get("BLAS_LIB")),
        os.path.basename(envir_var.get("GFORTRAN_LIB")),
        os.path.basename(envir_var.get("GNU_RT_LIB_1")),
    ]
    shutil.copy(envir_var.get("BLAS_LIB"), libs_path)
    shutil.copy(envir_var.get("LAPACK_LIB"), libs_path)
    shutil.copy(envir_var.get("GFORTRAN_LIB"), libs_path)
    shutil.copy(envir_var.get("GNU_RT_LIB_1"), libs_path)

    if envir_var.get("WITH_CUDNN_DSO") == 'ON' and os.path.exists(
        envir_var.get("CUDNN_LIBRARY")
    ):
        package_data['paddle.libs'] += [
            os.path.basename(envir_var.get("CUDNN_LIBRARY"))
        ]
        shutil.copy(envir_var.get("CUDNN_LIBRARY"), libs_path)
        if (
            sys.platform.startswith("linux")
            and envir_var.get("CUDNN_MAJOR_VERSION") == '8'
        ):
            # libcudnn.so includes libcudnn_ops_infer.so, libcudnn_ops_train.so,
            # libcudnn_cnn_infer.so, libcudnn_cnn_train.so, libcudnn_adv_infer.so,
            # libcudnn_adv_train.so
            cudnn_lib_files = glob.glob(
                os.path.dirname(envir_var.get("CUDNN_LIBRARY"))
                + '/libcudnn_*so.8'
            )
            for cudnn_lib in cudnn_lib_files:
                if os.path.exists(cudnn_lib):
                    package_data['paddle.libs'] += [os.path.basename(cudnn_lib)]
                    shutil.copy(cudnn_lib, libs_path)

    if not sys.platform.startswith("linux"):
        package_data['paddle.libs'] += [
            os.path.basename(envir_var.get("GNU_RT_LIB_2"))
        ]
        shutil.copy(envir_var.get("GNU_RT_LIB_2"), libs_path)

    if envir_var.get("WITH_MKL") == 'ON':
        shutil.copy(envir_var.get("MKLML_SHARED_LIB"), libs_path)
        shutil.copy(envir_var.get("MKLML_SHARED_IOMP_LIB"), libs_path)
        package_data['paddle.libs'] += [
            ('libmklml_intel' if os.name != 'nt' else 'mklml') + ext_suffix,
            ('libiomp5' if os.name != 'nt' else 'libiomp5md') + ext_suffix,
        ]
    else:
        if os.name == 'nt':
            # copy the openblas.dll
            shutil.copy(envir_var.get("OPENBLAS_SHARED_LIB"), libs_path)
            package_data['paddle.libs'] += ['openblas' + ext_suffix]
        elif (
            os.name == 'posix'
            and platform.machine() == 'aarch64'
            and envir_var.get("OPENBLAS_LIB").endswith('so')
        ):
            # copy the libopenblas.so on linux+aarch64
            # special: libpaddle.so without avx depends on 'libopenblas.so.0', not 'libopenblas.so'
            if os.path.exists(envir_var.get("OPENBLAS_LIB") + '.0'):
                shutil.copy(envir_var.get("OPENBLAS_LIB") + '.0', libs_path)
                package_data['paddle.libs'] += ['libopenblas.so.0']

    if envir_var.get("WITH_LITE") == 'ON':
        shutil.copy(envir_var.get("LITE_SHARED_LIB"), libs_path)
        package_data['paddle.libs'] += [
            'libpaddle_full_api_shared' + ext_suffix
        ]
        if envir_var.get("LITE_WITH_NNADAPTER") == 'ON':
            shutil.copy(envir_var.get("LITE_NNADAPTER_LIB"), libs_path)
            package_data['paddle.libs'] += ['libnnadapter' + ext_suffix]
            if envir_var.get("NNADAPTER_WITH_HUAWEI_ASCEND_NPU") == 'ON':
                shutil.copy(envir_var.get("LITE_NNADAPTER_NPU_LIB"), libs_path)
                package_data['paddle.libs'] += [
                    'libnnadapter_driver_huawei_ascend_npu' + ext_suffix
                ]

    if envir_var.get("WITH_CINN") == 'ON':
        shutil.copy(
            envir_var.get("CINN_LIB_LOCATION")
            + '/'
            + envir_var.get("CINN_LIB_NAME"),
            libs_path,
        )
        shutil.copy(
            envir_var.get("CINN_INCLUDE_DIR")
            + '/cinn/runtime/cuda/cinn_cuda_runtime_source.cuh',
            libs_path,
        )
        package_data['paddle.libs'] += ['libcinnapi.so']
        package_data['paddle.libs'] += ['cinn_cuda_runtime_source.cuh']
        if envir_var.get("CMAKE_BUILD_TYPE") == 'Release' and os.name != 'nt':
            command = (
                "patchelf --set-rpath '$ORIGIN/' %s/" % libs_path
                + envir_var.get("CINN_LIB_NAME")
            )
            if os.system(command) != 0:
                raise Exception(
                    'patch '
                    + libs_path
                    + '/'
                    + envir_var.get("CINN_LIB_NAME")
                    + ' failed',
                    'command: %s' % command,
                )

    if envir_var.get("WITH_PSLIB") == 'ON':
        shutil.copy(envir_var.get("PSLIB_LIB"), libs_path)
        if os.path.exists(envir_var.get("PSLIB_VERSION_PY")):
            shutil.copy(
                envir_var.get("PSLIB_VERSION_PY"),
                paddle_binary_dir
                + '/python/paddle/fluid/incubate/fleet/parameter_server/pslib/',
            )
        package_data['paddle.libs'] += ['libps' + ext_suffix]

    if envir_var.get("WITH_MKLDNN") == 'ON':
        if envir_var.get("CMAKE_BUILD_TYPE") == 'Release' and os.name != 'nt':
            # only change rpath in Release mode.
            # TODO(typhoonzero): use install_name_tool to patch mkl libs once
            # we can support mkl on mac.
            #
            # change rpath of libdnnl.so.1, add $ORIGIN/ to it.
            # The reason is that all thirdparty libraries in the same directory,
            # thus, libdnnl.so.1 will find libmklml_intel.so and libiomp5.so.
            command = "patchelf --set-rpath '$ORIGIN/' " + envir_var.get(
                "MKLDNN_SHARED_LIB"
            )  # -----------
            if os.system(command) != 0:
                raise Exception(
                    "patch libdnnl.so failed, command: %s" % command
                )
        shutil.copy(envir_var.get("MKLDNN_SHARED_LIB"), libs_path)
        if os.name != 'nt':
            shutil.copy(envir_var.get("MKLDNN_SHARED_LIB_1"), libs_path)
            shutil.copy(envir_var.get("MKLDNN_SHARED_LIB_2"), libs_path)
            package_data['paddle.libs'] += [
                'libmkldnn.so.0',
                'libdnnl.so.1',
                'libdnnl.so.2',
            ]
        else:
            package_data['paddle.libs'] += ['mkldnn.dll']

    if envir_var.get("WITH_ONNXRUNTIME") == 'ON':
        shutil.copy(envir_var.get("ONNXRUNTIME_SHARED_LIB"), libs_path)
        shutil.copy(envir_var.get("PADDLE2ONNX_LIB"), libs_path)
        if os.name == 'nt':
            package_data['paddle.libs'] += [
                'paddle2onnx.dll',
                'onnxruntime.dll',
            ]
        else:
            package_data['paddle.libs'] += [
                envir_var.get("PADDLE2ONNX_LIB_NAME"),
                envir_var.get("ONNXRUNTIME_LIB_NAME"),
            ]

    if envir_var.get("WITH_XPU") == 'ON':
        # only change rpath in Release mode,
        if envir_var.get("CMAKE_BUILD_TYPE") == 'Release':
            if os.name != 'nt':
                if envir_var.get("APPLE") == "1":
                    command = (
                        "install_name_tool -id \"@loader_path/\" "
                        + envir_var.get("XPU_API_LIB")
                    )
                else:
                    command = (
                        "patchelf --set-rpath '$ORIGIN/' "
                        + envir_var.get("XPU_API_LIB")
                    )
                if os.system(command) != 0:
                    raise Exception(
                        'patch ' + envir_var.get("XPU_API_LIB") + 'failed ,',
                        "command: %s" % command,
                    )
        shutil.copy(envir_var.get("XPU_API_LIB"), libs_path)
        shutil.copy(envir_var.get("XPU_RT_LIB"), libs_path)
        package_data['paddle.libs'] += [
            envir_var.get("XPU_API_LIB_NAME"),
            envir_var.get("XPU_RT_LIB_NAME"),
        ]

    if envir_var.get("WITH_XPU_BKCL") == 'ON':
        shutil.copy(envir_var.get("XPU_BKCL_LIB"), libs_path)
        package_data['paddle.libs'] += [envir_var.get("XPU_BKCL_LIB_NAME")]

    # remove unused paddle/libs/__init__.py
    if os.path.isfile(libs_path + '/__init__.py'):
        os.remove(libs_path + '/__init__.py')
    package_dir['paddle.libs'] = libs_path

    # change rpath of ${FLUID_CORE_NAME}.ext, add $ORIGIN/../libs/ to it.
    # The reason is that libwarpctc.ext, libiomp5.ext etc are in paddle.libs, and
    # ${FLUID_CORE_NAME}.ext is in paddle.fluid, thus paddle/fluid/../libs will pointer to above libraries.
    # This operation will fix https://github.com/PaddlePaddle/Paddle/issues/3213
    if envir_var.get("CMAKE_BUILD_TYPE") == 'Release':
        if os.name != 'nt':
            # only change rpath in Release mode, since in Debug mode, ${FLUID_CORE_NAME}.xx is too large to be changed.
            if envir_var.get("APPLE") == "1":
                commands = [
                    "install_name_tool -id '@loader_path/../libs/' "
                    + envir_var.get("PADDLE_BINARY_DIR")
                    + '/python/paddle/fluid/'
                    + envir_var.get("FLUID_CORE_NAME")
                    + '.so'
                ]
                commands.append(
                    "install_name_tool -add_rpath '@loader_path/../libs/' "
                    + envir_var.get("PADDLE_BINARY_DIR")
                    + '/python/paddle/fluid/'
                    + envir_var.get("FLUID_CORE_NAME")
                    + '.so'
                )
            else:
                commands = [
                    "patchelf --set-rpath '$ORIGIN/../libs/' "
                    + envir_var.get("PADDLE_BINARY_DIR")
                    + '/python/paddle/fluid/'
                    + envir_var.get("FLUID_CORE_NAME")
                    + '.so'
                ]
            # The sw_64 not suppot patchelf, so we just disable that.
            if platform.machine() != 'sw_64' and platform.machine() != 'mips64':
                for command in commands:
                    if os.system(command) != 0:
                        raise Exception(
                            'patch '
                            + envir_var.get("FLUID_CORE_NAME")
                            + '.%s failed' % ext_suffix,
                            'command: %s' % command,
                        )
    # A list of extensions that specify c++ -written modules that compile source code into dynamically linked libraries
    ext_modules = [Extension('_foo', [paddle_binary_dir + '/python/stub.cc'])]
    if os.name == 'nt':
        # fix the path separator under windows
        fix_package_dir = {}
        for k, v in package_dir.items():
            fix_package_dir[k] = v.replace('/', '\\')
        package_dir = fix_package_dir
        ext_modules = []
    elif sys.platform == 'darwin':
        ext_modules = []

    return package_data, package_dir, ext_modules


def get_headers():

    headers = (
        # paddle level api headers
        list(find_files('*.h', paddle_source_dir + '/paddle'))
        + list(find_files('*.h', paddle_source_dir + '/paddle/phi/api'))
        + list(  # phi unify api header
            find_files('*.h', paddle_source_dir + '/paddle/phi/api/ext')
        )
        + list(  # custom op api
            find_files('*.h', paddle_source_dir + '/paddle/phi/api/include')
        )
        + list(  # phi api
            find_files('*.h', paddle_source_dir + '/paddle/phi/common')
        )
        + list(
            find_files('*.h', paddle_source_dir + '/paddle/phi')
        )  # phi common headers
        # phi level api headers (low level api)
        + list(  # phi extension header
            find_files(
                '*.h', paddle_source_dir + '/paddle/phi/include', recursive=True
            )
        )
        + list(  # phi include headers
            find_files(
                '*.h',
                paddle_source_dir + '/paddle/phi/backends',
                recursive=True,
            )
        )
        + list(  # phi backends headers
            find_files(
                '*.h', paddle_source_dir + '/paddle/phi/core', recursive=True
            )
        )
        + list(  # phi core headers
            find_files(
                '*.h',
                paddle_source_dir + '/paddle/phi/infermeta',
                recursive=True,
            )
        )
        + list(  # phi infermeta headers
            find_files('*.h', paddle_source_dir + '/paddle/phi/kernels')
        )
        + list(  # phi kernels headers
            find_files('*.h', paddle_source_dir + '/paddle/phi/kernels/sparse')
        )
        + list(  # phi sparse kernels headers
            find_files(
                '*.h', paddle_source_dir + '/paddle/phi/kernels/selected_rows'
            )
        )
        + list(  # phi selected_rows kernels headers
            find_files('*.h', paddle_source_dir + '/paddle/phi/kernels/strings')
        )
        + list(  # phi sparse kernels headers
            find_files(
                '*.h', paddle_source_dir + '/paddle/phi/kernels/primitive'
            )
        )
        + list(  # phi kernel primitive api headers
            # capi headers
            find_files(
                '*.h', paddle_source_dir + '/paddle/phi/capi', recursive=True
            )
        )
        + list(  # phi capi headers
            # profiler headers
            find_files(
                'trace_event.h',
                paddle_source_dir + '/paddle/fluid/platform/profiler',
            )
        )
        + list(  # phi profiler headers
            # utils api headers
            find_files(
                '*.h', paddle_source_dir + '/paddle/utils', recursive=True
            )
        )
    )  # paddle utils headers

    jit_layer_headers = [
        'layer.h',
        'serializer.h',
        'serializer_utils.h',
        'all.h',
        'function.h',
    ]
    for f in jit_layer_headers:
        headers += list(
            find_files(
                f, paddle_source_dir + '/paddle/fluid/jit', recursive=True
            )
        )

    if envir_var.get("WITH_MKLDNN") == 'ON':
        headers += list(
            find_files('*', envir_var.get("MKLDNN_INSTALL_DIR") + '/include')
        )  # mkldnn

    if envir_var.get("WITH_GPU") == 'ON' or envir_var.get("WITH_ROCM") == 'ON':
        # externalErrorMsg.pb for External Error message
        headers += list(
            find_files('*.pb', envir_var.get("externalError_INCLUDE_DIR"))
        )
    return headers


def get_setup_parameters():
    # get setup_requires
    setup_requires = get_setup_requires()

    packages = [
        'paddle',
        'paddle.libs',
        'paddle.utils',
        'paddle.utils.gast',
        'paddle.utils.cpp_extension',
        'paddle.dataset',
        'paddle.reader',
        'paddle.distributed',
        'paddle.distributed.communication',
        'paddle.distributed.communication.stream',
        'paddle.distributed.metric',
        'paddle.distributed.ps',
        'paddle.distributed.ps.utils',
        'paddle.incubate',
        'paddle.incubate.autograd',
        'paddle.incubate.optimizer',
        'paddle.incubate.checkpoint',
        'paddle.incubate.operators',
        'paddle.incubate.tensor',
        'paddle.incubate.multiprocessing',
        'paddle.incubate.nn',
        'paddle.incubate.asp',
        'paddle.incubate.passes',
        'paddle.distribution',
        'paddle.distributed.utils',
        'paddle.distributed.sharding',
        'paddle.distributed.fleet',
        'paddle.distributed.launch',
        'paddle.distributed.launch.context',
        'paddle.distributed.launch.controllers',
        'paddle.distributed.launch.job',
        'paddle.distributed.launch.plugins',
        'paddle.distributed.launch.utils',
        'paddle.distributed.fleet.base',
        'paddle.distributed.fleet.recompute',
        'paddle.distributed.fleet.elastic',
        'paddle.distributed.fleet.meta_optimizers',
        'paddle.distributed.fleet.meta_optimizers.sharding',
        'paddle.distributed.fleet.meta_optimizers.ascend',
        'paddle.distributed.fleet.meta_optimizers.dygraph_optimizer',
        'paddle.distributed.fleet.runtime',
        'paddle.distributed.rpc',
        'paddle.distributed.fleet.dataset',
        'paddle.distributed.fleet.data_generator',
        'paddle.distributed.fleet.metrics',
        'paddle.distributed.fleet.proto',
        'paddle.distributed.fleet.utils',
        'paddle.distributed.fleet.layers',
        'paddle.distributed.fleet.layers.mpu',
        'paddle.distributed.fleet.meta_parallel',
        'paddle.distributed.fleet.meta_parallel.pp_utils',
        'paddle.distributed.fleet.meta_parallel.sharding',
        'paddle.distributed.fleet.meta_parallel.parallel_layers',
        'paddle.distributed.auto_parallel',
        'paddle.distributed.auto_parallel.operators',
        'paddle.distributed.auto_parallel.tuner',
        'paddle.distributed.auto_parallel.cost',
        'paddle.distributed.passes',
        'paddle.distributed.models',
        'paddle.distributed.models.moe',
        'paddle.framework',
        'paddle.jit',
        'paddle.jit.dy2static',
        'paddle.inference',
        'paddle.inference.contrib',
        'paddle.inference.contrib.utils',
        'paddle.fluid',
        # 'paddle.fluid.inference',
        'paddle.fluid.dygraph',
        'paddle.fluid.dygraph.dygraph_to_static',
        'paddle.fluid.dygraph.amp',
        'paddle.fluid.proto',
        'paddle.fluid.proto.profiler',
        'paddle.fluid.distributed',
        'paddle.fluid.layers',
        'paddle.fluid.dataloader',
        'paddle.fluid.contrib',
        'paddle.fluid.contrib.quantize',
        'paddle.fluid.contrib.slim',
        'paddle.fluid.contrib.slim.quantization',
        'paddle.fluid.contrib.slim.quantization.imperative',
        'paddle.fluid.contrib.extend_optimizer',
        'paddle.fluid.contrib.mixed_precision',
        'paddle.fluid.contrib.mixed_precision.bf16',
        'paddle.fluid.contrib.layers',
        'paddle.fluid.contrib.sparsity',
        'paddle.fluid.transpiler',
        'paddle.fluid.transpiler.details',
        'paddle.fluid.incubate',
        'paddle.fluid.incubate.data_generator',
        'paddle.fluid.incubate.fleet',
        'paddle.fluid.incubate.checkpoint',
        'paddle.fluid.incubate.fleet.base',
        'paddle.fluid.incubate.fleet.parameter_server',
        'paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler',
        'paddle.fluid.incubate.fleet.parameter_server.pslib',
        'paddle.fluid.incubate.fleet.parameter_server.ir',
        'paddle.fluid.incubate.fleet.collective',
        'paddle.fluid.incubate.fleet.utils',
        'paddle.amp',
        'paddle.cost_model',
        'paddle.hapi',
        'paddle.vision',
        'paddle.vision.models',
        'paddle.vision.transforms',
        'paddle.vision.datasets',
        'paddle.audio',
        'paddle.audio.functional',
        'paddle.audio.features',
        'paddle.audio.datasets',
        'paddle.audio.backends',
        'paddle.text',
        'paddle.text.datasets',
        'paddle.incubate',
        'paddle.incubate.nn',
        'paddle.incubate.nn.functional',
        'paddle.incubate.nn.layer',
        'paddle.incubate.optimizer.functional',
        'paddle.incubate.autograd',
        'paddle.incubate.distributed',
        'paddle.incubate.distributed.utils',
        'paddle.incubate.distributed.utils.io',
        'paddle.incubate.distributed.fleet',
        'paddle.incubate.distributed.models',
        'paddle.incubate.distributed.models.moe',
        'paddle.incubate.distributed.models.moe.gate',
        'paddle.sparse',
        'paddle.sparse.nn',
        'paddle.sparse.nn.layer',
        'paddle.sparse.nn.functional',
        'paddle.incubate.xpu',
        'paddle.io',
        'paddle.optimizer',
        'paddle.nn',
        'paddle.nn.functional',
        'paddle.nn.layer',
        'paddle.nn.quant',
        'paddle.nn.initializer',
        'paddle.nn.utils',
        'paddle.metric',
        'paddle.static',
        'paddle.static.nn',
        'paddle.static.amp',
        'paddle.static.sparsity',
        'paddle.tensor',
        'paddle.onnx',
        'paddle.autograd',
        'paddle.device',
        'paddle.device.cuda',
        'paddle.device.xpu',
        'paddle.version',
        'paddle.profiler',
        'paddle.geometric',
        'paddle.geometric.message_passing',
        'paddle.geometric.sampling',
    ]

    # packages=find_packages('python',exclude=['tools'])
    # get scripts
    paddle_bins = ''
    if not envir_var.get("WIN32"):
        paddle_bins = [
            envir_var.get("PADDLE_BINARY_DIR") + '/paddle/scripts/paddle'
        ]

    # get package_data and package_dir
    package_data, package_dir, ext_modules = get_package_data_and_package_dir()

    # get headers
    headers = get_headers()

    return (
        setup_requires,
        packages,
        paddle_bins,
        package_data,
        package_dir,
        ext_modules,
        headers,
    )


build_help_message = """
    The following commands of setup.py may help you

    if you want to install paddle and get the package that is stable and does not require you to edit, modify, or debug,plz run:
        $ python setup.py install

    if you want to install a package that requires you to modify it so that you have to reinstall it,plz run:
        $ python setup.py develop

    if you want to rerun cmake to re-generate native build files,plz run:
        $ python setuo.py develop rerun-cmake

    if you want to cmake only and then adjust build options by yourself, plz run:
        $ python setuo.py develop only-cmake
"""


def print_info_of_reminding(help_messages):
    print(help_messages)


def main():

    # Parse the command line and check arguments before we proceed with building steps and setup
    parse_input_command(filter_args_list)

    # os.system('pip install ninja')
    # Execute the build process,cmake and make
    if cmake_and_make:
        build_steps()

    # envir_dict is environment variables after cmake
    sys.path.append(TOP_DIR + "/build/python/")
    from build.python.env_dict import env_dict

    global envir_var
    envir_var = env_dict

    global paddle_binary_dir, paddle_source_dir
    paddle_binary_dir = envir_var.get("PADDLE_BINARY_DIR")
    paddle_source_dir = envir_var.get("PADDLE_SOURCE_DIR")

    # preparing parameters for setup()

    paddle_version = envir_var.get("PADDLE_VERSION")
    package_name = envir_var.get("PACKAGE_NAME")

    write_version_py(
        filename=paddle_binary_dir + '/python/paddle/version/__init__.py'
    )
    write_cuda_env_config_py(
        filename=paddle_binary_dir + '/python/paddle/cuda_env.py'
    )
    write_distributed_training_mode_py(
        filename=paddle_binary_dir
        + '/python/paddle/fluid/incubate/fleet/parameter_server/version.py'
    )

    (
        setup_requires,
        packages,
        scripts,
        package_data,
        package_dir,
        ext_modules,
        headers,
    ) = get_setup_parameters()

    # Log for PYPI, get long_description of setup()
    with open(
        paddle_source_dir + '/python/paddle/README.rst', "r", encoding='UTF-8'
    ) as f:
        long_description = f.read()

    # strip *.so to reduce package size
    if envir_var.get("WITH_STRIP") == 'ON':
        command = (
            'find '
            + paddle_binary_dir
            + '/python/paddle -name "*.so" | xargs -i strip {}'
        )
        if os.system(command) != 0:
            raise Exception("strip *.so failed, command: %s" % command)

    setup(
        name=package_name,
        version=paddle_version,
        description='Parallel Distributed Deep Learning',
        long_description=long_description,
        long_description_content_type="text/markdown",
        author_email="Paddle-better@baidu.com",
        maintainer="PaddlePaddle",
        maintainer_email="Paddle-better@baidu.com",
        url='https://www.paddlepaddle.org.cn/',
        download_url='https://github.com/paddlepaddle/paddle',
        license='Apache Software License',
        packages=packages,
        install_requires=setup_requires,
        ext_modules=ext_modules,
        package_data=package_data,
        package_dir=package_dir,
        scripts=scripts,
        distclass=BinaryDistribution,
        headers=headers,
        cmdclass={
            'install_headers': InstallHeaders,
            'install': InstallCommand,
            'egg_info': EggInfo,
            'install_lib': InstallLib,
        },
        entry_points={
            'console_scripts': [
                'fleetrun = paddle.distributed.launch.main:launch'
            ]
        },
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Operating System :: OS Independent',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: C++',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
    )
    if os.path.exists('${SETUP_LOG_FILE}'):
        os.system('grep -v "purelib" ${SETUP_LOG_FILE}')
    # print_info_of_reminding(build_help_message)


if __name__ == '__main__':
    main()
