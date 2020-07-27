# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import subprocess
import os
import os.path
import re
import shutil
import sys
import fnmatch
import errno
import platform

from contextlib import contextmanager
from setuptools import Command
from setuptools import setup, Distribution, Extension
from setuptools.command.install import install as InstallCommandBase


class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True


RC = 0

ext_name = '.dll' if os.name == 'nt' else ('.dylib' if sys.platform == 'darwin'
                                           else '.so')


def git_commit():
    try:
        cmd = ['git', 'rev-parse', 'HEAD']
        git_commit = subprocess.Popen(
            cmd, stdout=subprocess.PIPE,
            cwd="@PADDLE_SOURCE_DIR@").communicate()[0].strip()
    except:
        git_commit = 'Unknown'
    git_commit = git_commit.decode()
    return str(git_commit)


def _get_version_detail(idx):
    assert idx < 3, "vesion info consists of %(major)d.%(minor)d.%(patch)d, \
        so detail index must less than 3"

    if re.match('@TAG_VERSION_REGEX@', '@PADDLE_VERSION@'):
        version_details = '@PADDLE_VERSION@'.split('.')

        if len(version_details) >= 3:
            return version_details[idx]

    return 0


def get_major():
    return int(_get_version_detail(0))


def get_minor():
    return int(_get_version_detail(1))


def get_patch():
    return str(_get_version_detail(2))


def is_taged():
    try:
        cmd = [
            'git', 'describe', '--exact-match', '--tags', 'HEAD', '2>/dev/null'
        ]
        git_tag = subprocess.Popen(
            cmd, stdout=subprocess.PIPE,
            cwd="@PADDLE_SOURCE_DIR@").communicate()[0].strip()
        git_tag = git_tag.decode()
    except:
        return False

    if str(git_tag).replace('v', '') == '@PADDLE_VERSION@':
        return True
    else:
        return False


def write_version_py(filename='paddle/version.py'):
    cnt = '''# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY
#
full_version    = '%(major)d.%(minor)d.%(patch)s'
major           = '%(major)d'
minor           = '%(minor)d'
patch           = '%(patch)s'
rc              = '%(rc)d'
istaged         = %(istaged)s
commit          = '%(commit)s'
with_mkl        = '%(with_mkl)s'

def show():
    if istaged:
        print('full_version:', full_version)
        print('major:', major)
        print('minor:', minor)
        print('patch:', patch)
        print('rc:', rc)
    else:
        print('commit:', commit)

def mkl():
    return with_mkl
'''
    commit = git_commit()
    with open(filename, 'w') as f:
        f.write(cnt % {
            'major': get_major(),
            'minor': get_minor(),
            'patch': get_patch(),
            'rc': RC,
            'version': '${PADDLE_VERSION}',
            'commit': commit,
            'istaged': is_taged(),
            'with_mkl': '@WITH_MKL@'
        })


write_version_py(filename='@PADDLE_BINARY_DIR@/python/paddle/version.py')


def write_distributed_training_mode_py(
        filename='paddle/fluid/incubate/fleet/parameter_server/version.py'):
    cnt = '''from __future__ import print_function

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
        f.write(cnt %
                {'mode': 'PSLIB' if '${WITH_PSLIB}' == 'ON' else 'TRANSPILER'})


write_distributed_training_mode_py(
    filename='@PADDLE_BINARY_DIR@/python/paddle/fluid/incubate/fleet/parameter_server/version.py'
)

packages = [
    'paddle',
    'paddle.libs',
    'paddle.utils',
    'paddle.dataset',
    'paddle.reader',
    'paddle.distributed',
    'paddle.incubate',
    'paddle.incubate.complex',
    'paddle.incubate.complex.tensor',
    'paddle.fleet',
    'paddle.fleet.base',
    'paddle.fleet.meta_optimizers',
    'paddle.fleet.runtime',
    'paddle.fleet.dataset',
    'paddle.fleet.metrics',
    'paddle.fleet.proto',
    'paddle.framework',
    'paddle.fluid',
    'paddle.fluid.dygraph',
    'paddle.fluid.dygraph.dygraph_to_static',
    'paddle.fluid.proto',
    'paddle.fluid.proto.profiler',
    'paddle.fluid.distributed',
    'paddle.fluid.layers',
    'paddle.fluid.dataloader',
    'paddle.fluid.contrib',
    'paddle.fluid.contrib.decoder',
    'paddle.fluid.contrib.quantize',
    'paddle.fluid.contrib.reader',
    'paddle.fluid.contrib.slim',
    'paddle.fluid.contrib.slim.core',
    'paddle.fluid.contrib.slim.graph',
    'paddle.fluid.contrib.slim.prune',
    'paddle.fluid.contrib.slim.quantization',
    'paddle.fluid.contrib.slim.quantization.imperative',
    'paddle.fluid.contrib.slim.distillation',
    'paddle.fluid.contrib.slim.nas',
    'paddle.fluid.contrib.slim.searcher',
    'paddle.fluid.contrib.utils',
    'paddle.fluid.contrib.extend_optimizer',
    'paddle.fluid.contrib.mixed_precision',
    'paddle.fluid.contrib.layers',
    'paddle.fluid.transpiler',
    'paddle.fluid.transpiler.details',
    'paddle.fluid.incubate',
    'paddle.fluid.incubate.data_generator',
    'paddle.fluid.incubate.fleet',
    'paddle.fluid.incubate.fleet.base',
    'paddle.fluid.incubate.fleet.parameter_server',
    'paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler',
    'paddle.fluid.incubate.fleet.parameter_server.pslib',
    'paddle.fluid.incubate.fleet.collective',
    'paddle.fluid.incubate.fleet.utils',
    'paddle.incubate.hapi',
    'paddle.incubate.hapi.datasets',
    'paddle.incubate.hapi.vision',
    'paddle.incubate.hapi.vision.models',
    'paddle.incubate.hapi.vision.transforms',
    'paddle.incubate.hapi.text',
    'paddle.incubate',
    'paddle.io',
    'paddle.optimizer',
    'paddle.nn',
    'paddle.nn.functional',
    'paddle.nn.layer',
    'paddle.nn.initializer',
    'paddle.metric',
    'paddle.imperative',
    'paddle.imperative.jit',
    'paddle.tensor',
]

with open('@PADDLE_SOURCE_DIR@/python/requirements.txt') as f:
    setup_requires = f.read().splitlines()

# Note(wangzhongpu):
# When compiling paddle under python36, the dependencies belonging to python2.7 will be imported, resulting in errors when installing paddle
if sys.version_info >= (3, 6) and sys.version_info < (3, 7):
    setup_requires_tmp = []
    for setup_requires_i in setup_requires:
        if "<\"3.6\"" in setup_requires_i or "<\"3.5\"" in setup_requires_i or "<=\"3.5\"" in setup_requires_i:
            continue
        setup_requires_tmp += [setup_requires_i]
    setup_requires = setup_requires_tmp
if sys.version_info >= (3, 5) and sys.version_info < (3, 6):
    setup_requires_tmp = []
    for setup_requires_i in setup_requires:
        if "<\"3.5\"" in setup_requires_i:
            continue
        setup_requires_tmp += [setup_requires_i]
    setup_requires = setup_requires_tmp
if sys.version_info >= (3, 7):
    setup_requires_tmp = []
    for setup_requires_i in setup_requires:
        if "<\"3.6\"" in setup_requires_i or "<=\"3.6\"" in setup_requires_i or "<\"3.5\"" in setup_requires_i or "<=\"3.5\"" in setup_requires_i or "<\"3.7\"" in setup_requires_i:
            continue
        setup_requires_tmp += [setup_requires_i]
    setup_requires = setup_requires_tmp

if '${CMAKE_SYSTEM_PROCESSOR}' not in ['arm', 'armv7-a', 'aarch64']:
    setup_requires += ['opencv-python']

# the prefix is sys.prefix which should always be usr
paddle_bins = ''

if not '${WIN32}':
    paddle_bins = ['${PADDLE_BINARY_DIR}/paddle/scripts/paddle']
package_data = {
    'paddle.fluid':
    ['${FLUID_CORE_NAME}' + ('.so' if os.name != 'nt' else '.pyd')]
}
if '${HAS_NOAVX_CORE}' == 'ON':
    package_data['paddle.fluid'] += [
        'core_noavx' + ('.so' if os.name != 'nt' else '.pyd')
    ]

package_dir = {
    '': '${PADDLE_BINARY_DIR}/python',
    # The paddle.fluid.proto will be generated while compiling.
    # So that package points to other directory.
    'paddle.fluid.proto.profiler': '${PADDLE_BINARY_DIR}/paddle/fluid/platform',
    'paddle.fluid.proto': '${PADDLE_BINARY_DIR}/paddle/fluid/framework',
    'paddle.fluid': '${PADDLE_BINARY_DIR}/python/paddle/fluid',
}

# put all thirdparty libraries in paddle.libs
libs_path = '${PADDLE_BINARY_DIR}/python/paddle/libs'

package_data['paddle.libs'] = []
package_data['paddle.libs'] = [('libwarpctc'
                                if os.name != 'nt' else 'warpctc') + ext_name]
shutil.copy('${WARPCTC_LIBRARIES}', libs_path)

if '${WITH_MKL}' == 'ON':
    shutil.copy('${MKLML_SHARED_LIB}', libs_path)
    shutil.copy('${MKLML_SHARED_IOMP_LIB}', libs_path)
    package_data['paddle.libs'] += [
        ('libmklml_intel' if os.name != 'nt' else 'mklml') + ext_name,
        ('libiomp5' if os.name != 'nt' else 'libiomp5md') + ext_name
    ]
else:
    if os.name == 'nt':
        # copy the openblas.dll
        shutil.copy('${OPENBLAS_SHARED_LIB}', libs_path)
        package_data['paddle.libs'] += ['openblas' + ext_name]

if '${WITH_LITE}' == 'ON':
    shutil.copy('${LITE_SHARED_LIB}', libs_path)
    package_data['paddle.libs'] += ['libpaddle_full_api_shared' + ext_name]

if '${WITH_PSLIB}' == 'ON':
    shutil.copy('${PSLIB_LIB}', libs_path)
    if os.path.exists('${PSLIB_VERSION_PY}'):
        shutil.copy(
            '${PSLIB_VERSION_PY}',
            '${PADDLE_BINARY_DIR}/python/paddle/fluid/incubate/fleet/parameter_server/pslib/'
        )
    package_data['paddle.libs'] += ['libps' + ext_name]

if '${WITH_MKLDNN}' == 'ON':
    if '${CMAKE_BUILD_TYPE}' == 'Release' and os.name != 'nt':
        # only change rpath in Release mode.
        # TODO(typhoonzero): use install_name_tool to patch mkl libs once
        # we can support mkl on mac.
        #
        # change rpath of libdnnl.so.1, add $ORIGIN/ to it.
        # The reason is that all thirdparty libraries in the same directory,
        # thus, libdnnl.so.1 will find libmklml_intel.so and libiomp5.so.
        command = "patchelf --set-rpath '$ORIGIN/' ${MKLDNN_SHARED_LIB}"
        if os.system(command) != 0:
            raise Exception("patch libdnnl.so failed, command: %s" % command)
    shutil.copy('${MKLDNN_SHARED_LIB}', libs_path)
    if os.name != 'nt':
        shutil.copy('${MKLDNN_SHARED_LIB_1}', libs_path)
        package_data['paddle.libs'] += ['libmkldnn.so.0', 'libdnnl.so.1']
    else:
        package_data['paddle.libs'] += ['mkldnn.dll']

# copy libfuild_framework.so to libs
if os.name != 'nt' and sys.platform != 'darwin':
    paddle_framework_lib = '${FLUID_FRAMEWORK_SHARED_LIB}'
    shutil.copy(paddle_framework_lib, libs_path)
    package_data['paddle.libs'] += [
        ('libpaddle_framework'
         if os.name != 'nt' else 'paddle_framework') + ext_name
    ]

# remove unused paddle/libs/__init__.py
if os.path.isfile(libs_path + '/__init__.py'):
    os.remove(libs_path + '/__init__.py')
package_dir['paddle.libs'] = libs_path

# change rpath of ${FLUID_CORE_NAME}.ext, add $ORIGIN/../libs/ to it.
# The reason is that libwarpctc.ext, libiomp5.ext etc are in paddle.libs, and
# ${FLUID_CORE_NAME}.ext is in paddle.fluid, thus paddle/fluid/../libs will pointer to above libraries.
# This operation will fix https://github.com/PaddlePaddle/Paddle/issues/3213
if '${CMAKE_BUILD_TYPE}' == 'Release':
    if os.name != 'nt':
        # only change rpath in Release mode, since in Debug mode, ${FLUID_CORE_NAME}.xx is too large to be changed.
        if "@APPLE@" == "1":
            command = "install_name_tool -id \"@loader_path/../libs/\" ${PADDLE_BINARY_DIR}/python/paddle/fluid/${FLUID_CORE_NAME}" + '.so'
        else:
            command = "patchelf --set-rpath '$ORIGIN/../libs/' ${PADDLE_BINARY_DIR}/python/paddle/fluid/${FLUID_CORE_NAME}" + '.so'
        if platform.machine() != 'aarch64':
            if os.system(command) != 0:
                raise Exception(
                    "patch ${FLUID_CORE_NAME}.%s failed, command: %s" %
                    (ext_name, command))

ext_modules = [Extension('_foo', ['stub.cc'])]
if os.name == 'nt':
    # fix the path separator under windows
    fix_package_dir = {}
    for k, v in package_dir.items():
        fix_package_dir[k] = v.replace('/', '\\')
    package_dir = fix_package_dir
    ext_modules = []
elif sys.platform == 'darwin':
    ext_modules = []


def find_files(pattern, root):
    for dirpath, _, files in os.walk(root):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(dirpath, filename)


headers = (
    list(find_files('*.h', '@PADDLE_SOURCE_DIR@/paddle/fluid/framework')) +
    list(find_files('*.h', '@PADDLE_SOURCE_DIR@/paddle/fluid/imperative')) +
    list(find_files('*.h', '@PADDLE_SOURCE_DIR@/paddle/fluid/memory')) +
    list(find_files('*.h', '@PADDLE_SOURCE_DIR@/paddle/fluid/platform')) +
    list(find_files('*.h', '@PADDLE_SOURCE_DIR@/paddle/fluid/string')) +
    list(find_files('*.pb.h', '${PADDLE_BINARY_DIR}/paddle/fluid/platform')) +
    list(find_files('*.pb.h', '${PADDLE_BINARY_DIR}/paddle/fluid/framework')) +
    list(find_files('*.pb', '${cudaerror_INCLUDE_DIR}'))
    +  # errorMessage.pb for errormessage
    ['${EIGEN_INCLUDE_DIR}/Eigen/Core'] +  # eigen
    list(find_files('*', '${EIGEN_INCLUDE_DIR}/Eigen/src')) +  # eigen
    list(find_files('*', '${EIGEN_INCLUDE_DIR}/unsupported/Eigen')) +  # eigen
    list(find_files('*', '${GFLAGS_INSTALL_DIR}/include')) +  # gflags
    list(find_files('*', '${GLOG_INSTALL_DIR}/include')) +  # glog
    list(find_files('*', '${BOOST_INCLUDE_DIR}/boost')) +  # boost
    list(find_files('*', '${XXHASH_INSTALL_DIR}/include')) +  # xxhash
    list(find_files('*', '${PROTOBUF_INCLUDE_DIR}')) +  # protobuf
    list(find_files('*', '${DLPACK_INCLUDE_DIR}')) +  # dlpack
    list(find_files('*.h', '${THREADPOOL_INCLUDE_DIR}')))  # threadpool

if '${WITH_MKLDNN}' == 'ON':
    headers += list(find_files('*', '${MKLDNN_INSTALL_DIR}/include'))  # mkldnn

if '${WITH_GPU}' == 'ON':
    headers += list(find_files(
        '*.pb', '${cudaerror_INCLUDE_DIR}'))  # errorMessage.pb for errormessage


class InstallCommand(InstallCommandBase):
    def finalize_options(self):
        ret = InstallCommandBase.finalize_options(self)
        self.install_headers = os.path.join(self.install_purelib, 'paddle',
                                            'include')
        self.install_lib = self.install_platlib
        return ret


class InstallHeaders(Command):
    """Override how headers are copied.
    """
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
            'install', ('install_headers', 'install_dir'), ('force', 'force'))

    def mkdir_and_copy_file(self, header):
        if 'pb.h' in header:
            install_dir = re.sub('${PADDLE_BINARY_DIR}/', '', header)
        elif 'third_party' not in header:
            # framework
            install_dir = re.sub('@PADDLE_SOURCE_DIR@/', '', header)
        else:
            # third_party
            install_dir = re.sub('${THIRD_PARTY_PATH}', 'third_party', header)
            patterns = [
                'eigen3/src/extern_eigen3', 'boost/src/extern_boost',
                'dlpack/src/extern_dlpack/include', 'install/protobuf/include',
                'install/gflags/include', 'install/glog/include',
                'install/xxhash/include', 'install/mkldnn/include',
                'threadpool/src/extern_threadpool'
            ]
            for pattern in patterns:
                install_dir = re.sub(pattern, '', install_dir)
        install_dir = os.path.join(self.install_dir,
                                   os.path.dirname(install_dir))
        if not os.path.exists(install_dir):
            self.mkpath(install_dir)
        return self.copy_file(header, install_dir)

    def run(self):
        # only copy third_party/cudaErrorMessage.pb for cudaErrorMessage on mac or windows
        if os.name == 'nt' or sys.platform == 'darwin':
            if '${WITH_GPU}' == 'ON':
                self.mkdir_and_copy_file(
                    '${cudaerror_INCLUDE_DIR}/cudaErrorMessage.pb')
            return
        hdrs = self.distribution.headers
        if not hdrs:
            return
        self.mkpath(self.install_dir)
        for header in hdrs:
            (out, _) = self.mkdir_and_copy_file(header)
            self.outfiles.append(out)

    def get_inputs(self):
        return self.distribution.headers or []

    def get_outputs(self):
        return self.outfiles


# we redirect setuptools log for non-windows
if sys.platform != 'win32':

    @contextmanager
    def redirect_stdout():
        f_log = open('${SETUP_LOG_FILE}', 'w')
        origin_stdout = sys.stdout
        sys.stdout = f_log
        yield
        f_log = sys.stdout
        sys.stdout = origin_stdout
        f_log.close()
else:

    @contextmanager
    def redirect_stdout():
        yield


with redirect_stdout():
    setup(
        name='${PACKAGE_NAME}',
        version='${PADDLE_VERSION}',
        description='Parallel Distributed Deep Learning',
        install_requires=setup_requires,
        packages=packages,
        ext_modules=ext_modules,
        package_data=package_data,
        package_dir=package_dir,
        scripts=paddle_bins,
        distclass=BinaryDistribution,
        headers=headers,
        cmdclass={
            'install_headers': InstallHeaders,
            'install': InstallCommand,
        })

# As there are a lot of files in purelib which causes many logs,
# we don't print them on the screen, and you can open `setup.py.log`
# for the full logs.
if os.path.exists('${SETUP_LOG_FILE}'):
    os.system('grep -v "purelib" ${SETUP_LOG_FILE}')
