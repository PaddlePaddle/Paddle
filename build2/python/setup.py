# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import errno
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
from setuptools.command.egg_info import egg_info


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
            cwd="/wangxianming/myPro/Paddle").communicate()[0].strip()
    except:
        git_commit = 'Unknown'
    git_commit = git_commit.decode()
    return str(git_commit)


def _get_version_detail(idx):
    assert idx < 3, "vesion info consists of %(major)d.%(minor)d.%(patch)d, \
        so detail index must less than 3"

    if re.match('[0-9]+\.[0-9]+\.[0-9]+(\.(a|b|rc)\.[0-9]+)?', '0.0.0'):
        version_details = '0.0.0'.split('.')

        if len(version_details) >= 3:
            return version_details[idx]

    return 0


def get_major():
    return int(_get_version_detail(0))


def get_minor():
    return int(_get_version_detail(1))


def get_patch():
    return str(_get_version_detail(2))


def get_cuda_version():
    if 'ON' == 'ON':
        return '10.2'
    else:
        return 'False'


def get_cudnn_version():
    if 'ON' == 'ON':
        temp_cudnn_version = ''
        if '7':
            temp_cudnn_version += '7'
            if '6':
                temp_cudnn_version += '.6'
                if '5':
                    temp_cudnn_version += '.5'
        return temp_cudnn_version
    else:
        return 'False'


def is_taged():
    try:
        cmd = [
            'git', 'describe', '--exact-match', '--tags', 'HEAD', '2>/dev/null'
        ]
        git_tag = subprocess.Popen(
            cmd, stdout=subprocess.PIPE,
            cwd="/wangxianming/myPro/Paddle").communicate()[0].strip()
        git_tag = git_tag.decode()
    except:
        return False

    if str(git_tag).replace('v', '') == '0.0.0':
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
        f.write(cnt % {
            'major': get_major(),
            'minor': get_minor(),
            'patch': get_patch(),
            'rc': RC,
            'version': '0.0.0',
            'cuda': get_cuda_version(),
            'cudnn': get_cudnn_version(),
            'commit': commit,
            'istaged': is_taged(),
            'with_mkl': 'ON'
        })


write_version_py(
    filename='/wangxianming/myPro/Paddle/build2/python/paddle/version/__init__.py'
)


def write_cuda_env_config_py(filename='paddle/cuda_env.py'):
    cnt = ""
    if '' == 'ON':
        cnt = '''# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY
#
import os
os.environ['CUDA_CACHE_MAXSIZE'] = '805306368'
'''

    with open(filename, 'w') as f:
        f.write(cnt)


write_cuda_env_config_py(
    filename='/wangxianming/myPro/Paddle/build2/python/paddle/cuda_env.py')


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
        f.write(cnt % {'mode': 'PSLIB' if 'OFF' == 'ON' else 'TRANSPILER'})


write_distributed_training_mode_py(
    filename='/wangxianming/myPro/Paddle/build2/python/paddle/fluid/incubate/fleet/parameter_server/version.py'
)

packages = [
    'paddle',
    'paddle.libs',
    'paddle.utils',
    'paddle.utils.gast',
    'paddle.utils.cpp_extension',
    'paddle.dataset',
    'paddle.reader',
    'paddle.distributed',
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
    'paddle.incubate.passes',
    'paddle.distribution',
    'paddle.distributed.sharding',
    'paddle.distributed.fleet',
    'paddle.distributed.launch',
    'paddle.distributed.launch.context',
    'paddle.distributed.launch.controllers',
    'paddle.distributed.launch.job',
    'paddle.distributed.launch.plugins',
    'paddle.distributed.launch.utils',
    'paddle.distributed.fleet.base',
    'paddle.distributed.fleet.elastic',
    'paddle.distributed.fleet.meta_optimizers',
    'paddle.distributed.fleet.meta_optimizers.sharding',
    'paddle.distributed.fleet.meta_optimizers.ascend',
    'paddle.distributed.fleet.meta_optimizers.dygraph_optimizer',
    'paddle.distributed.fleet.runtime',
    'paddle.distributed.fleet.dataset',
    'paddle.distributed.fleet.data_generator',
    'paddle.distributed.fleet.metrics',
    'paddle.distributed.fleet.proto',
    'paddle.distributed.fleet.utils',
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
    'paddle.fluid.inference',
    'paddle.fluid.dygraph',
    'paddle.fluid.dygraph.dygraph_to_static',
    'paddle.fluid.dygraph.amp',
    'paddle.fluid.proto',
    'paddle.fluid.proto.profiler',
    'paddle.fluid.distributed',
    'paddle.fluid.layers',
    'paddle.fluid.dataloader',
    'paddle.fluid.contrib',
    'paddle.fluid.contrib.decoder',
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
    'paddle.text',
    'paddle.text.datasets',
    'paddle.incubate',
    'paddle.incubate.nn',
    'paddle.incubate.nn.functional',
    'paddle.incubate.nn.layer',
    'paddle.incubate.optimizer.functional',
    'paddle.incubate.autograd',
    'paddle.incubate.distributed',
    'paddle.incubate.distributed.models',
    'paddle.incubate.distributed.models.moe',
    'paddle.incubate.distributed.models.moe.gate',
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
    'paddle.version',
    'paddle.profiler',
    'paddle.sparse',
    'paddle.sparse.layer',
    'paddle.sparse.functional',
]

with open('/wangxianming/myPro/Paddle/python/requirements.txt') as f:
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

# the prefix is sys.prefix which should always be usr
paddle_bins = ''

if not '':
    paddle_bins = ['/wangxianming/myPro/Paddle/build2/paddle/scripts/paddle']

if os.name != 'nt':
    package_data = {'paddle.fluid': ['core_avx' + '.so']}
else:
    package_data = {'paddle.fluid': ['core_avx' + '.pyd', 'core_avx' + '.lib']}

package_data['paddle.fluid'] += [
    '/wangxianming/myPro/Paddle/build2/python/paddle/cost_model/static_op_benchmark.json'
]
if 'ON' == 'ON':
    package_data['paddle.fluid'] += [
        'core_noavx' + ('.so' if os.name != 'nt' else '.pyd')
    ]

package_dir = {
    '': '/wangxianming/myPro/Paddle/build2/python',
    # The paddle.fluid.proto will be generated while compiling.
    # So that package points to other directory.
    'paddle.fluid.proto.profiler':
    '/wangxianming/myPro/Paddle/build2/paddle/fluid/platform',
    'paddle.fluid.proto':
    '/wangxianming/myPro/Paddle/build2/paddle/fluid/framework',
    'paddle.fluid': '/wangxianming/myPro/Paddle/build2/python/paddle/fluid',
}

# put all thirdparty libraries in paddle.libs
libs_path = '/wangxianming/myPro/Paddle/build2/python/paddle/libs'

package_data['paddle.libs'] = []
package_data['paddle.libs'] = [('libwarpctc'
                                if os.name != 'nt' else 'warpctc') + ext_name]
shutil.copy(
    '/wangxianming/myPro/Paddle/build2/third_party/install/warpctc/lib/libwarpctc.so',
    libs_path)

package_data['paddle.libs'] += [
    os.path.basename(
        '/wangxianming/myPro/Paddle/build2/third_party/install/lapack/lib/liblapack.so.3'
    ), os.path.basename(
        '/wangxianming/myPro/Paddle/build2/third_party/install/lapack/lib/libblas.so.3'
    ), os.path.basename(
        '/wangxianming/myPro/Paddle/build2/third_party/install/lapack/lib/libgfortran.so.3'
    ), os.path.basename(
        '/wangxianming/myPro/Paddle/build2/third_party/install/lapack/lib/libquadmath.so.0'
    )
]
shutil.copy(
    '/wangxianming/myPro/Paddle/build2/third_party/install/lapack/lib/libblas.so.3',
    libs_path)
shutil.copy(
    '/wangxianming/myPro/Paddle/build2/third_party/install/lapack/lib/liblapack.so.3',
    libs_path)
shutil.copy(
    '/wangxianming/myPro/Paddle/build2/third_party/install/lapack/lib/libgfortran.so.3',
    libs_path)
shutil.copy(
    '/wangxianming/myPro/Paddle/build2/third_party/install/lapack/lib/libquadmath.so.0',
    libs_path)
if not sys.platform.startswith("linux"):
    package_data['paddle.libs'] += [os.path.basename('')]
    shutil.copy('', libs_path)

if 'ON' == 'ON':
    shutil.copy(
        '/wangxianming/myPro/Paddle/build2/third_party/install/mklml/lib/libmklml_intel.so',
        libs_path)
    shutil.copy(
        '/wangxianming/myPro/Paddle/build2/third_party/install/mklml/lib/libiomp5.so',
        libs_path)
    package_data['paddle.libs'] += [
        ('libmklml_intel' if os.name != 'nt' else 'mklml') + ext_name,
        ('libiomp5' if os.name != 'nt' else 'libiomp5md') + ext_name
    ]
else:
    if os.name == 'nt':
        # copy the openblas.dll
        shutil.copy('', libs_path)
        package_data['paddle.libs'] += ['openblas' + ext_name]
    elif os.name == 'posix' and platform.machine() == 'aarch64' and ''.endswith(
            'so'):
        # copy the libopenblas.so on linux+aarch64
        # special: core_noavx.so depends on 'libopenblas.so.0', not 'libopenblas.so'
        if os.path.exists('' + '.0'):
            shutil.copy('' + '.0', libs_path)
            package_data['paddle.libs'] += ['libopenblas.so.0']

if 'OFF' == 'ON':
    shutil.copy('', libs_path)
    package_data['paddle.libs'] += ['libpaddle_full_api_shared' + ext_name]
    if '' == 'ON':
        shutil.copy('', libs_path)
        package_data['paddle.libs'] += ['libnnadapter' + ext_name]
        if '' == 'ON':
            shutil.copy('', libs_path)
            package_data['paddle.libs'] += [
                'libnnadapter_driver_huawei_ascend_npu' + ext_name
            ]

if 'OFF' == 'ON':
    shutil.copy('/', libs_path)
    shutil.copy('/cinn/runtime/cuda/cinn_cuda_runtime_source.cuh', libs_path)
    package_data['paddle.libs'] += ['libcinnapi.so']
    package_data['paddle.libs'] += ['cinn_cuda_runtime_source.cuh']

if 'OFF' == 'ON':
    shutil.copy('', libs_path)
    if os.path.exists(''):
        shutil.copy(
            '',
            '/wangxianming/myPro/Paddle/build2/python/paddle/fluid/incubate/fleet/parameter_server/pslib/'
        )
    package_data['paddle.libs'] += ['libps' + ext_name]

if 'ON' == 'ON':
    if 'Release' == 'Release' and os.name != 'nt':
        # only change rpath in Release mode.
        # TODO(typhoonzero): use install_name_tool to patch mkl libs once
        # we can support mkl on mac.
        #
        # change rpath of libdnnl.so.1, add $ORIGIN/ to it.
        # The reason is that all thirdparty libraries in the same directory,
        # thus, libdnnl.so.1 will find libmklml_intel.so and libiomp5.so.
        command = "patchelf --set-rpath '$ORIGIN/' /wangxianming/myPro/Paddle/build2/third_party/install/mkldnn/libmkldnn.so.0"
        if os.system(command) != 0:
            raise Exception("patch libdnnl.so failed, command: %s" % command)
    shutil.copy(
        '/wangxianming/myPro/Paddle/build2/third_party/install/mkldnn/libmkldnn.so.0',
        libs_path)
    if os.name != 'nt':
        shutil.copy(
            '/wangxianming/myPro/Paddle/build2/third_party/install/mkldnn/libdnnl.so.1',
            libs_path)
        shutil.copy(
            '/wangxianming/myPro/Paddle/build2/third_party/install/mkldnn/libdnnl.so.2',
            libs_path)
        package_data['paddle.libs'] += [
            'libmkldnn.so.0', 'libdnnl.so.1', 'libdnnl.so.2'
        ]
    else:
        package_data['paddle.libs'] += ['mkldnn.dll']

if 'OFF' == 'ON':
    shutil.copy('', libs_path)
    if os.name == 'nt':
        shutil.copy('', libs_path)
        package_data['paddle.libs'] += ['paddle2onnx.dll', 'onnxruntime.dll']
    else:
        shutil.copy('', libs_path)
        if sys.platform == 'darwin':
            package_data['paddle.libs'] += [
                'libpaddle2onnx.dylib', 'libonnxruntime.1.10.0.dylib'
            ]
        else:
            package_data['paddle.libs'] += [
                'libpaddle2onnx.so', 'libonnxruntime.so.1.10.0'
            ]

if 'OFF' == 'ON':
    # only change rpath in Release mode,
    if 'Release' == 'Release':
        if os.name != 'nt':
            if "" == "1":
                command = "install_name_tool -id \"@loader_path/\" "
            else:
                command = "patchelf --set-rpath '$ORIGIN/' "
            if os.system(command) != 0:
                raise Exception("patch  failed, command: %s" % command)
    shutil.copy('', libs_path)
    shutil.copy('', libs_path)
    package_data['paddle.libs'] += ['', '']

if 'OFF' == 'ON':
    shutil.copy('', libs_path)
    package_data['paddle.libs'] += ['']

# remove unused paddle/libs/__init__.py
if os.path.isfile(libs_path + '/__init__.py'):
    os.remove(libs_path + '/__init__.py')
package_dir['paddle.libs'] = libs_path

# change rpath of core_avx.ext, add $ORIGIN/../libs/ to it.
# The reason is that libwarpctc.ext, libiomp5.ext etc are in paddle.libs, and
# core_avx.ext is in paddle.fluid, thus paddle/fluid/../libs will pointer to above libraries.
# This operation will fix https://github.com/PaddlePaddle/Paddle/issues/3213
if 'Release' == 'Release':
    if os.name != 'nt':
        # only change rpath in Release mode, since in Debug mode, core_avx.xx is too large to be changed.
        if "" == "1":
            commands = [
                "install_name_tool -id '@loader_path/../libs/' /wangxianming/myPro/Paddle/build2/python/paddle/fluid/core_avx"
                + '.so'
            ]
            commands.append(
                "install_name_tool -add_rpath '@loader_path/../libs/' /wangxianming/myPro/Paddle/build2/python/paddle/fluid/core_avx"
                + '.so')
        else:
            commands = [
                "patchelf --set-rpath '$ORIGIN/../libs/' /wangxianming/myPro/Paddle/build2/python/paddle/fluid/core_avx"
                + '.so'
            ]
        # The sw_64 not suppot patchelf, so we just disable that.
        if platform.machine() != 'sw_64' and platform.machine() != 'mips64':
            for command in commands:
                if os.system(command) != 0:
                    raise Exception("patch core_avx.%s failed, command: %s" %
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


def find_files(pattern, root, recursive=False):
    for dirpath, _, files in os.walk(root):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(dirpath, filename)
        if not recursive:
            break


headers = (
    # paddle level api headers
    list(find_files('*.h', '/wangxianming/myPro/Paddle/paddle')) +
    list(find_files('*.h', '/wangxianming/myPro/Paddle/paddle/phi/api'))
    +  # phi unify api header
    list(find_files('*.h', '/wangxianming/myPro/Paddle/paddle/phi/api/ext'))
    +  # custom op api
    list(
        find_files('*.h', '/wangxianming/myPro/Paddle/paddle/phi/api/include'))
    +  # phi api
    list(find_files('*.h', '/wangxianming/myPro/Paddle/paddle/phi/common'))
    +  # phi common headers
    # phi level api headers (low level api)
    list(find_files('*.h', '/wangxianming/myPro/Paddle/paddle/phi'))
    +  # phi extension header
    list(
        find_files(
            '*.h',
            '/wangxianming/myPro/Paddle/paddle/phi/include',
            recursive=True)) +  # phi include headers
    list(
        find_files(
            '*.h',
            '/wangxianming/myPro/Paddle/paddle/phi/backends',
            recursive=True)) +  # phi backends headers
    list(
        find_files(
            '*.h', '/wangxianming/myPro/Paddle/paddle/phi/core',
            recursive=True)) +  # phi core headers
    list(
        find_files(
            '*.h',
            '/wangxianming/myPro/Paddle/paddle/phi/infermeta',
            recursive=True)) +  # phi infermeta headers
    list(
        find_files(
            '*.h',
            '/wangxianming/myPro/Paddle/paddle/phi/kernels',
            recursive=True)) +  # phi kernels headers
    # utila api headers
    list(
        find_files(
            '*.h', '/wangxianming/myPro/Paddle/paddle/utils', recursive=True))
)  # paddle utils headers

if 'ON' == 'ON':
    headers += list(
        find_files(
            '*',
            '/wangxianming/myPro/Paddle/build2/third_party/install/mkldnn/include'
        ))  # mkldnn

if 'ON' == 'ON' or 'OFF' == 'ON':
    # externalErrorMsg.pb for External Error message
    headers += list(
        find_files(
            '*.pb',
            '/wangxianming/myPro/Paddle/build2/third_party/externalError/data'))


class InstallCommand(InstallCommandBase):
    def finalize_options(self):
        ret = InstallCommandBase.finalize_options(self)
        self.install_lib = self.install_platlib
        self.install_headers = os.path.join(self.install_platlib, 'paddle',
                                            'include')
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
            install_dir = re.sub('/wangxianming/myPro/Paddle/build2/', '',
                                 header)
        elif 'third_party' not in header:
            # paddle headers
            install_dir = re.sub('/wangxianming/myPro/Paddle/', '', header)
        else:
            # third_party
            install_dir = re.sub(
                '/wangxianming/myPro/Paddle/build2/third_party', 'third_party',
                header)
            patterns = ['install/mkldnn/include']
            for pattern in patterns:
                install_dir = re.sub(pattern, '', install_dir)
        install_dir = os.path.join(self.install_dir,
                                   os.path.dirname(install_dir))
        if not os.path.exists(install_dir):
            self.mkpath(install_dir)
        return self.copy_file(header, install_dir)

    def run(self):
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


class EggInfo(egg_info):
    """Copy license file into `.dist-info` folder."""

    def run(self):
        # don't duplicate license into `.dist-info` when building a distribution
        if not self.distribution.have_run.get('install', True):
            self.mkpath(self.egg_info)
            self.copy_file("/wangxianming/myPro/Paddle/LICENSE", self.egg_info)

        egg_info.run(self)


# we redirect setuptools log for non-windows
if sys.platform != 'win32':

    @contextmanager
    def redirect_stdout():
        f_log = open('setup.py.log', 'w')
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


# Log for PYPI
if sys.version_info > (3, 0):
    with open(
            "/wangxianming/myPro/Paddle/build2/python/paddle/README.rst",
            "r",
            encoding='UTF-8') as f:
        long_description = f.read()
else:
    with open("/wangxianming/myPro/Paddle/build2/python/paddle/README.rst",
              "r") as f:
        long_description = unicode(f.read(), 'UTF-8')

# strip *.so to reduce package size
if 'OFF' == 'ON':
    command = 'find /wangxianming/myPro/Paddle/build2/python/paddle -name "*.so" | xargs -i strip {}'
    if os.system(command) != 0:
        raise Exception("strip *.so failed, command: %s" % command)

with redirect_stdout():
    setup(
        name='paddlepaddle-gpu',
        version='0.0.0',
        description='Parallel Distributed Deep Learning',
        long_description=long_description,
        long_description_content_type="text/markdown",
        author_email="Paddle-better@baidu.com",
        maintainer="PaddlePaddle",
        maintainer_email="Paddle-better@baidu.com",
        project_urls={
            'Homepage': 'https://www.paddlepaddle.org.cn/',
            'Downloads': 'https://github.com/paddlepaddle/paddle'
        },
        license='Apache Software License',
        packages=packages,
        install_requires=setup_requires,
        ext_modules=ext_modules,
        package_data=package_data,
        package_dir=package_dir,
        scripts=paddle_bins,
        distclass=BinaryDistribution,
        headers=headers,
        cmdclass={
            'install_headers': InstallHeaders,
            'install': InstallCommand,
            'egg_info': EggInfo,
        },
        entry_points={
            'console_scripts':
            ['fleetrun = paddle.distributed.launch.main:launch']
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
        ], )

# As there are a lot of files in purelib which causes many logs,
# we don't print them on the screen, and you can open `setup.py.log`
# for the full logs.
if os.path.exists('setup.py.log'):
    os.system('grep -v "purelib" setup.py.log')
