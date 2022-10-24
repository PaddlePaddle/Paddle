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
import sys
import distro
import platform
import subprocess

envs_template = """
Paddle version: {paddle_version}
Paddle With CUDA: {paddle_with_cuda}

OS: {os_info}
Python version: {python_version}

CUDA version: {cuda_version}
cuDNN version: {cudnn_version}
Nvidia driver version: {nvidia_driver_version}
"""

envs = {}


def get_paddle_info():
    try:
        import paddle

        envs['paddle_version'] = paddle.__version__
        envs['paddle_with_cuda'] = paddle.fluid.core.is_compiled_with_cuda()
    except:
        envs['paddle_version'] = None
        envs['paddle_with_cuda'] = None


def get_os_info():
    plat = platform.system()
    if platform.system() == "Darwin":
        plat = "macOs"
        ver = platform.mac_ver()[0]
    elif platform.system() == "Linux":
        plat = distro.linux_distribution()[0]
        ver = distro.linux_distribution()[1]
    elif platform.system() == "Windows":
        plat = "Windows"
        ver = platform.win32_ver()[0]
    else:
        plat = None
        ver = None
    envs['os_info'] = "{0} {1}".format(plat, ver)


def get_python_info():
    envs['python_version'] = sys.version.split(' ')[0]


def run_shell_command(cmd):
    out, err = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    ).communicate()
    if err:
        return None
    else:
        return out.decode('utf-8')


def get_cuda_info():
    out = run_shell_command('nvcc --version')
    if out:
        envs['cuda_version'] = out.split('V')[-1].strip()
    else:
        envs['cuda_version'] = None


def get_cudnn_info():
    def _get_cudnn_ver(cmd):
        out = run_shell_command(cmd)
        if out:
            return out.split(' ')[-1].strip()
        else:
            return None

    if platform.system() == "Windows":
        cudnn_dll_path = run_shell_command('where cudnn*')
        if cudnn_dll_path:
            cudnn_header_path = (
                cudnn_dll_path.split('bin')[0] + r'include\cudnn.h'
            )
            cmd = 'type "{0}" | findstr "{1}" | findstr /v "CUDNN_VERSION"'
        else:
            envs['cudnn_version'] = None
            return
    else:
        cudnn_header_path = run_shell_command(
            'whereis "cudnn.h" | awk \'{print $2}\''
        )
        if cudnn_header_path:
            cudnn_header_path = cudnn_header_path.strip()
            cmd = 'cat "{0}" | grep "{1}" | grep -v "CUDNN_VERSION"'
        else:
            envs['cudnn_version'] = None
            return

    major = _get_cudnn_ver(cmd.format(cudnn_header_path, 'CUDNN_MAJOR'))
    minor = _get_cudnn_ver(cmd.format(cudnn_header_path, 'CUDNN_MINOR'))
    patch_level = _get_cudnn_ver(
        cmd.format(cudnn_header_path, 'CUDNN_PATCHLEVEL')
    )

    envs['cudnn_version'] = "{0}.{1}.{2}".format(major, minor, patch_level)


def get_driver_info():
    driver_ver = run_shell_command('nvidia-smi')
    if driver_ver:
        driver_ver = (
            driver_ver.split('Driver Version:')[1].strip().split(' ')[0]
        )
    else:
        driver_ver = None
    envs['nvidia_driver_version'] = driver_ver


def main():
    get_paddle_info()
    get_os_info()
    get_python_info()
    get_cuda_info()
    get_cudnn_info()
    get_driver_info()
    print('*' * 40 + envs_template.format(**envs) + '*' * 40)


if __name__ == '__main__':
    main()
