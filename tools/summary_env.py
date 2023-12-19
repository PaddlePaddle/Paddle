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
import platform
import subprocess
import sys

import distro

envs_template = """
Paddle version: {paddle_version}
Paddle With CUDA: {paddle_with_cuda}

OS: {os_info}
GCC version: {gcc_version}
Clang version: {clang_version}
CMake version: {cmake_version}
Libc version: {libc_version}
Python version: {python_version}

CUDA version: {cuda_version}
cuDNN version: {cudnn_version}
Nvidia driver version: {nvidia_driver_version}
Nvidia driver List: {nvidia_gpu_driver}
"""

envs = {}


def get_paddle_info():
    try:
        import paddle

        envs['paddle_version'] = paddle.__version__
        envs['paddle_with_cuda'] = paddle.base.core.is_compiled_with_cuda()
    except:
        envs['paddle_version'] = 'N/A'
        envs['paddle_with_cuda'] = 'N/A'


def get_os_info():
    if platform.system() == "Darwin":
        plat = "macOS"
        ver = run_shell_command('sw_vers -productVersion').strip('\n')
    elif platform.system() == "Linux":
        plat = distro.id()
        ver = distro.version()
    elif platform.system() == "Windows":
        plat = "Windows"
        ver = platform.win32_ver()[0]
    else:
        plat = 'N/A'
        ver = 'N/A'
    envs['os_info'] = f"{plat} {ver}"


def get_gcc_version():
    try:
        envs['gcc_version'] = (
            run_shell_command("gcc --version").split('\n')[0].split("gcc ")[1]
        )
    except:
        envs['gcc_version'] = 'N/A'


def get_clang_version():
    try:
        envs['clang_version'] = (
            run_shell_command("clang --version")
            .split('\n')[0]
            .split("clang version ")[1]
        )
    except:
        envs['clang_version'] = 'N/A'


def get_cmake_version():
    try:
        envs['cmake_version'] = (
            run_shell_command("cmake --version")
            .split('\n')[0]
            .split("cmake ")[1]
        )
    except:
        envs['cmake_version'] = 'N/A'


def get_libc_version():
    if platform.system() == "Linux":
        envs['libc_version'] = ' '.join(platform.libc_ver())
    else:
        envs['libc_version'] = 'N/A'


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
        envs['cuda_version'] = 'N/A'


def get_cudnn_info():
    def _get_cudnn_ver(cmd):
        out = run_shell_command(cmd)
        if out:
            return out.split(' ')[-1].strip()
        else:
            return 'N/A'

    if platform.system() == "Windows":
        cudnn_dll_path = run_shell_command('where cudnn*')
        if cudnn_dll_path:
            cudnn_header_path = (
                cudnn_dll_path.split('bin')[0] + r'include\cudnn.h'
            )
            cmd = 'type "{0}" | findstr "{1}" | findstr /v "CUDNN_VERSION"'
        else:
            envs['cudnn_version'] = 'N/A'
            return
    else:
        cudnn_header_path = run_shell_command(
            'whereis "cudnn.h" | awk \'{print $2}\''
        ).strip('\n')
        if cudnn_header_path:
            cmd = 'cat "{0}" | grep "{1}" | grep -v "CUDNN_VERSION"'
            if _get_cudnn_ver(cmd.format(cudnn_header_path, 'CUDNN_MAJOR')):
                cudnn_header_path = run_shell_command(
                    'whereis "cudnn_version.h" | awk \'{print $2}\''
                ).strip('\n')

        else:
            envs['cudnn_version'] = 'N/A'
            return

    major = _get_cudnn_ver(cmd.format(cudnn_header_path, 'CUDNN_MAJOR'))
    minor = _get_cudnn_ver(cmd.format(cudnn_header_path, 'CUDNN_MINOR'))
    patch_level = _get_cudnn_ver(
        cmd.format(cudnn_header_path, 'CUDNN_PATCHLEVEL')
    )

    if major != 'N/A':
        envs['cudnn_version'] = f"{major}.{minor}.{patch_level}"
    else:
        envs['cudnn_version'] = 'N/A'


def get_driver_info():
    driver_ver = run_shell_command('nvidia-smi')
    if driver_ver:
        driver_ver = (
            driver_ver.split('Driver Version:')[1].strip().split(' ')[0]
        )
    else:
        driver_ver = 'N/A'
    envs['nvidia_driver_version'] = driver_ver


def get_nvidia_gpu_driver():
    if platform.system() != "Windows" and platform.system() != "Linux":
        envs['nvidia_gpu_driver'] = 'N/A'
        return
    try:
        nvidia_smi = 'nvidia-smi'
        gpu_list = run_shell_command(nvidia_smi + " -L")
        result = "\n"
        for gpu_info in gpu_list.split("\n"):
            result += gpu_info.split(" (UUID:")[0] + "\n"
        envs['nvidia_gpu_driver'] = result
    except:
        envs['nvidia_gpu_driver'] = 'N/A'


def main():
    get_paddle_info()
    get_os_info()
    get_gcc_version()
    get_clang_version()
    get_cmake_version()
    get_libc_version()
    get_python_info()
    get_cuda_info()
    get_cudnn_info()
    get_driver_info()
    get_nvidia_gpu_driver()
    print('*' * 40 + envs_template.format(**envs) + '*' * 40)


if __name__ == '__main__':
    main()
