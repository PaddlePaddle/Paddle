#!/bin/python

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

import argparse
import os
import platform
import string
import yaml

package_path = os.getenv('PACKAGEPATH', default='/package')

def parse_args():
    parser = argparse.ArgumentParser("conda build for paddlepaddle version")
    parser.add_argument(
        "--paddle_version",
        type=str,
        required=True,
        help="paddle version for conda build.",
    )
    args = parser.parse_args()

    return args


class ConstantVar:
    def __init__(self):
        self.py_str = ["py38", "py39", "py310", "py311", "py312"]
        self.cuda_info = ["cuda11.8", "cuda12.3"]
        self.py_ver = {
            "py38":"python>=3.8, <3.9",
            "py39":"python>=3.9, <3.10",
            "py310":"python>=3.10, <3.11",
            "py311":"python>=3.11, <3.12",
            "py312":"python>=3.12, <3.13"
        }


def template_full(name, version, packages_string, python_version):
    # 读取模板文件
    with open('conda_build_template.yaml', 'r') as template_file:
        template_content = template_file.read()

    # 使用字典填充模板
    data = {
        'name': name,
        'version': version,
        'packages_string': packages_string,
        'python_version':python_version
    }

    # 使用 str.format 进行占位符替换
    filled_content = template_content.format(**data)

    # 将填充后的内容解析为YAML
    filled_yaml = yaml.safe_load(filled_content)

    # 将填充后的内容写入新的YAML文件
    with open('meta.yaml', 'w') as new_file:
        yaml.safe_dump(filled_yaml, new_file, default_flow_style=False, sort_keys=False)


def gen_build_scripts(name, cuda_major_version, paddle_version):
    sysstr = platform.system()
    if sysstr == "Linux":
        build_filename = "build.sh"
        PADDLE_CUDA_INSTALL_REQUIREMENTS = {
            "cuda11.8": (
                "nvidia-cuda-runtime-cu11==11.8.89 | "
                "nvidia-cuda-cupti-cu11==11.8.87 | "
                "nvidia-cudnn-cu11==8.7.0.84 | "
                "nvidia-cublas-cu11==11.11.3.6 | "
                "nvidia-cufft-cu11==10.9.0.58 | "
                "nvidia-curand-cu11==10.3.0.86 | "
                "nvidia-cusolver-cu11==11.4.1.48 | "
                "nvidia-cusparse-cu11==11.7.5.86 | "
                "nvidia-nccl-cu11==2.19.3 | "
                "nvidia-nvtx-cu11==11.8.86 | "
                "nvidia-cuda-nvrtc-cu11==11.8.89"
            ),
            "cuda12.3": (
                "nvidia-cuda-runtime-cu12==12.3.101 | "
                "nvidia-cuda-cupti-cu12==12.3.101 | "
                "nvidia-cudnn-cu12==9.0.0.312 | "
                "nvidia-cublas-cu12==12.3.4.1 | "
                "nvidia-cufft-cu12==11.2.1.3 | "
                "nvidia-curand-cu12==10.3.5.147 | "
                "nvidia-cusolver-cu12==11.6.1.9 | "
                "nvidia-cusparse-cu12==12.3.1.170 | "
                "nvidia-nccl-cu12==2.19.3 | "
                "nvidia-nvtx-cu12==12.4.127 | "
                "nvidia-cuda-nvrtc-cu12==12.3.107"
            ),
        }
    elif platform.system() == 'Windows':
        build_filename = "bld.bat"
        PADDLE_CUDA_INSTALL_REQUIREMENTS = {
            "cuda11.8": (
                "nvidia-cuda-runtime-cu11==11.8.89 | "
                "nvidia-cudnn-cu11==8.9.4.19 | "
                "nvidia-cublas-cu11==11.11.3.6 | "
                "nvidia-cufft-cu11==10.9.0.58 | "
                "nvidia-curand-cu11==10.3.0.86 | "
                "nvidia-cusolver-cu11==11.4.1.48 | "
                "nvidia-cusparse-cu11==11.7.5.86 "
            ),
            "cuda12.3": (
                "nvidia-cuda-runtime-cu12==12.3.101 | "
                "nvidia-cudnn-cu12==9.0.0.312 | "
                "nvidia-cublas-cu12==12.3.4.1 | "
                "nvidia-cufft-cu12==11.2.1.3 | "
                "nvidia-curand-cu12==10.3.5.147 | "
                "nvidia-cusolver-cu12==11.6.1.9 | "
                "nvidia-cusparse-cu12==12.3.1.170 "
            ),
        }
    
    if cuda_major_version in PADDLE_CUDA_INSTALL_REQUIREMENTS:
        paddle_cuda_requires = PADDLE_CUDA_INSTALL_REQUIREMENTS[
                cuda_major_version
            ].split("|")
    else:
        paddle_cuda_requires = []

    if cuda_major_version == 'cuda11.8':
        index_url = "https://www.paddlepaddle.org.cn/packages/stable/cu118/"
    elif cuda_major_version == 'cuda12.3':
        index_url = "https://www.paddlepaddle.org.cn/packages/stable/cu123/"
    else:
        index_url = "https://www.paddlepaddle.org.cn/packages/stable/cpu/"

    original_directory = os.getcwd()
    os.chdir(package_path)
    with open(build_filename, 'w') as f:
        for item in paddle_cuda_requires:
            os.system(f'pip download --no-deps {item} -i {index_url}')
        os.system(f'pip install {name}=={paddle_version} --no-deps -i {index_url}')
    os.chdir(original_directory)

    with open(build_filename, 'w') as f:
        for item in paddle_cuda_requires:
            f.write(f"pip install {item} -f {package_path}\n")
        f.write(f"pip install {name}=={paddle_version} -f {package_path}\n")



def conda_build(paddle_version, var):
    python_version_lists = []
    sysstr = platform.system()
    if sysstr == "Linux":
        # cpu安装包编译
        name = 'paddlepaddle'
        for i in range(len(var.py_str)):
            packages_string = var.py_str[i] + "_cpu_many_linux"
            python_version = var.py_ver[var.py_str[i]]
            template_full(name, paddle_version, packages_string, python_version)
            gen_build_scripts(name, 'cpu', paddle_version)
            os.system("conda build .")

        # gpu安装包编译
        name = 'paddlepaddle-gpu'
        for i in range(len(var.py_str)):
            for cuda_str in var.cuda_info:
                packages_string = var.py_str[i] + "_gpu_" + cuda_str + "_many_linux"
                python_version = var.py_ver[var.py_str[i]]
                template_full(name, paddle_version, packages_string, python_version)
                gen_build_scripts(name, cuda_str, paddle_version)
                os.system("conda build .")

    elif sysstr == "Windows":
        # cpu安装包编译
        name = 'paddlepaddle'
        for i in range(len(var.py_str)):
            packages_string = var.py_str[i] + "_cpu_windows"
            python_version = var.py_ver[var.py_str[i]]
            template_full(name, paddle_version, packages_string, python_version)
            gen_build_scripts(name, 'cpu', paddle_version)
            os.system("conda build .")

        # gpu安装包编译
        name = 'paddlepaddle-gpu'
        for i in range(len(var.py_str)):
            for cuda_str in var.cuda_info:
                packages_string = var.py_str[i] + "_gpu_" + cuda_str + "_windows"
                python_version = var.py_ver[var.py_str[i]]
                template_full(name, paddle_version, packages_string, python_version)
                gen_build_scripts(name, cuda_str, paddle_version)
                os.system("conda build .")   
    elif sysstr == "Darwin":
        # cpu安装包编译
        name = 'paddlepaddle'
        for i in range(len(var.py_str)):
            packages_string = var.py_str[i] + "_mac"
            python_version = var.py_ver[var.py_str[i]]
            template_full(name, paddle_version, packages_string, python_version)
            gen_build_scripts(name, 'cpu', paddle_version)
            os.system("conda build .")


if __name__ == "__main__":
    args = parse_args()
    paddle_version = args.paddle_version
    var = ConstantVar()
    conda_build(paddle_version, var)
