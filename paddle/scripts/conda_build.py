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
    parser.add_argument(
        "--only_download",
        type=str,
        default=None,
        help="requirement download",
    )
    args = parser.parse_args()

    return args


class ConstantVar:
    def __init__(self):
        self.py_str = ["py38", "py39", "py310", "py311", "py312"]
        self.pip_ver = ["3.8", "3.9", "3.10", "3.11", "3.12"]
        self.cuda_info = ["cuda11.8", "cuda12.3"]
        self.py_ver = {
            "py38":"python>=3.8, <3.9",
            "py39":"python>=3.9, <3.10",
            "py310":"python>=3.10, <3.11",
            "py311":"python>=3.11, <3.12",
            "py312":"python>=3.12, <3.13"
        }


def template_full(name, version, packages_string, python_version, cuda_str):
    # 读取模板文件
    with open('conda_build_template.yaml', 'r') as template_file:
        template_content = template_file.read()

    paddlepaddle_cuda=''
    if cuda_str == "cuda11.8":
        paddlepaddle_cuda = '- paddlepaddle-cuda>=11.8,<12.0'
    elif cuda_str == "cuda12.3":
        paddlepaddle_cuda = '- paddlepaddle-cuda>=12.3,<12.4'

    # 使用字典填充模板
    data = {
        'name': name,
        'version': version,
        'packages_string': packages_string,
        'python_version':python_version,
        'paddlepaddle_cuda':paddlepaddle_cuda
    }

    # 使用 str.format 进行占位符替换
    filled_content = template_content.format(**data)

    # 将填充后的内容解析为YAML
    filled_yaml = yaml.safe_load(filled_content)

    if os.path.exists('meta.yaml'):
        os.remove('meta.yaml')
    # 将填充后的内容写入新的YAML文件
    with open('meta.yaml', 'w') as new_file:
        yaml.safe_dump(filled_yaml, new_file, default_flow_style=False, sort_keys=False)


def gen_build_scripts(name, cuda_major_version, paddle_version, only_download=None):
    sysstr = platform.system()
    if sysstr == "Linux":
        build_filename = "build.sh"
    elif platform.system() == 'Windows':
        build_filename = "bld.bat"

    if cuda_major_version == 'cuda11.8':
        index_url = "https://www.paddlepaddle.org.cn/packages/stable/cu118/"
    elif cuda_major_version == 'cuda12.3':
        index_url = "https://www.paddlepaddle.org.cn/packages/stable/cu123/"
    else:
        index_url = "https://www.paddlepaddle.org.cn/packages/stable/cpu/"

    if only_download:
        original_directory = os.getcwd()
        cur_package_path = os.path.join(package_path, cuda_major_version)
        os.makedirs(cur_package_path, exist_ok=True)
        os.chdir(cur_package_path)
        os.system(f'pip download {name}=={paddle_version} --no-deps -i {index_url}')
        os.chdir(original_directory)
    else:
        cur_package_path = os.path.join(package_path, cuda_major_version)
        if os.path.exists(build_filename):
            os.remove(build_filename)
        with open(build_filename, 'w') as f:
            f.write(f"pip install {name}=={paddle_version} -f {cur_package_path}\n")


def requirement_download(paddle_version, var):
    gen_build_scripts('paddlepaddle', 'cpu', paddle_version, only_download=True)
    for cuda_str in var.cuda_info:
        gen_build_scripts('paddlepaddle-gpu', cuda_str, paddle_version, only_download=True)



def conda_build(paddle_version, var):
    python_version_lists = []
    sysstr = platform.system()
    if sysstr == "Linux":
        # cpu安装包编译
        name = 'paddlepaddle'
        for i in range(len(var.py_str)):
            packages_string = var.py_str[i] + "_cpu_many_linux"
            python_version = var.py_ver[var.py_str[i]]
            template_full(name, paddle_version, packages_string, python_version, 'cpu')
            gen_build_scripts(name, 'cpu', paddle_version)
            os.system("conda build .")

        # gpu安装包编译
        name = 'paddlepaddle-gpu'
        for i in range(len(var.py_str)):
            for cuda_str in var.cuda_info:
                packages_string = var.py_str[i] + "_gpu_" + cuda_str + "_many_linux"
                python_version = var.py_ver[var.py_str[i]]
                template_full(name, paddle_version, packages_string, python_version, cuda_str)
                gen_build_scripts(name, cuda_str, paddle_version)
                os.system("conda build .")

    elif sysstr == "Windows":
        # cpu安装包编译
        name = 'paddlepaddle'
        for i in range(len(var.py_str)):
            packages_string = var.py_str[i] + "_cpu_windows"
            python_version = var.py_ver[var.py_str[i]]
            template_full(name, paddle_version, packages_string, python_version, cuda_str)
            gen_build_scripts(name, 'cpu', paddle_version)
            os.system("conda build .")

        # gpu安装包编译
        name = 'paddlepaddle-gpu'
        for i in range(len(var.py_str)):
            for cuda_str in var.cuda_info:
                packages_string = var.py_str[i] + "_gpu_" + cuda_str + "_windows"
                python_version = var.py_ver[var.py_str[i]]
                template_full(name, paddle_version, packages_string, python_version, cuda_str)
                gen_build_scripts(name, cuda_str, paddle_version)
                os.system("conda build .")   
    elif sysstr == "Darwin":
        # cpu安装包编译
        name = 'paddlepaddle'
        for i in range(len(var.py_str)):
            packages_string = var.py_str[i] + "_mac"
            python_version = var.py_ver[var.py_str[i]]
            template_full(name, paddle_version, packages_string, python_version, cuda_str)
            gen_build_scripts(name, 'cpu', paddle_version)
            os.system("conda build .")


if __name__ == "__main__":
    args = parse_args()
    paddle_version = args.paddle_version
    var = ConstantVar()
    if args.only_download is not None:
        requirement_download(paddle_version,var)
    else:
        conda_build(paddle_version, var)
