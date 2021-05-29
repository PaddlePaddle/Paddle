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

#
import platform
from sys import argv
import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser("conda build for paddlepaddle version")
    parser.add_argument(
        "--paddle_version",
        type=str,
        required=True,
        help="paddle version for conda build.")
    args = parser.parse_args()

    return args


class ConstantVar:
    def __init__(self):
        self.build = r"""
build:
  number: '0'
  string: """

        self.requirement_build = r"""
requirements:
  build:
    - numpy>=1.13
    - cython
    - setuptools
"""

        self.requirement_run = r"""
  run:
    - requests>=2.20.0
    - numpy>=1.13
    - protobuf>=3.1.0
    - gast==0.3.3
    - Pillow
    - six
    - decorator
    - astor
"""

        self.requirement_run_windows = r"""
  run:
    - requests>=2.20.0
    - numpy>=1.13
    - protobuf>=3.1.0
    - gast==0.3.3
    - Pillow
    - six
    - decorator
    - astor
"""
        self.test = r"""
test:
  import:
    paddle
"""

        self.about = r"""
about:
  home: http://www.paddlepaddle.org/
  license: APACHE 2.0
  license_family: APACHE
  summary: an easy-to-use, efficient, flexible and scalable deep learning platform
"""

        self.build_const = r"""
"""

        self.blt_const = r""" 
"""

        self.python36 = r"    - python>=3.6, <3.7"
        self.python37 = r"    - python>=3.7, <3.8"
        self.python38 = r"    - python>=3.8, <3.9"
        self.python39 = r"    - python>=3.9, <3.10"

        self.python_version = [
            self.python36, self.python37, self.python38, self.python39
        ]

        self.cuda101 = r"""
    - cudatoolkit>=10.1, <10.2
    - cudnn>=7.6, <7.7
    """
        self.cuda102 = r"""
    - cudatoolkit>=10.2, <10.3
    - cudnn>=7.6, <7.7
    """
        self.cuda112 = r"""
    - cudatoolkit>=11.2, <11.3
    - cudnn>=8.1, <8.2
    """

        self.cuda_info = [(self.cuda101, "cuda10.1", ".post101"),
                          (self.cuda102, "cuda10.2", ""),
                          (self.cuda112, "cuda11.2", ".post112")]
        self.py_str = ["py36", "py37", "py38", "py39"]
        self.pip_end = ".whl --no-deps"
        self.pip_prefix_linux = "pip install /package/paddlepaddle"
        self.pip_prefix_windows = r"pip install C:\package\paddlepaddle"
        self.pip_gpu = "_gpu-"
        self.pip_cpu = "-"
        self.mac_pip = [
            "-cp36-cp36m-macosx_10_6_intel", "-cp37-cp37m-macosx_10_6_intel",
            "-cp38-cp38-macosx_10_14_x86_64", "-cp39-cp39-macosx_10_14_x86_64"
        ]
        self.linux_pip = [
            "-cp36-cp36m-linux_x86_64", "-cp37-cp37m-linux_x86_64",
            "-cp38-cp38-linux_x86_64", "-cp39-cp39-linux_x86_64"
        ]
        self.windows_pip = [
            "-cp36-cp36m-win_amd64", "-cp37-cp37m-win_amd64",
            "-cp38-cp38-win_amd64", "-cp39-cp39-win_amd64"
        ]


def meta_build_mac(var, python_str, paddle_version, build_var, build_name_str):
    package_str = """
package:
  name: paddlepaddle
  version: """ + paddle_version
    requirement = var.requirement_build + python_str + var.requirement_run + python_str
    meta_build = var.build + build_name_str
    meta_str = package_str + meta_build + requirement + var.test + var.about
    build_str = var.build_const + build_var

    meta_filename = "meta.yaml"
    build_filename = "build.sh"
    with open(meta_filename, 'w') as f:
        f.write(meta_str)
    with open(build_filename, 'w') as f:
        f.write(build_str)


def meta_build_linux(var,
                     python_str,
                     paddle_version,
                     build_var,
                     build_name_str,
                     cuda_str=None):
    if cuda_str == None:
        package_str = """
package:
  name: paddlepaddle
  version: """ + paddle_version
    else:
        package_str = """
package:
  name: paddlepaddle-gpu
  version: """ + paddle_version
    requirement = var.requirement_build + python_str + var.requirement_run + python_str
    meta_build = var.build + build_name_str
    meta_str = package_str + meta_build + requirement
    if not (cuda_str == None):
        meta_str = meta_str + cuda_str
    meta_str = meta_str + var.test + var.about

    build_str = var.build_const + build_var

    meta_filename = "meta.yaml"
    build_filename = "build.sh"
    with open(meta_filename, 'w') as f:
        f.write(meta_str)
    with open(build_filename, 'w') as f:
        f.write(build_str)


def meta_build_windows(var,
                       python_str,
                       paddle_version,
                       blt_var,
                       build_name_str,
                       cuda_str=None):
    if cuda_str == None:
        package_str = """
package:
  name: paddlepaddle
  version: """ + paddle_version
    else:
        package_str = """
package:
  name: paddlepaddle-gpu
  version: """ + paddle_version

    requirement = var.requirement_build + python_str + var.requirement_run_windows + python_str
    meta_build = var.build + build_name_str
    meta_str = package_str + meta_build + requirement

    if not (cuda_str == None):
        meta_str = meta_str + cuda_str

    blt_str = var.blt_const + blt_var

    meta_str = meta_str + var.test + var.about
    meta_filename = "meta.yaml"
    build_filename = "bld.bat"
    with open(meta_filename, 'w') as f:
        f.write(meta_str)
    with open(build_filename, 'w') as f:
        f.write(blt_str)


def conda_build(paddle_version, var):
    sysstr = platform.system()
    if (sysstr == "Windows"):
        os.system("mkdir paddle")
        os.chdir(r"./paddle")
        for i in range(len(var.python_version)):
            blt_var = var.pip_prefix_windows + var.pip_cpu + paddle_version + var.windows_pip[
                i] + var.pip_end
            name = var.py_str[i] + "_cpu_windows"
            python_str = var.python_version[i]
            meta_build_windows(var, python_str, paddle_version, blt_var, name)
            os.system("conda build .")

        for i in range(len(var.python_version)):
            for cuda_str in var.cuda_info:
                post = cuda_str[2]
                blt_var = var.pip_prefix_windows + var.pip_gpu + paddle_version + post + var.windows_pip[
                    i] + var.pip_end
                name = var.py_str[i] + "_gpu_" + cuda_str[1] + "_windows"
                cuda_cudnn_str = cuda_str[0]
                python_str = var.python_version[i]
                meta_build_windows(var, python_str, paddle_version, blt_var,
                                   name, cuda_cudnn_str)
                os.system("conda build .")

    elif (sysstr == "Linux"):
        os.system("mkdir paddle")
        os.chdir(r"./paddle")
        for i in range(len(var.python_version)):
            build_var = var.pip_prefix_linux + var.pip_cpu + paddle_version + var.linux_pip[
                i] + var.pip_end
            name = var.py_str[i] + "_cpu_many_linux"
            python_str = var.python_version[i]
            meta_build_linux(var, python_str, paddle_version, build_var, name)
            os.system("conda build .")

        for i in range(len(var.python_version)):
            for cuda_str in var.cuda_info:
                post = cuda_str[2]
                build_var = var.pip_prefix_linux + var.pip_gpu + paddle_version + post + var.linux_pip[
                    i] + var.pip_end
                name = var.py_str[i] + "_gpu_" + cuda_str[1] + "_many_linux"
                cuda_cudnn_str = cuda_str[0]
                python_str = var.python_version[i]
                meta_build_linux(var, python_str, paddle_version, build_var,
                                 name, cuda_cudnn_str)
                os.system("conda build .")

        os.system("cd ..")

    elif (sysstr == "Darwin"):
        os.system("mkdir paddle")
        os.chdir(r"./paddle")
        for i in range(len(var.python_version)):
            build_var = var.pip_prefix_linux + var.pip_cpu + paddle_version + var.mac_pip[
                i] + var.pip_end
            name = var.py_str[i] + "_mac"
            python_str = var.python_version[i]
            meta_build_mac(var, python_str, paddle_version, build_var, name)
            os.system("conda build .")

        os.system("cd ..")


if __name__ == "__main__":
    args = parse_args()
    paddle_version = args.paddle_version
    var = ConstantVar()
    conda_build(paddle_version, var)
