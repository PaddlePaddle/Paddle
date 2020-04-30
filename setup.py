# -*- coding: UTF-8 -*-
################################################################################
#
#   Copyright (c) 2020  Baidu.com, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
################################################################################
"""
Setup script.
Authors: zhouxiangyang(zhouxiangyang@baidu.com)
Date:    2020/2/4 00:00:01
"""
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="hapi",
    version="0.0.1",
    author="PaddlePaddle",
    author_email="zhouxiangyang@baidu.com",
    description="A Paddle High-level API that supports both static and dynamic execution modes (still under development)",
    url="https://github.com/PaddlePaddle/hapi",
    packages=[
        'hapi',
        'hapi.datasets',
        'hapi.text',
        'hapi.text.tokenizer',
        'hapi.text.bert',
        'hapi.text.bert.utils',
        'hapi.vision',
        'hapi.vision.models',
        'hapi.vision.transforms',
    ],
    package_dir={
        'hapi': './hapi',
        'hapi.datasets': './hapi/datasets',
        'hapi.text': './hapi/text',
        'hapi.text.tokenizer': './hapi/text/tokenizer',
        'hapi.text.bert': './hapi/text/bert',
        'hapi.text.bert.utils': './hapi/text/bert/utils',
        'hapi.vision': './hapi/vision',
        'hapi.vision.models': './hapi/vision/models',
        'hapi.vision.transforms': './hapi/vision/transforms',
    },
    platforms="any",
    license='Apache 2.0',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ], )
