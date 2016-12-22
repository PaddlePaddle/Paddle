#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
Print model parameters in last model

Usage:
    python evaluate_model.py
"""
import numpy as np
import os


def load(file_name):
    with open(file_name, 'rb') as f:
        f.read(16)  # skip header for float type.
        return np.fromfile(f, dtype=np.float32)


def main():
    print 'w=%.6f, b=%.6f from pass 29' % (load('output/pass-00029/w'),
                                           load('output/pass-00029/b'))


if __name__ == '__main__':
    main()
