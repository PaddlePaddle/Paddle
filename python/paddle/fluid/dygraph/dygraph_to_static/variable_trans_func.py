#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import gast

import paddle.fluid as fluid
import paddle.fluid.layers as layers


def to_static_variable_gast_node(name):
    func_code = "%s = fluid.dygraph.dygraph_to_static.to_static_variable(%s)".format(
        name, name)
    return gast.parse(func_code)


def create_static_variable_gast_node(name):
    func_code = "%s = fluid.layers.data(name='%s', shape=[-1], dtype='float32')".format(
        name, name)
    return gast.parse(func_code)


def to_static_variable(x):
    '''
    Translate a Python variable to PaddlePaddle static graph variable
    '''
    if isinstance(x, bool):
        return layers.fill_constant(shape=[1], dtype='bool', value=x)
    if isinstance(x, int):
        return layers.fill_constant(shape=[1], dtype='int64', value=x)
    if isinstance(x, float):
        return layers.fill_constant(shape=[1], dtype='float64', value=x)
    return x
