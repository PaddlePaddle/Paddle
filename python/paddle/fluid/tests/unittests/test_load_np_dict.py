# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.initializer import Constant, NormalInitializer
import numpy as np


class Model(fluid.Layer):
    def __init__(self, name_scope):
        super(Model, self).__init__(name_scope)
        self.add_parameter(
            "w_int32",
            self.create_parameter(
                attr=fluid.ParamAttr(name='w_int32'),
                shape=[10, 10],
                dtype=core.VarDesc.VarType.INT32,
                is_bias=False,
                default_initializer=Constant(1)))
        self.add_parameter(
            "w_int64",
            self.create_parameter(
                attr=fluid.ParamAttr(name='w_int64'),
                shape=[10, 10],
                dtype=core.VarDesc.VarType.INT64,
                is_bias=False,
                default_initializer=Constant(2)))
        self.add_parameter(
            "w_fp16",
            self.create_parameter(
                attr=fluid.ParamAttr(name='w_fp16'),
                shape=[10, 10],
                dtype=core.VarDesc.VarType.FP16,
                is_bias=False))
        self.add_parameter(
            "w_fp32",
            self.create_parameter(
                attr=fluid.ParamAttr(name='w_fp32'),
                shape=[10, 10],
                dtype=core.VarDesc.VarType.FP32,
                is_bias=False))
        self.add_parameter(
            "w_fp64",
            self.create_parameter(
                attr=fluid.ParamAttr(name='w_fp64'),
                shape=[10, 10],
                dtype=core.VarDesc.VarType.FP64,
                is_bias=False))
        self.add_parameter(
            "w_bool",
            self.create_parameter(
                attr=fluid.ParamAttr(name='w_bool'),
                shape=[10, 10],
                dtype=core.VarDesc.VarType.BOOL,
                is_bias=False,
                default_initializer=Constant(False)))


class TestLoadNpDict(unittest.TestCase):
    def test_load_np_dict(self):
        with fluid.dygraph.guard():
            m = Model("model")
            fluid.save_dygraph(m.state_dict(), "dy")
            np_dic = core._load_np_dict("dy.pdparams")
            para_names = [
                'model/Model_0.w_int32', 'model/Model_0.w_int64',
                'model/Model_0.w_fp16', 'model/Model_0.w_fp32',
                'model/Model_0.w_fp64', 'model/Model_0.w_bool'
            ]
            np_types = [
                np.int32, np.int64, np.float16, np.float32, np.float64, np.bool
            ]
            for i, para_name in enumerate(para_names):
                self.assertTrue(np_dic[para_name].dtype == np_types[i])


if __name__ == '__main__':
    unittest.main()
