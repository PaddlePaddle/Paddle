# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy

os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = 'true'
os.environ['FLAGS_print_ir'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_use_cinn'] = '1'
os.environ['FLAGS_deny_cinn_ops'] = 'slice;'

import paddle

build_strategy = paddle.static.BuildStrategy()
build_strategy.build_cinn_pass = True


def init():
    var_52 = paddle.rand([4000, 512])
    var_54 = paddle.rand([4000, 512])
    var_38 = paddle.rand([4000, 512])
    var_17 = paddle.rand([512])
    var_57 = paddle.rand([4000, 512])
    var_53 = paddle.rand([4000, 512])
    var_58 = paddle.rand([4000, 512])
    var_56 = paddle.rand([4000, 512])
    var_55 = paddle.rand([4000, 512])
    return (
        var_52,
        var_54,
        var_38,
        var_17,
        var_57,
        var_53,
        var_58,
        var_56,
        var_55,
    )


def func(
    var_52, var_54, var_38, var_17, var_57, var_53, var_58, var_56, var_55
):
    var_86 = paddle.broadcast_to(var_17, [4000, 512])
    var_87 = var_38 + var_86
    var_88 = var_87
    var_89 = paddle.full(shape=[4000, 512], dtype='float32', fill_value=-1)
    var_90 = var_89 * var_87
    var_91 = paddle.exp(var_90)
    var_92 = paddle.full(shape=[4000, 512], dtype='float32', fill_value=1)
    var_93 = var_91 + var_92
    var_94 = var_87 / var_93
    var_95 = var_87 * -1.0 + 0.0
    var_96 = paddle.exp(var_95)
    var_97 = var_96
    var_98 = paddle.full(shape=[4000, 512], dtype='float32', fill_value=1)
    var_99 = var_98 + var_96
    var_100 = var_99
    var_101 = var_52 / var_99
    var_102 = var_87 * -1.0 + 0.0
    var_103 = paddle.exp(var_102)
    var_104 = var_103
    var_105 = paddle.full(shape=[4000, 512], dtype='float32', fill_value=1)
    var_106 = var_105 + var_103
    var_107 = var_106
    var_108 = var_53 / var_106
    var_109 = var_87 * -1.0 + 0.0
    var_110 = paddle.exp(var_109)
    var_111 = var_110
    var_112 = paddle.full(shape=[4000, 512], dtype='float32', fill_value=1)
    var_113 = var_112 + var_110
    var_114 = var_113
    var_115 = var_54 / var_113
    var_116 = var_99 * var_99
    var_117 = var_87 * -1.0 + 0.0
    var_118 = paddle.exp(var_117)
    var_119 = var_118
    var_120 = paddle.full(shape=[4000, 512], dtype='float32', fill_value=1)
    var_121 = var_120 + var_118
    var_122 = var_121
    var_123 = var_55 / var_121
    var_124 = var_99 * var_99
    var_125 = var_87 * -1.0 + 0.0
    var_126 = paddle.exp(var_125)
    var_127 = var_126
    var_128 = paddle.full(shape=[4000, 512], dtype='float32', fill_value=1)
    var_129 = var_128 + var_126
    var_130 = var_129
    var_131 = var_56 / var_129
    var_132 = var_106 * var_106
    var_133 = var_87 * -1.0 + 0.0
    var_134 = paddle.exp(var_133)
    var_135 = var_134
    var_136 = paddle.full(shape=[4000, 512], dtype='float32', fill_value=1)
    var_137 = var_136 + var_134
    var_138 = var_137
    var_139 = var_57 / var_137
    var_140 = var_106 * var_106
    var_141 = var_87 * -1.0 + 0.0
    var_142 = paddle.exp(var_141)
    var_143 = var_142
    var_144 = paddle.full(shape=[4000, 512], dtype='float32', fill_value=1)
    var_145 = var_144 + var_142
    var_146 = var_145
    var_147 = var_58 / var_145

    return (
        var_88,
        var_94,
        var_97,
        var_100,
        var_101,
        var_104,
        var_107,
        var_108,
        var_111,
        var_114,
        var_115,
        var_116,
        var_119,
        var_122,
        var_123,
        var_124,
        var_127,
        var_130,
        var_131,
        var_132,
        var_135,
        var_138,
        var_139,
        var_140,
        var_143,
        var_146,
        var_147,
    )


class TestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def compare_result(self, dy_compute, data_init):
        static_compute = paddle.jit.to_static(
            full_graph=True, build_strategy=build_strategy
        )(dy_compute)
        inputs = data_init()
        dy_out = dy_compute(*inputs)
        st_out = static_compute(*inputs)
        numpy.testing.assert_allclose(dy_out, st_out, atol=1e-5, rtol=1e-6)

    def test_case(self):
        self.compare_result(func, init)


if __name__ == "__main__":
    unittest.main()
