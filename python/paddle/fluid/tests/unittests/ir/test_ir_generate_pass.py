#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
from paddle.static import InputSpec
from paddle.fluid import ir
import numpy as np


# 0: ewadd(X=mul(X=x, Y=w), Y=b) => fc(Input=x, W=w, Bias=b)
# 1: relu(X=ewadd(X=mul(X=x, Y=w), Y=b)) => fc(Input=x, W=w, Bias=b)
@ir.RegisterPass
def generate_fc_fuse():
    def create_pass_pair(with_relu):
        def pattern(x, w, b):
            mul = ir.PassDesc.OP.mul(X=x, Y=w)
            ewadd = ir.PassDesc.OP.elementwise_add(X=mul, Y=b)
            if with_relu:
                return ir.PassDesc.OP.relu(X=ewadd)
            else:
                return ewadd

        def replace(x, w, b):
            fc = ir.PassDesc.OP.fc
            fc.Attr("in_num_col_dims").ReusePattern(
                "mul", name="x_num_col_dims")
            if with_relu:
                fc.SetAttr("activation_type", "relu")
            return fc(Input=x, W=w, Bias=b)

        return pattern, replace

    return list(map(create_pass_pair, [True, False]))


# add(X=add(x, y), Y=z)z => add_n(X=[x, y, z])
@ir.RegisterPass
def generate_add_n():
    def pattern(x, y, z):
        return paddle.add(paddle.add(x, y), z)

    def replace(x, y, z):
        return paddle.add_n([x, y, z])

    return pattern, replace


# mul(x, y1), mul(x, y2) => slice(mul(x, concat(y1, y2)))
@ir.RegisterPass(input_specs={
    'x': InputSpec([1, 1]),
    'y1': InputSpec([1, 1]),
    'y2': InputSpec([1, 1])
})
def generate_combine_mul_v1():
    def pattern(x, y1, y2):
        mul1 = paddle.matmul(x, y1)
        mul2 = paddle.matmul(x, y2)
        return mul1, mul2

    def replace(x, y1, y2):
        concat_out = paddle.concat([y1, y2], axis=-1)
        mul_out = paddle.matmul(x, concat_out)
        out1 = paddle.slice(mul_out, axes=[1], starts=[0], ends=[1])
        out2 = paddle.slice(mul_out, axes=[1], starts=[1], ends=[2])
        return out1, out2

    return pattern, replace


@ir.RegisterPass
def generate_combine_mul_v2():
    def pattern(x, y1, y2):
        mul1 = ir.PassDesc.OP.matmul_v2(x, y1)
        mul2 = ir.PassDesc.OP.matmul_v2(x, y2)
        return mul1, mul2

    def replace(x, y1, y2):
        concat = ir.PassDesc.OP.concat(X=[y1, y2])
        matmul = ir.PassDesc.OP.matmul_v2(X=x, Y=concat)
        out1 = ir.PassDesc.OP.slice(Input=matmul)
        out2 = ir.PassDesc.OP.slice(Input=matmul)
        return out1, out2

    return pattern, replace


# reshape(reshape(x)) => x
@ir.RegisterPass(input_specs={'x': InputSpec([-1, 16, 16, 16])})
def generate_simplify_inference():
    def pattern(x):
        transpose = paddle.transpose(x, [0, 3, 1, 2])
        return paddle.transpose(transpose, [0, 3, 1, 2])

    return pattern, lambda x: x


def get_register_pass_helper(pass_pairs, input_specs=None):
    helper = ir.RegisterPassHelper()
    helper.SetPassPairs(pass_pairs)
    helper.SetInputSpecs(input_specs)
    return helper


def get_multi_pass_desc_from_str(s):
    multi_pass_desc = ir.pass_desc_pb2.MultiPassDesc()
    multi_pass_desc.ParseFromString(s)
    return multi_pass_desc


class TestGeneratePass(unittest.TestCase):
    def test_generate_fc_fuse(self):
        def _check_fc_fuse_pass(pass_desc, with_relu):
            if with_relu:
                pattern_op_num = 3  # relu, ewadd, mul
            else:
                pattern_op_num = 2  # ewadd, mul
            self.assertEqual(len(pass_desc.var_maps), 4)
            self.assertEqual(
                len(pass_desc.pattern.blocks[0].ops), pattern_op_num)
            self.assertEqual(len(pass_desc.attr_maps), 1)

        helper = get_register_pass_helper(generate_fc_fuse())
        s = helper.SerializeMultiPassDesc()
        multi_pass_desc = get_multi_pass_desc_from_str(s)
        self.assertEqual(len(multi_pass_desc.pass_descs), 2)
        _check_fc_fuse_pass(multi_pass_desc.pass_descs[0], True)
        _check_fc_fuse_pass(multi_pass_desc.pass_descs[1], False)

    def test_generate_add_n(self):
        helper = get_register_pass_helper([generate_add_n()])
        s = helper.SerializeMultiPassDesc()
        multi_pass_desc = get_multi_pass_desc_from_str(s)
        self.assertEqual(len(multi_pass_desc.pass_descs), 1)
        pass_desc = multi_pass_desc.pass_descs[0]
        self.assertEqual(len(pass_desc.var_maps), 4)
        self.assertEqual(len(pass_desc.attr_maps), 0)

    def test_generate_combine_mul_v1(self):
        input_specs = {
            'x': InputSpec([1, 1]),
            'y1': InputSpec([1, 1]),
            'y2': InputSpec([1, 1])
        }
        helper = get_register_pass_helper([generate_combine_mul_v1()],
                                          input_specs)
        s = helper.SerializeMultiPassDesc()
        multi_pass_desc = get_multi_pass_desc_from_str(s)
        self.assertEqual(len(multi_pass_desc.pass_descs), 1)
        pass_desc = multi_pass_desc.pass_descs[0]
        self.assertEqual(len(pass_desc.var_maps), 5)

    def test_generate_combine_mul_v2(self):
        helper = get_register_pass_helper([generate_combine_mul_v2()])
        s = helper.SerializeMultiPassDesc()
        multi_pass_desc = get_multi_pass_desc_from_str(s)
        self.assertEqual(len(multi_pass_desc.pass_descs), 1)
        pass_desc = multi_pass_desc.pass_descs[0]
        self.assertEqual(len(pass_desc.var_maps), 5)

    def test_generate_simplify_inference(self):
        input_specs = {'x': InputSpec([-1, 16, 16, 16])}
        helper = get_register_pass_helper([generate_simplify_inference()],
                                          input_specs)
        s = helper.SerializeMultiPassDesc()
        multi_pass_desc = get_multi_pass_desc_from_str(s)
        self.assertEqual(len(multi_pass_desc.pass_descs), 1)
        pass_desc = multi_pass_desc.pass_descs[0]
        self.assertEqual(len(pass_desc.var_maps), 2)
