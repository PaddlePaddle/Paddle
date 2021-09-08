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
from paddle.fluid.framework import IrGraph
import numpy as np


# 0: ewadd(mul(x, w), b) => fc(x, w, b)
# 1: relu(ewadd(mul(x, w), b)) => fc(x, w, b)
@ir.RegisterPass("generate_fc_fuse")
def generate_fc_fuse():
    def create_pass_pair(with_relu):
        #pattern
        def pattern(x, w, b):
            mul = ir.PassDesc.OP.mul(X=x, Y=w)
            ewadd = ir.PassDesc.OP.elementwise_add(X=mul, Y=b)
            if with_relu:
                return ir.PassDesc.OP.relu(X=ewadd)
            else:
                return ewadd

        #replace
        def replace(x, w, b):
            fc = ir.PassDesc.OP.fc
            fc.Attr("in_num_col_dims").ReusePattern(
                "mul", name="x_num_col_dims")
            if with_relu:
                fc.SetAttr("activation_type", "relu")
            return fc(Input=x, W=w, Bias=b)

        return pattern, replace

    return list(map(create_pass_pair, [True, False]))


# add(add(x, y), z) => add_n([x, y, z])
@ir.RegisterPass("generate_add_n")
def generate_add_n():
    # pattern
    def pattern(x, y, z):
        return paddle.add(paddle.add(x, y), z)

    # replace
    def replace(x, y, z):
        return paddle.add_n([x, y, z])

    return pattern, replace


# [feature] Use graph as input, not subgraph pair.
#@ir.RegisterPass("py_change_graph")
def py_change_graph(graph: IrGraph) -> IrGraph:
    return graph


# mul(x, y1), mul(x, y2) => slice(mul(x, concat(y1, y2)))
@ir.RegisterPass("generate_combine_mul")
def generate_combine_mul():
    # pattern
    def pattern(x, y1, y2):
        mul1 = paddle.matmul(x, y1)
        mul2 = paddle.matmul(x, y2)
        return mul1, mul2

    # replace
    def replace(x, y1, y2):
        concat_out = paddle.concat([y1, y2])
        mul_out = paddle.matmul(x, concat_out)
        out1 = paddle.slice(mul_out, axes=[0, 1], starts=[0, 1], ends=[2, 10])
        out2 = paddle.slice(mul_out, axes=[0, 1], starts=[0, 1], ends=[2, 10])
        return out1, out2

    return pattern, replace


# OP-ANY(const...) => Evaluate
@ir.RegisterPass("generate_constant_fold")
def generate_constant_fold():
    # pattern
    def pattern(*args):
        return ir.PassDesc.OP_ANY(ir.PassDesc.OP.const(X=args))

    # replace - Evaluate

    return pattern, pattern  #ir.PassDesc.MethodType.kEvaluate


# OP-X(reshape(reshape)) => OP-X
@ir.RegisterPass("generate_simplify_inference")
def generate_simplify_inference():
    # pattern
    def pattern(x):
        reshape1_out = ir.PassDesc.OP.reshape(X=x)
        reshape2_out = ir.PassDesc.OP.reshape(X=reshape1_out)
        return ir.PassDesc.OP_ANY(reshape2_out)

    # replace
    def replace(x):
        return ir.PassDesc.OP_ANY(x)

    return pattern, replace


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
        pass_pairs = [generate_add_n()]
        helper = get_register_pass_helper(pass_pairs)
        s = helper.SerializeMultiPassDesc()
        multi_pass_desc = get_multi_pass_desc_from_str(s)
        self.assertEqual(len(multi_pass_desc.pass_descs), 1)
        pass_desc = multi_pass_desc.pass_descs[0]
        self.assertEqual(len(pass_desc.var_maps), 4)
        self.assertEqual(len(pass_desc.attr_maps), 0)

    def test_py_change_graph(self):
        pass

    def test_generate_combine_mul(self):
        helper = get_register_pass_helper([generate_combine_mul()])
        s = helper.SerializeMultiPassDesc()
        multi_pass_desc = get_multi_pass_desc_from_str(s)
        self.assertEqual(len(multi_pass_desc.pass_descs), 1)
        pass_desc = multi_pass_desc.pass_descs[0]
        self.assertEqual(len(pass_desc.var_maps), 5)

    @unittest.skip
    def test_generate_constant_fold(self):
        helper = get_register_pass_helper([generate_constant_fold()])
        helper.SerializeMultiPassDesc()

    @unittest.skip
    def test_generate_simplify_inference(self):
        helper = get_register_pass_helper([generate_simplify_inference()])
        helper.SerializeMultiPassDesc()
