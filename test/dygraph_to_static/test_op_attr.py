# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
    test_legacy_and_pt_and_pir,
)

import paddle
from paddle.framework import use_pir_api
from paddle.static import InputSpec


def walk(block, fn):
    fn(block)
    for op in block.ops:
        for sub_block in op.blocks():
            walk(sub_block, fn)


def run_on_each_op(block, fn):
    def check_block(block):
        for op in block.ops:
            fn(op)

    walk(block, check_block)


class MySub(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return paddle.subtract(x, y, name)


def create_net_with_op_attr_class():
    class NetWithOpAttr(paddle.nn.Layer):
        def __init__(self, in_num, out_num):
            super().__init__()

            self.linear = paddle.nn.Linear(in_num, out_num)
            self.bn = paddle.nn.BatchNorm(out_num)
            self.sub = MySub()

        def forward(self, x):
            out = self.linear(x)
            out = self.sub(out, x)
            out = self.bn(out)
            return out

        @paddle.jit.to_static(input_spec=[InputSpec([10, 16])], full_graph=True)
        def with_cond(self, x):
            if paddle.mean(x) > 0.0:
                out = self.linear(x)
            else:
                out = self.sub(x, x)
            out = self.bn(out)
            return out

    return NetWithOpAttr


class CheckOpAttr(Dy2StTestBase):
    def setUp(self):
        self.in_num = 16
        self.out_num = 16
        self.x = paddle.randn([10, self.in_num])

    def expected_results(self):
        fc_attrs = {
            "int_val": 10,
            "int_vals": [10, 20],
            "float_val": 3.8,
            "float_vals": [3.8, -0.2],
        }
        bn_attrs = {"bool_val": True, "bool_vals": [True, False]}
        sub_attrs = {"int_vals": [10, 20], "bool_vals": [True, False]}

        if use_pir_api():
            infos = {
                'pd_op.matmul': fc_attrs,
                'pd_op.add': fc_attrs,
                'pd_op.batch_norm_': bn_attrs,
                'pd_op.subtract': sub_attrs,
            }
        else:
            infos = {
                'matmul': fc_attrs,
                'elementwise_add': fc_attrs,
                'batch_norm': bn_attrs,
                'tanh': bn_attrs,
                'elementwise_sub': sub_attrs,
            }
        return fc_attrs, bn_attrs, sub_attrs, infos

    @test_ast_only
    @test_legacy_and_pt_and_pir
    def test_set_op_attrs(self):
        fc_attrs, bn_attrs, sub_attrs, _ = self.expected_results()
        net = create_net_with_op_attr_class()(self.in_num, self.out_num)
        # set attrs
        net.linear._set_op_attrs(fc_attrs)
        net.bn._set_op_attrs({"bool_val": False})  # test overwrite behavior
        net.bn._set_op_attrs(bn_attrs)
        net.sub._set_op_attrs(sub_attrs)
        # assert hooks exist.
        self.assertEqual(len(net.linear._forward_pre_hooks), 1)
        self.assertEqual(len(net.linear._forward_post_hooks), 1)
        # to_static
        net = paddle.jit.to_static(
            net, input_spec=[InputSpec.from_tensor(self.x)]
        )

        # assert attrs have be set.
        self.check_op_attrs(net.forward.concrete_program.main_program)

        # assert hooks have be clean.
        self.assertEqual(len(net.linear._forward_pre_hooks), 0)
        self.assertEqual(len(net.linear._forward_post_hooks), 0)

    def check_op_attrs(self, main_program):
        _, _, _, infos = self.expected_results()
        if not use_pir_api():
            for cur_block in main_program.blocks:
                ops = cur_block.ops
                for op in ops:
                    if op.type not in infos:
                        continue
                    for attr_name, expect_vals in infos[op.type].items():
                        op_vals = op.desc.attr(attr_name)
                        if not isinstance(expect_vals, list):
                            expect_vals = [expect_vals]
                            op_vals = [op_vals]

                        for op_val, expect_val in zip(op_vals, expect_vals):
                            if isinstance(op_val, float):
                                # C++ vs python: 3.799999952316284 ~= 3.8
                                self.assertAlmostEqual(op_val, expect_val)
                            else:
                                self.assertEqual(op_val, expect_val)
            return
        global_block = main_program.global_block()

        def check_op(op):
            if op.name() not in infos:
                return
            for attr_name, expect_vals in infos[op.name()].items():
                op_vals = op.attrs()[attr_name]
                if not isinstance(expect_vals, list):
                    expect_vals = [expect_vals]
                    op_vals = [op_vals]

                for op_val, expect_val in zip(op_vals, expect_vals):
                    if isinstance(op_val, float):
                        # C++ vs python: 3.799999952316284 ~= 3.8
                        self.assertAlmostEqual(op_val, expect_val)
                    else:
                        self.assertEqual(op_val, expect_val)

        run_on_each_op(global_block, check_op)

    @test_ast_only
    @test_legacy_and_pt_and_pir
    def test_set_op_attrs_with_sub_block(self):
        fc_attrs, bn_attrs, sub_attrs, _ = self.expected_results()
        net = create_net_with_op_attr_class()(self.in_num, self.out_num)
        # set attrs
        net.linear._set_op_attrs(
            {"int_vals": [0, 0]}
        )  # test overwrite behavior
        net.linear._set_op_attrs(fc_attrs)
        net.bn._set_op_attrs(bn_attrs)
        net.sub._set_op_attrs(sub_attrs)
        # assert hooks exist.
        self.assertEqual(len(net.linear._forward_pre_hooks), 1)
        self.assertEqual(len(net.linear._forward_post_hooks), 1)

        # assert attrs have be set.
        self.check_op_attrs(net.with_cond.concrete_program.main_program)

        # assert hooks have be clean.
        self.assertEqual(len(net.linear._forward_pre_hooks), 0)
        self.assertEqual(len(net.linear._forward_post_hooks), 0)

    @test_legacy_and_pt_and_pir
    def test_type_error(self):
        fc_attrs, _, _, _ = self.expected_results()
        net = create_net_with_op_attr_class()(self.in_num, self.out_num)
        # attrs should be dict
        with self.assertRaises(TypeError):
            net.linear._set_op_attrs([fc_attrs])


if __name__ == '__main__':
    unittest.main()
