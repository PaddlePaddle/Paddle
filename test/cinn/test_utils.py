#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

import logging
import unittest

import numpy as np

from paddle import cinn
from paddle.cinn import common, framework, ir, lang, runtime


class SingleOpTester(unittest.TestCase):
    '''
    A unittest framework for testing a single operator.

    Two methods one should override for each Operator's unittest

    1. create_target_data
    2. test_op
    '''

    def setUp(self):
        np.random.seed(0)
        self.counter = 0
        self.target = common.DefaultHostTarget()

    def create_target_data(self, inputs_data, attrs):
        '''
        create the target of the operator's execution output.
        '''
        raise NotImplementedError

    def test_op(self):
        '''
        USER API

        The real use case should implement this method!
        '''
        pass

    def to_test_op(
        self,
        input_shapes,
        output_shapes,
        op_name,
        attrs,
        out_index=None,
        do_infer_shape=False,
    ):
        '''
        Test the operator.
        '''
        self.compiler = cinn.Compiler.create(self.target)
        inputs = []
        inputs_data = []

        for i_shape in input_shapes:
            expr_shape = []
            inputs_data.append(
                np.around(np.random.random(i_shape).astype("float32"), 3)
            )

            for dim_shape in i_shape:
                expr_shape.append(ir.Expr(dim_shape))

            inputs.append(
                lang.Placeholder(
                    "float32", self.__gen_var_name(), expr_shape
                ).to_tensor()
            )

        args = []
        temp_inputs = []
        alignment = 0
        if self.target.arch.IsX86Arch():
            alignment = 32
        for in_data in inputs_data:
            temp_inputs.append(
                runtime.cinn_buffer_t(
                    in_data, runtime.cinn_x86_device, alignment
                )
            )
        for in_data in temp_inputs:
            args.append(runtime.cinn_pod_value_t(in_data))
        if output_shapes is None:
            correct_result, output_shapes = self.create_target_data(
                inputs_data, attrs
            )
        else:
            correct_result = self.create_target_data(inputs_data, attrs)

        func = self.__lower(op_name, inputs, output_shapes, attrs)
        builder = lang.Module.Builder(op_name, self.target)
        builder.add_function(func)
        module = builder.build()

        self.compiler.build(module)
        fn = self.compiler.lookup(func.name())

        out = []

        for out_shape in output_shapes:
            out.append(
                runtime.cinn_buffer_t(
                    np.zeros(out_shape).astype("float32"),
                    runtime.cinn_x86_device,
                    alignment,
                )
            )
        if do_infer_shape:
            infer_shapes = framework.Operator.get_op_shape_attrs("infershape")
            out_shapes = infer_shapes.infer_shape(
                op_name, input_shapes, attrs.attr_store
            )
            print("out_shapes", out_shapes)
            for out_shape in out_shapes[1:]:
                out.append(
                    runtime.cinn_buffer_t(
                        np.zeros(out_shape).astype("float32"),
                        runtime.cinn_x86_device,
                        alignment,
                    )
                )

        for out_data in out:
            args.append(runtime.cinn_pod_value_t(out_data))
        fn(args)

        out_result = out[len(out) - 1].numpy()
        if out_index is not None:
            out_result = out[out_index].numpy()
        np.testing.assert_allclose(out_result, correct_result, atol=1e-4)

    def __lower(self, op_name, inputs, output_shapes, attrs):
        types = [common.Float(32)]
        strategy_map = framework.Operator.get_op_attrs("CINNStrategy")
        func = strategy_map.apply_strategy(
            op_name, attrs, inputs, types, output_shapes, self.target
        )
        logging.warning('func:\n\n%s\n', func)
        return func

    def __gen_var_name(self):
        self.counter = self.counter + 1
        return "Var_" + str(self.counter)


if __name__ == "__main__":
    unittest.main()
