#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework


class TestInferNoNeedBufferSlots(unittest.TestCase):
    def net(self):
        x1 = (
            fluid.default_main_program()
            .global_block()
            .create_var(dtype="float32", shape=[1], lod_level=0, name="x1")
        )
        x2 = (
            fluid.default_main_program()
            .global_block()
            .create_var(dtype="float32", shape=[1], lod_level=0, name="x2")
        )
        x = paddle.add(x1, x2)
        return x

    def test_infer_no_need_buffer_slots(self):
        program = framework.Program()
        startup_program = framework.Program()
        with fluid.program_guard(program, startup_program):
            loss = self.net()
            sgd = fluid.optimizer.SGD(learning_rate=0.01)
            sgd.minimize(loss)

        block = program.global_block()
        for idx, op in enumerate(block.ops):
            op_desc = op.desc
            inputs = {}
            for input_name in op_desc.input_names():
                inputs[input_name] = op_desc.input(input_name)
            outputs = {}
            for output_name in op_desc.output_names():
                outputs[output_name] = op_desc.output(output_name)
            attrs = {}
            for attr_name in op_desc.attr_names():
                attrs[attr_name] = op_desc.attr(attr_name)
            if idx == 0:
                # elementwise_add op
                self.assertEqual(
                    core.infer_no_need_buffer_slots(
                        op.type, inputs, outputs, attrs
                    ),
                    set([]),
                )
            elif idx == 1:
                # fill constant op
                self.assertEqual(
                    core.infer_no_need_buffer_slots(
                        op.type, inputs, outputs, attrs
                    ),
                    set([]),
                )
            else:
                # elementwise_add_grad op
                self.assertEqual(
                    core.infer_no_need_buffer_slots(
                        op.type, inputs, outputs, attrs
                    ),
                    set(['Y', 'X']),
                )


if __name__ == '__main__':
    unittest.main()
