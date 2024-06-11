# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import warnings

import paddle
from paddle.base import core, in_pir_mode
from paddle.pir_utils import test_with_pir_api

paddle.enable_static()


def execute(main_program, startup_program):
    if paddle.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    else:
        place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    exe.run(startup_program)
    exe.run(main_program)


def get_valid_warning_num(warning, w):
    num = 0
    for i in range(len(w)):
        if warning in str(w[i].message):
            num += 1
    return num


class TestDeviceGuard(unittest.TestCase):
    def test_device_guard(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data1 = paddle.full(
                shape=[1, 3, 8, 8], fill_value=0.5, dtype='float32'
            )
            data2 = paddle.full(
                shape=[1, 3, 5, 5], fill_value=0.5, dtype='float32'
            )
            shape = paddle.shape(data2)
            with paddle.static.device_guard("cpu"):
                shape = paddle.slice(shape, axes=[0], starts=[0], ends=[4])
                with paddle.static.device_guard("gpu"):
                    out = paddle.crop(data1, shape=shape)
        # check if the device attr is set correctly
        all_ops = main_program.global_block().ops
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        for op in all_ops:
            if op.type == 'slice':
                self.assertEqual(op.desc.attr(device_attr_name), "cpu")
            if op.type == 'crop_tensor':
                self.assertEqual(op.desc.attr(device_attr_name), "gpu")

        execute(main_program, startup_program)

    def test_device_guard_with_id(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data1 = paddle.full(
                shape=[1, 3, 8, 8], fill_value=0.5, dtype='float32'
            )
            data2 = paddle.full(
                shape=[1, 3, 5, 5], fill_value=0.5, dtype='float32'
            )
            shape = paddle.shape(data2)
            with paddle.static.device_guard("cpu"):
                shape = paddle.slice(shape, axes=[0], starts=[0], ends=[4])
                with paddle.static.device_guard("gpu:1"):
                    out = paddle.crop(data1, shape=shape)
        # check if the device attr is set correctly
        all_ops = main_program.global_block().ops
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        for op in all_ops:
            if op.type == 'slice':
                self.assertEqual(op.desc.attr(device_attr_name), "cpu")
            if op.type == 'crop_tensor':
                self.assertEqual(op.desc.attr(device_attr_name), "gpu:1")

        execute(main_program, startup_program)

    @test_with_pir_api
    def test_without_kernel_op(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.full(shape=[1], dtype='int64', fill_value=0)
            loop_len = paddle.full(shape=[1], dtype='int64', fill_value=10)
            cond = paddle.less_than(x=i, y=loop_len)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with paddle.static.device_guard("cpu"):
                    while_op = paddle.static.nn.control_flow.While(cond=cond)
                    with while_op.block():
                        i = paddle.increment(x=i, value=1)
                        paddle.assign(paddle.less_than(x=i, y=loop_len), cond)
        if not in_pir_mode():
            warning = "The Op(while) is not support to set device."
            warning_num = get_valid_warning_num(warning, w)
            assert warning_num == 1

        all_ops = main_program.global_block().ops
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        for op in all_ops:
            op_name = op.name() if in_pir_mode() else op.type
            if op_name == 'while':
                self.assertEqual(op.desc.attr(device_attr_name), "")

        execute(main_program, startup_program)

    # check if op_descs have op_device attr
    def test_op_descs_device_attr(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data1 = paddle.static.data(
                name="data_1", shape=[4, 2], dtype="float32"
            )
            label = paddle.static.data(
                name="label", shape=[4, 1], dtype="int64"
            )
            fc1 = paddle.static.nn.fc(x=data1, size=10)
            fc2 = paddle.static.nn.fc(x=fc1, size=10)
            with paddle.static.device_guard("gpu"):
                out = paddle.nn.functional.softmax_with_cross_entropy(
                    logits=fc1 + fc2, label=label
                )
                loss = paddle.mean(out)
                opt = paddle.optimizer.SGD(0.1)
                opt.minimize(loss)

        all_ops = main_program.global_block().ops
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        for op in all_ops:
            self.assertEqual(True, op.desc.has_attr(device_attr_name))
            # fill_constant(backward op) is append to mean op, which should have
            # the same op_device value as mean op
            if op.desc == 'fill_constant':
                self.assertEqual(op.desc.attr(device_attr_name), "gpu")


if __name__ == '__main__':
    unittest.main()
