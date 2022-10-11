# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys

sys.path.append("..")
from op_test import OpTest

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import warnings

paddle.enable_static()


def execute(main_program, startup_program):
    if paddle.is_compiled_with_xpu():
        place = paddle.XPUPlace(0)
    else:
        place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    exe.run(startup_program)
    exe.run(main_program)


def get_vaild_warning_num(warning, w):
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
            data1 = paddle.full(shape=[1, 3, 8, 8],
                                fill_value=0.5,
                                dtype='float32')
            data2 = paddle.full(shape=[1, 3, 5, 5],
                                fill_value=0.5,
                                dtype='float32')
            shape = paddle.shape(data2)
            with paddle.static.device_guard("cpu"):
                shape = paddle.slice(shape, axes=[0], starts=[0], ends=[4])
                with paddle.static.device_guard("xpu"):
                    out = fluid.layers.crop_tensor(data1, shape=shape)
        # check if the device attr is set correctly
        all_ops = main_program.global_block().ops
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        for op in all_ops:
            if op.type == 'slice':
                self.assertEqual(op.desc.attr(device_attr_name), "cpu")
            if op.type == 'crop_tensor':
                self.assertEqual(op.desc.attr(device_attr_name), "xpu")

        execute(main_program, startup_program)

    def test_device_guard_with_id(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data1 = paddle.full(shape=[1, 3, 8, 8],
                                fill_value=0.5,
                                dtype='float32')
            data2 = paddle.full(shape=[1, 3, 5, 5],
                                fill_value=0.5,
                                dtype='float32')
            shape = paddle.shape(data2)
            with paddle.static.device_guard("cpu"):
                shape = paddle.slice(shape, axes=[0], starts=[0], ends=[4])
                with paddle.static.device_guard("xpu:1"):
                    out = fluid.layers.crop_tensor(data1, shape=shape)
        # check if the device attr is set correctly
        all_ops = main_program.global_block().ops
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        for op in all_ops:
            if op.type == 'slice':
                self.assertEqual(op.desc.attr(device_attr_name), "cpu")
            if op.type == 'crop_tensor':
                self.assertEqual(op.desc.attr(device_attr_name), "xpu:1")

        execute(main_program, startup_program)

    def test_cpu_only_op(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.full(shape=[2, 255, 13, 13],
                            fill_value=0.3,
                            dtype='float32')
            gt_box = paddle.full(shape=[2, 6, 4],
                                 fill_value=0.5,
                                 dtype='float32')
            gt_label = paddle.full(shape=[2, 6], fill_value=1.0, dtype='int32')
            gt_score = paddle.full(shape=[2, 6],
                                   fill_value=0.5,
                                   dtype='float32')
            anchors = [
                10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156,
                198, 373, 326
            ]
            anchor_mask = [0, 1, 2]
            with paddle.static.device_guard("xpu"):
                # yolov3_loss only has cpu kernel, so its cpu kernel will be executed
                loss = fluid.layers.yolov3_loss(x=x,
                                                gt_box=gt_box,
                                                gt_label=gt_label,
                                                gt_score=gt_score,
                                                anchors=anchors,
                                                anchor_mask=anchor_mask,
                                                class_num=80,
                                                ignore_thresh=0.7,
                                                downsample_ratio=32)

        execute(main_program, startup_program)

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
                    while_op = fluid.layers.While(cond=cond)
                    with while_op.block():
                        i = paddle.increment(x=i, value=1)
                        fluid.layers.less_than(x=i, y=loop_len, cond=cond)

        warning = "The Op(while) is not support to set device."
        warning_num = get_vaild_warning_num(warning, w)
        assert warning_num == 1

        all_ops = main_program.global_block().ops
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        for op in all_ops:
            if op.type == 'while':
                self.assertEqual(op.desc.attr(device_attr_name), "")

        execute(main_program, startup_program)

    def test_error(self):

        def device_attr():
            with paddle.static.device_guard("cpu1"):
                out = paddle.full(shape=[1], fill_value=0.2, dtype='float32')

        def device_attr2():
            with paddle.static.device_guard("cpu:1"):
                out = paddle.full(shape=[1], fill_value=0.2, dtype='float32')

        self.assertRaises(ValueError, device_attr)
        self.assertRaises(ValueError, device_attr2)

    # check if op_descs have op_device attr
    def test_op_descs_device_attr(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data1 = paddle.static.data(name="data_1",
                                       shape=[4, 2],
                                       dtype="float32")
            label = paddle.static.data(name="label",
                                       shape=[4, 1],
                                       dtype="int64")
            fc1 = paddle.static.nn.fc(x=data1, size=10)
            fc2 = paddle.static.nn.fc(x=fc1, size=10)
            with paddle.static.device_guard("xpu"):
                out = paddle.nn.functional.softmax_with_cross_entropy(
                    logits=fc1 + fc2, label=label)
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
                self.assertEqual(op.desc.attr(device_attr_name), "xpu")


if __name__ == '__main__':
    unittest.main()
