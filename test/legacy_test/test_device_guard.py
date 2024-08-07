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

import paddle

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
    def test_cpu_only_op(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.full(
                shape=[2, 255, 13, 13], fill_value=0.3, dtype='float32'
            )
            gt_box = paddle.full(
                shape=[2, 6, 4], fill_value=0.5, dtype='float32'
            )
            gt_label = paddle.full(shape=[2, 6], fill_value=1.0, dtype='int32')
            gt_score = paddle.full(
                shape=[2, 6], fill_value=0.5, dtype='float32'
            )
            anchors = [
                10,
                13,
                16,
                30,
                33,
                23,
                30,
                61,
                62,
                45,
                59,
                119,
                116,
                90,
                156,
                198,
                373,
                326,
            ]
            anchor_mask = [0, 1, 2]
            with paddle.static.device_guard("gpu"):
                # yolo_loss only has cpu kernel, so its cpu kernel will be executed
                loss = paddle.vision.ops.yolo_loss(
                    x=x,
                    gt_box=gt_box,
                    gt_label=gt_label,
                    gt_score=gt_score,
                    anchors=anchors,
                    anchor_mask=anchor_mask,
                    class_num=80,
                    ignore_thresh=0.7,
                    downsample_ratio=32,
                )

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


if __name__ == '__main__':
    unittest.main()
