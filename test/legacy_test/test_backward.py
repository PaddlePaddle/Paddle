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

import numpy as np

import paddle
from paddle.base import backward


class BackwardNet:
    """
    Abstract Base Class.
    All Net inherited this Class should implement two functions:
        build_model: build net to test the logic of backward
        init_data: fake input data to test all programs.
    """

    def __init__(self):
        self.stop_gradient_grad_vars = set()
        self.no_grad_vars = set()
        self.params_names = set()
        self.op_path = []

    def build_model(self):
        """
        Build net to test the logic of backward.
        :return: loss
        """
        raise NotImplementedError

    def init_data(self):
        """
        Fake input data to test all programs.
        :return: dict, {'var_name': var_data}
        """
        raise NotImplementedError


# TODO(Aurelius84): add conditional network test
class ConditionalNet(BackwardNet):
    def __init__(self):
        super().__init__()


class TestBackwardUninitializedVariable(unittest.TestCase):
    """this case is found in yolov5 while to_static.
    gradient aggregation may cause sum a invalid variable.
    """

    def test(self):
        paddle.enable_static()
        main_prg, startup_prg = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(main_prg, startup_prg):
            gt = paddle.static.data(name='gt', shape=[4], dtype='float32')
            x = paddle.static.data(name='x', shape=[2], dtype='float32')
            gt.stop_gradient = True
            x.stop_gradient = False
            gt = gt.reshape([4, 1]).reshape([4])
            loss = (
                paddle.nn.functional.binary_cross_entropy(x, gt[:2])
                + (gt[2:4] * x).sum()
            )
            exe = paddle.static.Executor()
            paddle.base.backward.gradients(loss, [])
            exe.run(startup_prg)
            # Optimizer
            out = exe.run(
                main_prg,
                feed={
                    'gt': np.array([1.0, 1.0, 0.0, 0.0], dtype='float32'),
                    'x': np.array([0.5, 0.5], dtype='float32'),
                },
                fetch_list=[loss],
            )
            print(out)


class TestStripGradSuffix(unittest.TestCase):
    def test_strip_grad_suffix(self):
        cases = (
            ('x@GRAD', 'x'),
            ('x@GRAD@GRAD', 'x'),
            ('x@GRAD@RENAME@1', 'x'),
            ('x@GRAD_slice_0@GRAD', 'x@GRAD_slice_0'),
            ('grad/grad/x@GRAD@RENAME@block0@1@GRAD', 'x'),
        )
        for input_, desired in cases:
            self.assertEqual(backward._strip_grad_suffix_(input_), desired)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
