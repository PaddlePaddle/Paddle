# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid import core

paddle.enable_static()


class TestCastOpTranscriber(unittest.TestCase):
    def test_op(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.to_tensor([2, 3, 4], 'float64')
                y = paddle.cast(x, 'uint8')

                default_job = core.Job("default")
                type_to_program = {"default": main_program.desc}
                plan = core.Plan([default_job], type_to_program)
                new_exe = core.StandaloneExecutor(place, plan, new_scope)


class TestEmbeddingOpTranscriber(unittest.TestCase):
    def test_op(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.static.data(name="x", shape=[2, 4], dtype=np.int64)
                embedding = paddle.nn.Embedding(
                    10, 3, weight_attr=paddle.nn.initializer.Constant(value=1.0)
                )
                output = embedding(x)

                default_job = core.Job("default")
                type_to_program = {"default": main_program.desc}
                plan = core.Plan([default_job], type_to_program)
                new_exe = core.StandaloneExecutor(place, plan, new_scope)


class TestIncrementOpTranscriber(unittest.TestCase):
    def test_op(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                data = paddle.zeros(shape=[1], dtype='float32')
                counter = paddle.increment(data)

                default_job = core.Job("default")
                type_to_program = {"default": main_program.desc}
                plan = core.Plan([default_job], type_to_program)
                new_exe = core.StandaloneExecutor(place, plan, new_scope)


class TestAssignValueOpTranscriber(unittest.TestCase):
    def test_op(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.to_tensor(
                    [[0.1, 0.2], [0.3, 0.4]],
                    place=paddle.CPUPlace(),
                    stop_gradient=False,
                )

                default_job = core.Job("default")
                type_to_program = {"default": main_program.desc}
                plan = core.Plan([default_job], type_to_program)
                new_exe = core.StandaloneExecutor(place, plan, new_scope)


class TestRnnOpTranscriber(unittest.TestCase):
    def test_op(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.randn((4, 16))
                prev_h = paddle.randn((4, 32))

                cell = paddle.nn.SimpleRNNCell(16, 32)
                y, h = cell(x, prev_h)

                default_job = core.Job("default")
                type_to_program = {"default": main_program.desc}
                plan = core.Plan([default_job], type_to_program)
                new_exe = core.StandaloneExecutor(place, plan, new_scope)


class TestOneHotOpTranscriber(unittest.TestCase):
    def test_mutable_attribute(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                depth = paddle.assign(np.array([10], dtype=np.int32))
                label = paddle.static.data(
                    name="label", shape=[-1, 1], dtype="int64"
                )
                one_hot_label = paddle.nn.functional.one_hot(
                    x=label, num_classes=depth
                )

        _ = paddle.fluid.core.translate_newirprogram(main_program.desc)

    def test_normal_attribute(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                depth = 10
                label = paddle.static.data(
                    name="label", shape=[-1, 1], dtype="int64"
                )
                one_hot_label = paddle.nn.functional.one_hot(
                    x=label, num_classes=depth
                )

        _ = paddle.fluid.core.translate_newirprogram(main_program.desc)


if __name__ == "__main__":
    unittest.main()
