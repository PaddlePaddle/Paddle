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
from fused_pass.pass_test import PassTest

import paddle

paddle.enable_static()


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_cuda(),
    "core is not complied with CUDA",
)
class TestRemoveUselessScalePattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_progam(self):
        pir_program = None
        with paddle.pir_utils.IrGuard():
            pir_program = paddle.static.Program()
            with paddle.pir.core.program_guard(pir_program):
                x = paddle.static.data(
                    name='x', shape=[3, 1, 28, 28], dtype='float32'
                )
                out = paddle.scale(x, scale=1.0, bias=0.0)
        self.pass_list = ['identity_op_clean_pass']
        self.feeds = {"x": np.random.random((3, 1, 28, 28)).astype("float32")}
        self.fetch_list = [out]
        self.valid_op_map = {"pd_op.scale": 0}
        return pir_program

    def sample_program(self):
        pir_program = self.build_ir_progam()
        yield pir_program, False

    def test_check_output(self):
        self.check_pass_correct()


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_cuda(),
    "core is not complied with CUDA",
)
class TestRemoveRedundentScalePattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for bias_after_scale_1 in [True, False]:
            for bias_after_scale_2 in [True, False]:
                pir_program = None
                with paddle.pir_utils.IrGuard():
                    pir_program = paddle.static.Program()
                    with paddle.pir.core.program_guard(pir_program):
                        x = paddle.static.data(
                            name='x', shape=[3, 1, 28, 28], dtype='float32'
                        )
                        scale_out1 = paddle.scale(
                            x,
                            scale=2.0,
                            bias=1.0,
                            bias_after_scale=bias_after_scale_1,
                        )
                        out = paddle.scale(
                            scale_out1,
                            scale=2.0,
                            bias=2.0,
                            bias_after_scale=bias_after_scale_2,
                        )
                self.pass_list = ['identity_op_clean_pass']
                self.feeds = {
                    "x": np.random.random((3, 1, 28, 28)).astype("float32")
                }
                self.fetch_list = [out]
                self.valid_op_map = {"pd_op.scale": 1}
                yield pir_program, False

    def test_check_output(self):
        self.check_pass_correct()


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_cuda(),
    "core is not complied with CUDA",
)
class TestRemoveUselessCastPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for tmp_type in ['float32', 'float16']:
            pir_program = None
            with paddle.pir_utils.IrGuard():
                pir_program = paddle.static.Program()
                with paddle.pir.core.program_guard(pir_program):
                    x = paddle.static.data(
                        name='x', shape=[3, 1, 28, 28], dtype=tmp_type
                    )
                    out = paddle.cast(x, tmp_type)
            self.pass_list = ['identity_op_clean_pass']
            self.feeds = {
                "x": np.random.random((3, 1, 28, 28)).astype(tmp_type)
            }
            self.fetch_list = [out]
            self.valid_op_map = {"pd_op.cast": 0}
            yield pir_program, False

    def test_check_output(self):
        self.check_pass_correct()


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_cuda(),
    "core is not complied with CUDA",
)
class TestRemoveUselessConcatPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        pir_program = None
        with paddle.pir_utils.IrGuard():
            pir_program = paddle.static.Program()
            with paddle.pir.core.program_guard(pir_program):
                x_input = paddle.static.data(
                    name='x_input', shape=[3, 1, 28, 28], dtype="float32"
                )
                out = paddle.concat(x=[x_input])
        self.pass_list = ['identity_op_clean_pass']
        self.feeds = {
            "x_input": np.random.random((3, 1, 28, 28)).astype("float32")
        }
        self.fetch_list = [out]
        self.valid_op_map = {"pd_op.concat": 0}
        yield pir_program, False

    def test_check_output(self):
        self.check_pass_correct()


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_cuda(),
    "core is not complied with CUDA",
)
class TestRemoveRedundentCastPattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for type_1 in ["float16"]:
            for type_2 in ["int32"]:
                pir_program = None
                with paddle.pir_utils.IrGuard():
                    pir_program = paddle.static.Program()
                    with paddle.pir.core.program_guard(pir_program):
                        x = paddle.static.data(
                            name='x', shape=[3, 1, 28, 28], dtype="float32"
                        )
                        out = paddle.cast(paddle.cast(x, type_1), type_2)
                self.pass_list = ['identity_op_clean_pass']
                self.feeds = {
                    "x": np.random.random((3, 1, 28, 28)).astype("float32")
                }
                self.fetch_list = [out]
                self.valid_op_map = {"pd_op.cast": 1}
                yield pir_program, False

    def test_check_output(self):
        self.check_pass_correct()


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_cuda(),
    "core is not complied with CUDA",
)
class TestRemoveRedundentTransposePattern(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for perm1_shape in [[1, 2, 0]]:
            for perm2_shape in [[0, 2, 1]]:
                pir_program = None
                with paddle.pir_utils.IrGuard():
                    pir_program = paddle.static.Program()
                    with paddle.pir.core.program_guard(pir_program):
                        x = paddle.static.data(
                            name='x', shape=[2, 3, 4], dtype="float32"
                        )
                        out = paddle.transpose(
                            paddle.transpose(x, perm1_shape), perm2_shape
                        )
                self.pass_list = ['identity_op_clean_pass']
                self.feeds = {
                    "x": np.random.random((2, 3, 4)).astype("float32")
                }
                self.fetch_list = [out]
                self.valid_op_map = {"pd_op.transpose": 1}
                yield pir_program, False

    def test_check_output(self):
        self.check_pass_correct()


class TestRemoveRedundentTransposePatternWithCpu(
    TestRemoveRedundentTransposePattern
):
    def setUp(self):
        self.place_runtime = "cpu"


class TestRemoveRedundentCastPatternWithCpu(TestRemoveRedundentCastPattern):
    def setUp(self):
        self.place_runtime = "cpu"


class TestRemoveUselessCastPatternWithCpu(TestRemoveUselessCastPattern):
    def setUp(self):
        self.place_runtime = "cpu"


class TestRemoveUselessConcatPatternWithCpu(TestRemoveUselessConcatPattern):
    def setUp(self):
        self.place_runtime = "cpu"


class TestRemoveRedundentScalePatternWithCpu(TestRemoveRedundentScalePattern):
    def setUp(self):
        self.place_runtime = "cpu"


class TestRemoveUselessScalePatternWithCpu(TestRemoveUselessScalePattern):
    def setUp(self):
        self.place_runtime = "cpu"


if __name__ == "__main__":
    unittest.main()
