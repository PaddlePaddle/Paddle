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
from paddle.base.proto import framework_pb2


class TestSetValue(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def _test_for_new_program_format(self, program_bytes):
        restored_prog_as_is = framework_pb2.ProgramDesc.FromString(
            program_bytes
        )
        for block in restored_prog_as_is.blocks:
            for op in block.ops:
                if op.type in ("set_value", "set_value_grad"):
                    attr_names = [attr.name for attr in op.attrs]
                    self.assertTrue("values" in attr_names)
                    self.assertFalse("bool_values" in attr_names)
                    self.assertFalse("int32_values" in attr_names)
                    self.assertFalse("int64_values" in attr_names)
                    self.assertFalse("fp32_values" in attr_names)
                    self.assertFalse("fp64_values" in attr_names)
                    self.assertFalse("fp16_values" in attr_names)

    def _test_for_legacy_program_format(self, program_bytes):
        restored_prog_as_is = framework_pb2.ProgramDesc.FromString(
            program_bytes
        )
        for block in restored_prog_as_is.blocks:
            for op in block.ops:
                if op.type in ("set_value", "set_value_grad"):
                    attr_names = [attr.name for attr in op.attrs]
                    self.assertFalse("values" in attr_names)
                    self.assertTrue("bool_values" in attr_names)
                    self.assertTrue("int32_values" in attr_names)
                    self.assertTrue("int64_values" in attr_names)
                    self.assertTrue("fp32_values" in attr_names)
                    self.assertTrue("fp64_values" in attr_names)
                    self.assertTrue("fp16_values" in attr_names)

    def _test_equivalence(
        self,
        new_program_bytes,
        legacy_program_bytes,
        fetch_list,
        expected_outputs,
    ):
        normal_program = paddle.static.io.deserialize_program(new_program_bytes)
        converted_back_program = paddle.static.io.deserialize_program(
            legacy_program_bytes
        )

        exe = paddle.static.Executor(paddle.CPUPlace())
        [out] = exe.run(normal_program, fetch_list=fetch_list)
        np.testing.assert_allclose(out, expected_outputs[0])

        [out] = exe.run(converted_back_program, fetch_list=fetch_list)
        np.testing.assert_allclose(out, expected_outputs[0])

    def test_int32(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = paddle.ones([3, 4], dtype=paddle.int32)
            patch = np.array([41, 42]).astype(np.int32)
            index = (slice(None, 1), slice(None, 2))
            x = paddle.static.setitem(x, index, patch)

        x_input = np.ones([3, 4], dtype=np.int32)
        x_output = x_input.copy()
        x_output[:1, :2] = patch

        normal_program_bytes = mp._get_desc().serialize_to_string()
        legacy_program_bytes = mp._get_desc().serialize_to_string(
            legacy_format=True
        )

        self.assertNotEqual(normal_program_bytes, legacy_program_bytes)
        self._test_for_new_program_format(normal_program_bytes)
        self._test_for_legacy_program_format(legacy_program_bytes)
        self._test_equivalence(
            normal_program_bytes,
            legacy_program_bytes,
            fetch_list=[x.name],
            expected_outputs=[x_output],
        )

    def test_int64(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = paddle.ones([3, 4], dtype=paddle.int64)
            patch = np.array(
                [np.iinfo(np.int64).max, np.iinfo(np.int64).min]
            ).astype(np.int64)
            index = (slice(None, 1), slice(None, 2))
            x = paddle.static.setitem(x, index, patch)

        x_input = np.ones([3, 4], dtype=np.int64)
        x_output = x_input.copy()

        x_output[:1, :2] = patch

        self.fetch_list = [x.name]
        self.expected_outputs = [x_output]

        normal_program_bytes = mp._get_desc().serialize_to_string()
        legacy_program_bytes = mp._get_desc().serialize_to_string(
            legacy_format=True
        )

        self.assertNotEqual(normal_program_bytes, legacy_program_bytes)
        self._test_for_new_program_format(normal_program_bytes)
        self._test_for_legacy_program_format(legacy_program_bytes)
        self._test_equivalence(
            normal_program_bytes,
            legacy_program_bytes,
            fetch_list=[x.name],
            expected_outputs=[x_output],
        )

    def test_float32(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = paddle.ones([3, 4], dtype=paddle.float32)
            patch = np.array(
                [np.finfo(np.float32).max, np.finfo(np.float32).min]
            ).astype(np.float32)
            index = (slice(None, 1), slice(None, 2))
            x = paddle.static.setitem(x, index, patch)

        x_input = np.ones([3, 4], dtype=np.float32)
        x_output = x_input.copy()
        x_output[:1, :2] = patch

        normal_program_bytes = mp._get_desc().serialize_to_string()
        legacy_program_bytes = mp._get_desc().serialize_to_string(
            legacy_format=True
        )

        self.assertNotEqual(normal_program_bytes, legacy_program_bytes)
        self._test_for_new_program_format(normal_program_bytes)
        self._test_for_legacy_program_format(legacy_program_bytes)
        self._test_equivalence(
            normal_program_bytes,
            legacy_program_bytes,
            fetch_list=[x.name],
            expected_outputs=[x_output],
        )

    def test_float64(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = paddle.ones([3, 4], dtype=paddle.float64)
            patch = np.array(
                [np.finfo(np.float64).max, np.finfo(np.float64).min]
            ).astype(np.float64)
            index = (slice(None, 1), slice(None, 2))
            x = paddle.static.setitem(x, index, patch)

        x_input = np.ones([3, 4], dtype=np.float64)
        x_output = x_input.copy()
        x_output[:1, :2] = patch

        normal_program_bytes = mp._get_desc().serialize_to_string()
        legacy_program_bytes = mp._get_desc().serialize_to_string(
            legacy_format=True
        )

        self.assertNotEqual(normal_program_bytes, legacy_program_bytes)
        self._test_for_new_program_format(normal_program_bytes)
        self._test_for_legacy_program_format(legacy_program_bytes)
        self._test_equivalence(
            normal_program_bytes,
            legacy_program_bytes,
            fetch_list=[x.name],
            expected_outputs=[x_output],
        )

    def test_float16(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = paddle.ones([3, 4], dtype=paddle.float16)
            patch = np.array(
                [np.finfo(np.float16).max, np.finfo(np.float16).min]
            ).astype(np.float16)
            index = (slice(None, 1), slice(None, 2))
            x = paddle.static.setitem(x, index, patch)

        x_input = np.ones([3, 4], dtype=np.float16)
        x_output = x_input.copy()
        x_output[:1, :2] = patch

        normal_program_bytes = mp._get_desc().serialize_to_string()
        legacy_program_bytes = mp._get_desc().serialize_to_string(
            legacy_format=True
        )

        self.assertNotEqual(normal_program_bytes, legacy_program_bytes)
        self._test_for_new_program_format(normal_program_bytes)
        self._test_for_legacy_program_format(legacy_program_bytes)
        self._test_equivalence(
            normal_program_bytes,
            legacy_program_bytes,
            fetch_list=[x.name],
            expected_outputs=[x_output],
        )

    def test_bool(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = paddle.ones([3, 4], dtype=paddle.bool)
            patch = np.array([True, False])
            index = (slice(None, 1), slice(None, 2))
            x = paddle.static.setitem(x, index, patch)

        x_input = np.ones([3, 4], dtype=bool)
        x_output = x_input.copy()
        x_output[:1, :2] = patch

        normal_program_bytes = mp._get_desc().serialize_to_string()
        legacy_program_bytes = mp._get_desc().serialize_to_string(
            legacy_format=True
        )

        self.assertNotEqual(normal_program_bytes, legacy_program_bytes)
        self._test_for_new_program_format(normal_program_bytes)
        self._test_for_legacy_program_format(legacy_program_bytes)
        self._test_equivalence(
            normal_program_bytes,
            legacy_program_bytes,
            fetch_list=[x.name],
            expected_outputs=[x_output],
        )

    def test_complex64(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = paddle.complex(
                paddle.ones([3, 4], dtype=paddle.float32),
                paddle.ones([3, 4], dtype=paddle.float32),
            )
            patch = np.array([42.1 + 42.1j, 42.2 + 42.2j]).astype(np.complex64)
            index = (slice(None, 1), slice(None, 2))
            x = paddle.static.setitem(x, index, patch)

        x_input = (np.ones([3, 4]) + 1j * np.ones([3, 4])).astype(np.complex64)
        x_output = x_input.copy()
        x_output[:1, :2] = patch

        with self.assertRaisesRegex(RuntimeError, "Invalid data type"):
            legacy_program_bytes = mp._get_desc().serialize_to_string(
                legacy_format=True
            )

    def test_complex128(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = paddle.complex(
                paddle.ones([3, 4], dtype=paddle.float64),
                paddle.ones([3, 4], dtype=paddle.float64),
            )
            patch = np.array(
                [
                    np.finfo(np.float64).max + 1j * np.finfo(np.float64).min,
                    np.finfo(np.float64).min + 1j * np.finfo(np.float64).max,
                ]
            ).astype(np.complex128)
            index = (slice(None, 1), slice(None, 2))
            x = paddle.static.setitem(x, index, patch)

        x_input = (np.ones([3, 4]) + 1j * np.ones([3, 4])).astype(np.complex128)
        x_output = x_input.copy()
        x_output[:1, :2] = patch

        with self.assertRaisesRegex(RuntimeError, "Invalid data type"):
            legacy_program_bytes = mp._get_desc().serialize_to_string(
                legacy_format=True
            )


class TestAssignValue(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def _test_for_new_program_format(self, program_bytes):
        restored_prog_as_is = framework_pb2.ProgramDesc.FromString(
            program_bytes
        )
        for block in restored_prog_as_is.blocks:
            for op in block.ops:
                if op.type in ("assign_value"):
                    attr_names = [attr.name for attr in op.attrs]
                    self.assertTrue("values" in attr_names)
                    self.assertFalse("bool_values" in attr_names)
                    self.assertFalse("int32_values" in attr_names)
                    self.assertFalse("int64_values" in attr_names)
                    self.assertFalse("fp32_values" in attr_names)

    def _test_for_legacy_program_format(self, program_bytes):
        restored_prog_as_is = framework_pb2.ProgramDesc.FromString(
            program_bytes
        )
        for block in restored_prog_as_is.blocks:
            for op in block.ops:
                if op.type in ("set_value", "set_value_grad"):
                    attr_names = [attr.name for attr in op.attrs]
                    self.assertFalse("values" in attr_names)
                    self.assertTrue("bool_values" in attr_names)
                    self.assertTrue("int32_values" in attr_names)
                    self.assertTrue("int64_values" in attr_names)
                    self.assertTrue("fp32_values" in attr_names)

    def _test_equivalence(
        self,
        new_program_bytes,
        legacy_program_bytes,
        fetch_list,
        expected_outputs,
    ):
        normal_program = paddle.static.io.deserialize_program(new_program_bytes)
        converted_back_program = paddle.static.io.deserialize_program(
            legacy_program_bytes
        )
        exe = paddle.static.Executor(paddle.CPUPlace())
        out = exe.run(normal_program, fetch_list=fetch_list)
        np.testing.assert_allclose(out[0], expected_outputs[0])
        out = exe.run(converted_back_program, fetch_list=fetch_list)
        np.testing.assert_allclose(out[0], expected_outputs[0])

    def test_int32(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = np.array([[1, 1], [3, 4], [1, 3]]).astype(np.int32)
            out = paddle.assign(x)

        normal_program_bytes = mp._get_desc().serialize_to_string()
        legacy_program_bytes = mp._get_desc().serialize_to_string(
            legacy_format=True
        )
        self.assertNotEqual(normal_program_bytes, legacy_program_bytes)
        self._test_for_new_program_format(normal_program_bytes)
        self._test_for_legacy_program_format(legacy_program_bytes)
        self._test_equivalence(
            normal_program_bytes,
            legacy_program_bytes,
            fetch_list=[out.name],
            expected_outputs=[x],
        )

    def test_int64(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = np.array([[1, 1], [3, 4], [1, 3]]).astype(np.int64)
            out = paddle.assign(x)

        normal_program_bytes = mp._get_desc().serialize_to_string()
        legacy_program_bytes = mp._get_desc().serialize_to_string(
            legacy_format=True
        )

        self.assertNotEqual(normal_program_bytes, legacy_program_bytes)
        self._test_for_new_program_format(normal_program_bytes)
        self._test_for_legacy_program_format(legacy_program_bytes)
        self._test_equivalence(
            normal_program_bytes,
            legacy_program_bytes,
            fetch_list=[out.name],
            expected_outputs=[x],
        )

    def test_float32(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = np.random.random(size=(2, 5)).astype(np.float32)
            out = paddle.assign(x)

        normal_program_bytes = mp._get_desc().serialize_to_string()
        legacy_program_bytes = mp._get_desc().serialize_to_string(
            legacy_format=True
        )

        self.assertNotEqual(normal_program_bytes, legacy_program_bytes)
        self._test_for_new_program_format(normal_program_bytes)
        self._test_for_legacy_program_format(legacy_program_bytes)
        self._test_equivalence(
            normal_program_bytes,
            legacy_program_bytes,
            fetch_list=[out.name],
            expected_outputs=[x],
        )

    def test_float64(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = np.random.random(size=(2, 5)).astype(np.float64)
            out = paddle.assign(x)

        normal_program_bytes = mp._get_desc().serialize_to_string()
        legacy_program_bytes = mp._get_desc().serialize_to_string(
            legacy_format=True
        )

        self.assertNotEqual(normal_program_bytes, legacy_program_bytes)
        self._test_for_new_program_format(normal_program_bytes)
        self._test_for_legacy_program_format(legacy_program_bytes)
        self._test_equivalence(
            normal_program_bytes,
            legacy_program_bytes,
            fetch_list=[out.name],
            expected_outputs=[x],
        )

    def test_bool(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = np.random.choice(a=[False, True], size=(2, 5)).astype(np.bool_)
            out = paddle.assign(x)

        normal_program_bytes = mp._get_desc().serialize_to_string()
        legacy_program_bytes = mp._get_desc().serialize_to_string(
            legacy_format=True
        )

        self.assertNotEqual(normal_program_bytes, legacy_program_bytes)
        self._test_for_new_program_format(normal_program_bytes)
        self._test_for_legacy_program_format(legacy_program_bytes)
        self._test_equivalence(
            normal_program_bytes,
            legacy_program_bytes,
            fetch_list=[out.name],
            expected_outputs=[x],
        )

    def test_complex64(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = (
                np.random.random(size=(2, 5))
                + 1j * np.random.random(size=(2, 5))
            ).astype(np.complex64)
            out = paddle.assign(x)

        with self.assertRaisesRegex(RuntimeError, "Invalid data type"):
            legacy_program_bytes = mp._get_desc().serialize_to_string(
                legacy_format=True
            )

    def test_complex128(self):
        mp = paddle.static.Program()
        sp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x = (
                np.random.random(size=(2, 5))
                + 1j * np.random.random(size=(2, 5))
            ).astype(np.complex128)
            out = paddle.assign(x)

        with self.assertRaisesRegex(RuntimeError, "Invalid data type"):
            legacy_program_bytes = mp._get_desc().serialize_to_string(
                legacy_format=True
            )


if __name__ == '__main__':
    unittest.main()
