# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Tests for paddle/distributed/fleet/data_generator/data_generator.py
# Target: DataGenerator, MultiSlotStringDataGenerator, MultiSlotDataGenerator
# Coverage target: ~74.4% -> improved

"""
测试 paddle/distributed/fleet/data_generator/data_generator.py 中的数据生成器类。

Tests for data generator classes in paddle/distributed/fleet/data_generator/data_generator.py.
Covers DataGenerator (base class methods), MultiSlotStringDataGenerator,
MultiSlotDataGenerator, generate_sample, generate_batch, run_from_memory.
All I/O operations are mocked.
"""

import io
import unittest
from unittest.mock import patch


class TestDataGenerator(unittest.TestCase):
    """测试 DataGenerator 基类 / Test DataGenerator base class."""

    def test_init(self):
        """测试初始化默认值 / Test initialization defaults."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            DataGenerator,
        )

        dg = DataGenerator()
        self.assertIsNone(dg._proto_info)
        self.assertEqual(dg.batch_size_, 32)

    def test_set_batch(self):
        """测试设置批次大小 / Test set batch size."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            DataGenerator,
        )

        dg = DataGenerator()
        dg.set_batch(128)
        self.assertEqual(dg.batch_size_, 128)

    def test_generate_sample_not_implemented(self):
        """测试 generate_sample 未实现 / Test generate_sample raises NotImplementedError."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            DataGenerator,
        )

        dg = DataGenerator()
        with self.assertRaises(NotImplementedError):
            dg.generate_sample("test line")

    def test_gen_str_not_implemented(self):
        """测试 _gen_str 未实现 / Test _gen_str raises NotImplementedError."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            DataGenerator,
        )

        dg = DataGenerator()
        with self.assertRaises(NotImplementedError):
            dg._gen_str("test")

    def test_generate_batch_default(self):
        """测试默认 generate_batch / Test default generate_batch."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            DataGenerator,
        )

        dg = DataGenerator()
        samples = [("words", [1, 2, 3]), ("label", [0])]
        gen = dg.generate_batch(samples)
        result = list(gen())
        self.assertEqual(result, samples)

    def test_run_from_memory(self):
        """测试从内存运行数据生成 / Test run_from_memory."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            DataGenerator,
        )

        class TestDG(DataGenerator):
            def generate_sample(self, line):
                def local_iter():
                    for i in range(5):
                        yield ("words", [i, i + 1])

                return local_iter

            def _gen_str(self, line):
                return f"{line[0]} {line[1]}\n"

        dg = TestDG()
        dg.set_batch(2)

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            dg.run_from_memory()

        output = captured.getvalue()
        # Should have output for 5 samples in batches of 2 (3 batches: 2+2+1)
        self.assertTrue(len(output) > 0)

    def test_run_from_memory_with_none(self):
        """测试从内存运行时跳过None / Test run_from_memory skips None."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            DataGenerator,
        )

        class TestDG(DataGenerator):
            def generate_sample(self, line):
                def local_iter():
                    yield None
                    yield ("words", [1, 2])

                return local_iter

            def _gen_str(self, line):
                return f"{line[0]} {line[1]}\n"

        dg = TestDG()
        dg.set_batch(2)

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            dg.run_from_memory()

        output = captured.getvalue()
        # Only 1 sample (None was skipped)
        self.assertTrue(len(output) > 0)

    def test_run_from_stdin(self):
        """测试从标准输入运行数据生成 / Test run_from_stdin."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            DataGenerator,
        )

        class TestDG(DataGenerator):
            def generate_sample(self, line):
                def local_iter():
                    words = [int(x) for x in line.strip().split()]
                    yield ("words", words)

                return local_iter

            def _gen_str(self, line):
                return f"{line[0]} {line[1]}\n"

        dg = TestDG()
        dg.set_batch(2)

        captured = io.StringIO()
        with (
            patch("sys.stdout", captured),
            patch("sys.stdin", io.StringIO("1 2 3\n4 5 6\n")),
        ):
            dg.run_from_stdin()

        output = captured.getvalue()
        self.assertTrue(len(output) > 0)


class TestMultiSlotStringDataGenerator(unittest.TestCase):
    """测试 MultiSlotStringDataGenerator / Test MultiSlotStringDataGenerator."""

    def test_gen_str_list(self):
        """测试列表格式生成字符串 / Test _gen_str with list."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotStringDataGenerator,
        )

        dg = MultiSlotStringDataGenerator()
        result = dg._gen_str(
            [("words", ["1926", "08", "17"]), ("label", ["1"])]
        )
        self.assertEqual(result, "3 1926 08 17 1 1\n")

    def test_gen_str_tuple(self):
        """测试元组格式生成字符串 / Test _gen_str with tuple."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotStringDataGenerator,
        )

        dg = MultiSlotStringDataGenerator()
        result = dg._gen_str(
            (("words", ["1926", "08", "17"]), ("label", ["1"]))
        )
        self.assertEqual(result, "3 1926 08 17 1 1\n")

    def test_gen_str_zip(self):
        """测试 zip 格式生成字符串 / Test _gen_str with zip."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotStringDataGenerator,
        )

        dg = MultiSlotStringDataGenerator()
        zipped = zip(["words", "label"], [["1926", "08"], ["1"]])
        result = dg._gen_str(zipped)
        self.assertEqual(result, "2 1926 08 1 1\n")

    def test_gen_str_invalid_type(self):
        """测试无效类型抛出异常 / Test invalid type raises ValueError."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotStringDataGenerator,
        )

        dg = MultiSlotStringDataGenerator()
        with self.assertRaises(ValueError):
            dg._gen_str("invalid_input")

    def test_gen_str_empty(self):
        """测试空列表 / Test empty list."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotStringDataGenerator,
        )

        dg = MultiSlotStringDataGenerator()
        result = dg._gen_str([])
        self.assertEqual(result, "\n")

    def test_gen_str_single_slot(self):
        """测试单个slot / Test single slot."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotStringDataGenerator,
        )

        dg = MultiSlotStringDataGenerator()
        result = dg._gen_str([("words", ["hello", "world"])])
        self.assertEqual(result, "2 hello world\n")


class TestMultiSlotDataGenerator(unittest.TestCase):
    """测试 MultiSlotDataGenerator / Test MultiSlotDataGenerator."""

    def test_gen_str_first_call_int(self):
        """测试首次调用整型元素 / Test first call with int elements."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        result = dg._gen_str([("words", [1926, 8, 17]), ("label", [1])])
        self.assertEqual(result, "3 1926 8 17 1 1\n")
        self.assertIsNotNone(dg._proto_info)
        self.assertEqual(len(dg._proto_info), 2)
        self.assertEqual(dg._proto_info[0], ("words", "uint64"))
        self.assertEqual(dg._proto_info[1], ("label", "uint64"))

    def test_gen_str_first_call_float(self):
        """测试首次调用浮点元素 / Test first call with float elements."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        result = dg._gen_str([("words", [1.5, 2.5])])
        self.assertEqual(result, "2 1.5 2.5\n")
        self.assertEqual(dg._proto_info[0], ("words", "float"))

    def test_gen_str_first_call_invalid_name_type(self):
        """测试首次调用无效名称类型 / Test first call with invalid name type."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        with self.assertRaises(ValueError) as ctx:
            dg._gen_str([(123, [1, 2])])
        self.assertIn("must be in str type", str(ctx.exception))

    def test_gen_str_first_call_invalid_elements_type(self):
        """测试首次调用无效元素类型 / Test first call with invalid elements type."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        with self.assertRaises(ValueError) as ctx:
            dg._gen_str([("words", "not_a_list")])
        self.assertIn("must be in list type", str(ctx.exception))

    def test_gen_str_first_call_empty_elements(self):
        """测试首次调用空元素列表 / Test first call with empty elements."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        with self.assertRaises(ValueError) as ctx:
            dg._gen_str([("words", [])])
        self.assertIn("can not be empty", str(ctx.exception))

    def test_gen_str_first_call_invalid_elem_type(self):
        """测试首次调用无效元素类型 / Test first call with invalid element type."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        with self.assertRaises(ValueError) as ctx:
            dg._gen_str([("words", ["hello"])])
        self.assertIn("must be in int or float", str(ctx.exception))

    def test_gen_str_subsequent_call_int(self):
        """测试后续调用整型元素 / Test subsequent call with int elements."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        # First call to set proto_info
        dg._gen_str([("words", [1926, 8, 17]), ("label", [1])])
        # Second call
        result = dg._gen_str([("words", [100, 200]), ("label", [0])])
        self.assertEqual(result, "2 100 200 1 0\n")

    def test_gen_str_subsequent_call_float_promotion(self):
        """测试后续调用浮点升级 / Test subsequent call with float promotion."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        # First call with int
        dg._gen_str([("words", [1926, 8, 17])])
        self.assertEqual(dg._proto_info[0], ("words", "uint64"))

        # Second call with float
        result = dg._gen_str([("words", [1.5, 2.5])])
        self.assertEqual(result, "2 1.5 2.5\n")
        self.assertEqual(dg._proto_info[0], ("words", "float"))

    def test_gen_str_subsequent_call_inconsistent_fields(self):
        """测试后续调用字段不一致 / Test subsequent call with inconsistent fields."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        dg._gen_str([("words", [1, 2])])

        with self.assertRaises(ValueError) as ctx:
            dg._gen_str([("words", [1, 2]), ("label", [3])])
        self.assertIn("inconsistent", str(ctx.exception))

    def test_gen_str_subsequent_call_name_mismatch(self):
        """测试后续调用名称不匹配 / Test subsequent call with name mismatch."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        dg._gen_str([("words", [1, 2]), ("label", [3])])

        with self.assertRaises(ValueError) as ctx:
            dg._gen_str([("words", [1, 2]), ("other", [3])])
        self.assertIn("not match", str(ctx.exception))

    def test_gen_str_subsequent_call_invalid_elem(self):
        """测试后续调用无效元素类型 / Test subsequent call with invalid element type."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        dg._gen_str([("words", [1, 2])])

        with self.assertRaises(ValueError) as ctx:
            dg._gen_str([("words", ["hello", "world"])])
        self.assertIn("must be in int or float", str(ctx.exception))

    def test_gen_str_subsequent_call_invalid_name(self):
        """测试后续调用无效名称类型 / Test subsequent call with invalid name type."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        dg._gen_str([("words", [1, 2])])

        with self.assertRaises(ValueError) as ctx:
            dg._gen_str([(123, [1, 2])])
        self.assertIn("must be in str type", str(ctx.exception))

    def test_gen_str_subsequent_call_invalid_elements(self):
        """测试后续调用无效元素类型 / Test subsequent call with invalid elements type."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        dg._gen_str([("words", [1, 2])])

        with self.assertRaises(ValueError) as ctx:
            dg._gen_str([("words", "not_a_list")])
        self.assertIn("must be in list type", str(ctx.exception))

    def test_gen_str_subsequent_call_empty_elements(self):
        """测试后续调用空元素 / Test subsequent call with empty elements."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        dg._gen_str([("words", [1, 2])])

        with self.assertRaises(ValueError) as ctx:
            dg._gen_str([("words", [])])
        self.assertIn("can not be empty", str(ctx.exception))

    def test_gen_str_zip_format(self):
        """测试 zip 格式首次调用 / Test zip format first call."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        zipped = zip(["words", "label"], [[1, 2], [3]])
        result = dg._gen_str(zipped)
        self.assertEqual(result, "2 1 2 1 3\n")

    def test_gen_str_mixed_int_float_first_slot(self):
        """测试第一个slot中混合int和float / Test mixed int and float in first slot."""
        from paddle.distributed.fleet.data_generator.data_generator import (
            MultiSlotDataGenerator,
        )

        dg = MultiSlotDataGenerator()
        result = dg._gen_str([("words", [1, 2.5, 3])])
        self.assertEqual(result, "3 1 2.5 3\n")
        self.assertEqual(dg._proto_info[0], ("words", "float"))


if __name__ == '__main__':
    unittest.main()
