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

from __future__ import annotations

import inspect
import random
import unittest
from typing import TYPE_CHECKING
from unittest.mock import patch

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

from paddle.jit.sot.opcode_translator.custom_code import CustomCode
from paddle.jit.sot.opcode_translator.executor.executor_cache import (
    OpcodeExecutorCache,
)

if TYPE_CHECKING:
    from types import FrameType


def fake_frames() -> tuple[
    FrameType,
    FrameType,
    FrameType,
    FrameType,
    FrameType,
]:
    def fake_inner_fn_1():
        frame = inspect.currentframe()
        assert frame is not None
        return frame

    def fake_inner_fn_2():
        frame = inspect.currentframe()
        assert frame is not None
        return frame

    def fake_inner_fn_3():
        frame = inspect.currentframe()
        assert frame is not None
        return frame

    def fake_inner_fn_4():
        frame = inspect.currentframe()
        assert frame is not None
        return frame

    def fake_inner_fn_5():
        frame = inspect.currentframe()
        assert frame is not None
        return frame

    return (
        fake_inner_fn_1(),
        fake_inner_fn_2(),
        fake_inner_fn_3(),
        fake_inner_fn_4(),
        fake_inner_fn_5(),
    )


(
    FRAME_1,
    FRAME_2,
    FRAME_3,
    FRAME_4,
    FRAME_5,
) = fake_frames()


def mock_start_translate(frame: FrameType, **kwargs):
    translate_map = {
        FRAME_1: (CustomCode(FRAME_2.f_code, False), lambda frame: True),
        FRAME_3: (
            CustomCode(FRAME_4.f_code, False),
            lambda frame: False,
        ),  # Always re-compile
        FRAME_5: (CustomCode(None, False), lambda frame: True),
    }
    return translate_map[frame]


class TestOpcodeExecutorCache(unittest.TestCase):
    def reset(self):
        global translate_count
        translate_count = 0
        OpcodeExecutorCache().clear()

    @patch(
        "paddle.jit.sot.opcode_translator.executor.executor_cache.start_translate",
        mock_start_translate,
    )
    def test_cache_hit(self):
        with test_instruction_translator_cache_context() as ctx:
            translated_code_1 = OpcodeExecutorCache()(FRAME_1)
            assert translated_code_1 is not None
            self.assertEqual(translated_code_1.code, FRAME_2.f_code)
            self.assertEqual(ctx.translate_count, 1)
            # cache hit
            translated_code_2 = OpcodeExecutorCache()(FRAME_1)
            assert translated_code_2 is not None
            self.assertEqual(translated_code_2.code, FRAME_2.f_code)
            self.assertEqual(ctx.translate_count, 1)

    @patch(
        "paddle.jit.sot.opcode_translator.executor.executor_cache.start_translate",
        mock_start_translate,
    )
    def test_cache_miss_due_to_unknown_code(self):
        with test_instruction_translator_cache_context() as ctx:
            translated_code_1 = OpcodeExecutorCache()(FRAME_1)
            assert translated_code_1 is not None
            self.assertEqual(translated_code_1.code, FRAME_2.f_code)
            self.assertEqual(ctx.translate_count, 1)
            # cache miss
            translated_code_2 = OpcodeExecutorCache()(FRAME_3)
            assert translated_code_2 is not None
            self.assertEqual(translated_code_2.code, FRAME_4.f_code)
            self.assertEqual(ctx.translate_count, 2)

    @patch(
        "paddle.jit.sot.opcode_translator.executor.executor_cache.start_translate",
        mock_start_translate,
    )
    def test_cache_miss_due_to_check_failed(self):
        with test_instruction_translator_cache_context() as ctx:
            translated_code_1 = OpcodeExecutorCache()(FRAME_3)
            assert translated_code_1 is not None
            self.assertEqual(translated_code_1.code, FRAME_4.f_code)
            self.assertEqual(ctx.translate_count, 1)
            # cache miss
            translated_code_2 = OpcodeExecutorCache()(FRAME_3)
            assert translated_code_2 is not None
            self.assertEqual(translated_code_2.code, FRAME_4.f_code)
            self.assertEqual(ctx.translate_count, 2)


def foo(x):
    return x + 1


class TestCacheExceedLimit(TestCaseBase):
    def test_cache_exceed_limit(self):
        for _ in range(30):
            input = random.random()
            self.assert_results(foo, input)


if __name__ == '__main__':
    unittest.main()
