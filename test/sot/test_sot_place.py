# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from contextlib import contextmanager

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


@contextmanager
def device_guard(place: str):
    original_place = paddle.get_device()
    try:
        paddle.set_device(place)
        yield
    finally:
        paddle.set_device(original_place)


@check_no_breakgraph
def run_diff_logic_by_check_expected_place(x: paddle.Tensor):
    expected_place_str = paddle.get_device()
    if "cpu" in expected_place_str:
        return x + 1
    elif expected_place_str.startswith("gpu"):
        return x + 2
    elif "xpu" in expected_place_str:
        return x + 3
    elif "npu" in expected_place_str:
        return x + 4
    return x


class TestCheckExpectedPlace(TestCaseBase):
    def test_check_cpu(self):
        x = paddle.to_tensor(0.0)
        with device_guard("cpu"):
            self.assert_results(run_diff_logic_by_check_expected_place, x.cpu())

    @unittest.skipUnless(
        paddle.is_compiled_with_cuda(),
        "This test case needs to be compiled with CUDA",
    )
    def test_check_gpu(self):
        x = paddle.to_tensor(0.0)
        with device_guard("gpu"):
            self.assert_results(
                run_diff_logic_by_check_expected_place, x.cuda()
            )

    @unittest.skipUnless(
        paddle.is_compiled_with_xpu(),
        "This test case needs to be compiled with XPU",
    )
    def test_check_xpu(self):
        x = paddle.to_tensor(0.0)
        with device_guard("xpu"):
            self.assert_results(
                run_diff_logic_by_check_expected_place, x.to("xpu")
            )


class TestExpectedPlaceGuard(TestCaseBase):
    @unittest.skipUnless(
        paddle.is_compiled_with_cuda(),
        "This test case needs to be compiled with cuda",
    )
    def test_expected_place_guard(self):
        x = paddle.to_tensor(0.0)
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            with device_guard("cpu"):
                self.assert_results(
                    run_diff_logic_by_check_expected_place, x.cpu()
                )
            self.assertEqual(ctx.translate_count, 1)
            with device_guard("gpu"):
                self.assert_results(
                    run_diff_logic_by_check_expected_place, x.cuda()
                )
            self.assertEqual(ctx.translate_count, 2)


if __name__ == "__main__":
    unittest.main()
