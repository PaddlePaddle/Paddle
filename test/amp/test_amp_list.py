# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core
from paddle.static.amp import AutoMixedPrecisionLists, fp16_lists


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not core.is_compiled_with_xpu(),
    "Require compiled with CUDA or XPU.",
)
@unittest.skipIf(
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
class TestAMPList(unittest.TestCase):
    def setUp(self):
        self.default_black_list = [
            'linear_interp_v2',
            'nearest_interp_v2',
            'bilinear_interp_v2',
            'bicubic_interp_v2',
            'trilinear_interp_v2',
        ]
        self.custom_white_list = [
            'lookup_table',
            'lookup_table_v2',
        ]

    def check_if_op_in_list(self, op_list, amp_list):
        for op in op_list:
            self.assertTrue(op in amp_list)

    def check_if_op_not_in_list(self, op_list, amp_list):
        for op in op_list:
            self.assertTrue(op not in amp_list)

    def test_static(self):
        amp_list = AutoMixedPrecisionLists(
            custom_white_list=self.custom_white_list
        )
        self.check_if_op_in_list(self.default_black_list, amp_list.black_list)
        self.check_if_op_in_list(self.custom_white_list, amp_list.white_list)
        self.check_if_op_not_in_list(
            self.custom_white_list, amp_list.black_list
        )
        if paddle.amp.is_float16_supported():
            self.check_if_op_not_in_list(
                self.custom_white_list, amp_list.black_list
            )

    def test_eager(self):
        if not paddle.amp.is_float16_supported():
            return
        white_list = paddle.amp.white_list()
        black_list = paddle.amp.black_list()
        self.check_if_op_in_list(
            self.default_black_list, black_list["float16"]["O2"]
        )
        self.check_if_op_not_in_list(['log', 'elementwise_add'], white_list)
        with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}):
            out1 = paddle.rand([2, 3]) + paddle.rand([2, 3])
            out2 = out1.mean()
            out3 = paddle.log(out2)
        self.check_if_op_not_in_list(['log', 'elementwise_add'], white_list)
        self.assertEqual(out1.dtype, paddle.float16)
        self.assertEqual(out2.dtype, paddle.float32)
        self.assertEqual(out3.dtype, paddle.float32)

    def test_pir(self):
        with paddle.pir_utils.IrGuard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                white_list = paddle.amp.white_list()
                black_list = paddle.amp.black_list()
                self.check_if_op_in_list(
                    self.default_black_list, black_list["float16"]["O2"]
                )
                self.check_if_op_not_in_list(
                    ['log', 'elementwise_add'], white_list
                )
                with paddle.amp.auto_cast(
                    custom_white_list={'elementwise_add'}
                ):
                    out1 = paddle.rand([2, 3]) + paddle.rand([2, 3])
                    out2 = out1.mean()
                    out3 = paddle.log(out2)
                self.check_if_op_not_in_list(
                    ['log', 'elementwise_add'], white_list
                )
                self.assertEqual(out1.dtype, core.DataType.FLOAT16)
                self.assertEqual(out2.dtype, core.DataType.FLOAT32)
                self.assertEqual(out3.dtype, core.DataType.FLOAT32)

    def test_apis(self):
        def _run_check_dtype():
            fp16_lists.check_amp_dtype(dtype="int64")

        self.assertRaises(ValueError, _run_check_dtype)

        for vartype in [core.VarDesc.VarType.FP16, core.VarDesc.VarType.BF16]:
            self.assertEqual(
                fp16_lists.get_low_precision_vartype(vartype), vartype
            )
        self.assertEqual(
            fp16_lists.get_low_precision_vartype("float16"),
            core.VarDesc.VarType.FP16,
        )
        self.assertEqual(
            fp16_lists.get_low_precision_vartype("bfloat16"),
            core.VarDesc.VarType.BF16,
        )

        def _run_get_vartype():
            fp16_lists.get_low_precision_vartype(dtype="int64")

        self.assertRaises(ValueError, _run_get_vartype)

        for dtype in ["float16", "bfloat16"]:
            self.assertEqual(
                fp16_lists.get_low_precision_dtypestr(dtype), dtype
            )
        if paddle.framework.use_pir_api():
            self.assertEqual(
                fp16_lists.get_low_precision_dtypestr(core.DataType.FLOAT16),
                "float16",
            )
            self.assertEqual(
                fp16_lists.get_low_precision_dtypestr(core.DataType.BFLOAT16),
                "bfloat16",
            )
        else:
            self.assertEqual(
                fp16_lists.get_low_precision_dtypestr(
                    core.VarDesc.VarType.FP16
                ),
                "float16",
            )
            self.assertEqual(
                fp16_lists.get_low_precision_dtypestr(
                    core.VarDesc.VarType.BF16
                ),
                "bfloat16",
            )

        def _run_get_dtypestr():
            fp16_lists.get_low_precision_dtypestr(dtype="int64")

        self.assertRaises(ValueError, _run_get_dtypestr)


if __name__ == "__main__":
    unittest.main()
