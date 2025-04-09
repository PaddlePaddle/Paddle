# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core
from paddle.incubate.nn.functional import blha_get_max_len


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not core.is_compiled_with_xpu(),
    "Only support XPU or GPU in CUDA mode.",
)
class TestBlhaGetMaxLenOp(unittest.TestCase):
    def setUp(self):
        self.name = "TestBlhaGetMaxLenOpDynamic"
        if paddle.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        elif paddle.device.is_compiled_with_xpu():
            place = paddle.device.XPUPlace(0)
        else:
            raise ValueError("Only support CUDA or XPU Place.")
        self.batch_size = 10
        self.test_encoder_data = np.random.randint(
            1, 100, size=self.batch_size
        ).astype("int32")
        self.test_decoder_data = np.random.randint(
            1, 100, size=self.batch_size
        ).astype("int32")

    def test_dynamic_api(self):
        paddle.disable_static()
        test_encoder_data_res = paddle.to_tensor(
            np.max(self.test_encoder_data), "int32"
        )
        test_decoder_data_res = paddle.to_tensor(
            np.max(self.test_decoder_data), "int32"
        )
        seq_lens_encoder = paddle.to_tensor(
            self.test_encoder_data,
            "int32",
        )
        seq_lens_decoder = paddle.to_tensor(
            self.test_decoder_data,
            "int32",
        )
        batch_size_tensor = paddle.ones([self.batch_size])
        max_enc_len_this_time, max_dec_len_this_time = blha_get_max_len(
            seq_lens_encoder,
            seq_lens_decoder,
            batch_size_tensor,
        )
        assert (
            max_enc_len_this_time == test_encoder_data_res
            and max_dec_len_this_time == test_decoder_data_res
        )

    def test_static_api(self):
        paddle.enable_static()
        test_encoder_data_res = np.max(self.test_encoder_data).astype("int32")
        test_decoder_data_res = np.max(self.test_decoder_data).astype("int32")

        if paddle.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        elif paddle.device.is_compiled_with_xpu():
            place = paddle.device.XPUPlace(0)
        else:
            raise ValueError("Only support CUDA or XPU Place.")

        with paddle.static.program_guard(paddle.static.Program()):
            seq_lens_encoder = paddle.static.data(
                "seq_lens_encoder", self.test_encoder_data.shape, "int32"
            )
            seq_lens_decoder = paddle.static.data(
                "seq_lens_decoder", self.test_decoder_data.shape, "int32"
            )
            batch_size_tensor = paddle.ones([self.batch_size], "int32")
            max_enc_len_this_time, max_dec_len_this_time = blha_get_max_len(
                seq_lens_encoder,
                seq_lens_decoder,
                batch_size_tensor,
            )
            exe = paddle.static.Executor(place)
            res_max_enc_len_this_time, res_max_dec_len_this_time = exe.run(
                feed={
                    "seq_lens_encoder": self.test_encoder_data,
                    "seq_lens_decoder": self.test_decoder_data,
                },
                fetch_list=[max_enc_len_this_time, max_dec_len_this_time],
            )
        assert (
            res_max_enc_len_this_time == test_encoder_data_res
            and res_max_dec_len_this_time == test_decoder_data_res
        )


if __name__ == '__main__':
    unittest.main()
