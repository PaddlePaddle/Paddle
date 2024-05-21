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
    not core.is_compiled_with_cuda(), "Only support GPU in CUDA mode."
)
class TestBlhaGetMaxLenOp(unittest.TestCase):
    def setUp(self):
        self.name = "TestBlhaGetMaxLenOpDynamic"
        self.place = paddle.CUDAPlace(0)
        self.batch_size = 10
        self.test_encoder_data = np.random.randint(1, 100, size=self.batch_size)
        self.test_encoder_data_res = paddle.to_tensor(
            np.max(self.test_encoder_data), "int32"
        )
        self.test_decoder_data = np.random.randint(1, 100, size=self.batch_size)
        self.test_decoder_data_res = paddle.to_tensor(
            np.max(self.test_decoder_data), "int32"
        )
        self.seq_lens_encoder = paddle.to_tensor(
            self.test_encoder_data,
            "int32",
        )
        self.seq_lens_decoder = paddle.to_tensor(
            self.test_decoder_data,
            "int32",
        )
        self.batch_size_tensor = paddle.ones([self.batch_size])

    def test_dynamic_api(self):
        paddle.disable_static()
        max_enc_len_this_time, max_dec_len_this_time = blha_get_max_len(
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.batch_size_tensor,
        )
        assert (
            max_enc_len_this_time == self.test_encoder_data_res
            and max_dec_len_this_time == self.test_decoder_data_res
        )

    def test_static_api(self):
        paddle.enable_static()
        max_enc_len_this_time, max_dec_len_this_time = blha_get_max_len(
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.batch_size_tensor,
        )
        assert (
            max_enc_len_this_time == self.test_encoder_data_res
            and max_dec_len_this_time == self.test_decoder_data_res
        )


if __name__ == '__main__':
    unittest.main()
