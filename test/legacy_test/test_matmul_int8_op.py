#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import tempfile
import unittest

import numpy as np
from test_sparse_attention_op import get_cuda_version

import paddle
from paddle.base import core

paddle.disable_static()


# TODO: verify the requirments of CUDA ARCH
@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11060,
    "MatmulInt8 requires CUDA >= 11.6",
)
class TestMatmulInt8(unittest.TestCase):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (8, 64)
        self.y_shape = (64, 64)
        self.trans_x = False
        self.trans_y = True

    def setUp(self):
        self.config()
        self.input_a_np = np.random.randint(-127, 127, self.x_shape).astype(
            'int32'
        )
        self.input_b_np = np.random.randint(-127, 127, self.y_shape).astype(
            'int32'
        )
        self.input_a = paddle.to_tensor(self.input_a_np, dtype=self.dtype)
        self.input_b = paddle.to_tensor(self.input_b_np, dtype=self.dtype)

        if self.trans_x:
            if self.input_a_np.ndim == 1:
                self.input_a_np = self.input_a_np.reshape(
                    (self.input_a_np.size,)
                )
            elif self.input_a_np.ndim == 2:
                self.input_a_np = self.input_a_np.T
            else:
                dim = list(range(len(self.input_a_np.shape)))
                dim[-1], dim[len(self.input_a_np.shape) - 2] = (
                    dim[len(self.input_a_np.shape) - 2],
                    dim[-1],
                )
                self.input_a_np = np.transpose(self.input_a_np, tuple(dim))
        if self.trans_y:
            if self.input_b_np.ndim == 1:
                self.input_b_np = self.input_b_np.reshape(
                    (self.input_b_np.size,)
                )
            elif self.input_b_np.ndim == 2:
                self.input_b_np = self.input_b_np.T
            else:
                dim = list(range(len(self.input_b_np.shape)))
                dim[-1], dim[len(self.input_b_np.shape) - 2] = (
                    dim[len(self.input_b_np.shape) - 2],
                    dim[-1],
                )
                self.input_b_np = np.transpose(self.input_b_np, tuple(dim))

    def get_reference_out(self):
        out = np.matmul(self.input_a_np, self.input_b_np)
        return out

    def get_op_out(self):
        out = paddle._C_ops.matmul(
            self.input_a, self.input_b, self.trans_x, self.trans_y
        )
        return out.numpy()

    def test_matmul_int8(self):
        out_real = self.get_op_out()
        out_expect = self.get_reference_out()
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )


class TestMatmulInt8Op2(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (100,)
        self.y_shape = (1, 3, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatmulInt8Op3(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (4,)
        self.y_shape = (1, 1, 4, 100)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op4(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (100,)
        self.y_shape = (1, 2, 100, 4)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op5(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (1, 1, 100, 4)
        self.y_shape = (100,)
        self.trans_x = True
        self.trans_y = False


class TestMatmulInt8Op6(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (1, 2, 104, 4)
        self.y_shape = (104,)
        self.trans_x = True
        self.trans_y = False


class TestMatmulInt8Op7(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (1, 2, 4, 100)
        self.y_shape = (100,)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op8(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (1, 1, 4, 100)
        self.y_shape = (1, 1, 100, 4)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op9(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (1, 1, 4, 100)
        self.y_shape = (2, 1, 8, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatmulInt8Op10(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (1, 1, 24, 4)
        self.y_shape = (1, 2, 4, 24)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op11(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (2, 1, 4, 100)
        self.y_shape = (1, 1, 100, 4)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op12(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (2, 1, 4, 24)
        self.y_shape = (1, 1, 4, 24)
        self.trans_x = True
        self.trans_y = False


class TestMatmulInt8Op13(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (2, 2, 12, 12)
        self.y_shape = (2, 2, 12, 12)
        self.trans_x = True
        self.trans_y = False


class TestMatmulInt8Op14(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (3, 1, 8, 8)
        self.y_shape = (1, 2, 8, 8)
        self.trans_x = True
        self.trans_y = False


class TestMatmulInt8Op15(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (3, 1, 8, 8)
        self.y_shape = (1, 2, 8, 8)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op16(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = 100
        self.y_shape = (1, 2, 2, 100, 4)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op17(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (2, 4, 100)
        self.y_shape = 100
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op18(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (4096, 1024)
        self.y_shape = (4096, 1024)
        self.trans_x = False
        self.trans_y = True

    def setUp(self):
        self.matmul_int8_search_config = tempfile.NamedTemporaryFile(mode='w+')

        matmul_int8_search_config_content = (
            "1,1024,4096,21,0,0,15,5,6,24,0,0.00856064\n"
            "1,2048,8192,21,0,0,15,2,0,24,0,0.0108355\n"
            "2,1024,4096,21,0,0,15,6,0,18,0,0.00855072\n"
            "2,2048,8192,21,0,0,15,16,3,23,0,0.011007\n"
            "3,1024,4096,21,0,0,15,2,0,24,0,0.00856992\n"
            "3,2048,8192,21,0,0,15,2,0,24,0,0.0108426\n"
            "4,1024,4096,21,0,0,15,4,0,18,0,0.0086016\n"
            "4,2048,8192,21,0,0,15,4,5,18,0,0.0109056\n"
            "8,1024,4096,21,0,0,15,4,5,24,0,0.00874304\n"
            "8,2048,8192,21,0,0,15,12,0,23,0,0.0113786\n"
            "12,1024,4096,21,0,0,15,16,5,23,0,0.0085504\n"
            "12,2048,8192,21,0,0,15,4,5,24,0,0.0118294\n"
            "16,1024,4096,21,0,0,15,6,6,23,0,0.0085504\n"
            "16,2048,8192,21,0,0,15,4,5,24,0,0.0118886\n"
            "32,1024,4096,21,0,0,15,32,3,18,0,0.00864256\n"
            "32,2048,8192,21,0,0,15,16,5,23,0,0.0125747\n"
            "48,1024,4096,21,0,0,15,2,3,18,0,0.0087344\n"
            "48,2048,8192,21,0,0,15,6,3,23,0,0.0134963\n"
            "64,1024,4096,21,0,0,15,12,0,24,0,0.00903168\n"
            "64,2048,8192,21,0,0,18,32,5,18,0,0.0140778\n"
            "96,1024,4096,21,0,0,15,2,6,23,0,0.0103629\n"
            "96,2048,8192,21,0,0,17,5,0,21,0,0.0172032\n"
            "128,1024,4096,21,0,0,15,4,5,24,0,0.0105498\n"
            "128,2048,8192,21,0,0,18,32,0,21,0,0.0179302\n"
            "160,1024,4096,21,0,0,18,4,0,18,0,0.0116931\n"
            "160,2048,8192,21,0,0,20,2,3,17,0,0.0220877\n"
            "192,1024,4096,21,0,0,18,8,3,18,0,0.0115712\n"
            "192,2048,8192,21,0,0,20,8,5,17,0,0.0223437\n"
            "224,1024,4096,21,0,0,17,5,3,18,0,0.0126874\n"
            "224,2048,8192,21,0,0,20,8,6,17,0,0.0231117\n"
            "256,1024,4096,21,0,0,17,8,3,18,0,0.0131379\n"
            "256,2048,8192,21,0,0,20,2,5,17,0,0.0239411\n"
        )

        self.matmul_int8_search_config.write(matmul_int8_search_config_content)
        self.matmul_int8_search_config.seek(0)

        paddle.set_flags({'FLAGS_enable_blaslt_global_search': 1})
        paddle.set_flags(
            {
                'FLAGS_cublaslt_device_best_config': self.matmul_int8_search_config.name
            }
        )
        super().setUp()

    def tearDown(self):
        paddle.set_flags({'FLAGS_enable_blaslt_global_search': 0})
        paddle.set_flags({'FLAGS_cublaslt_device_best_config': ''})
        self.matmul_int8_search_config.close()


class TestMatmulInt8OpBroadcast1(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (3, 1, 12, 12)
        self.y_shape = (1, 2, 12, 12)
        self.trans_x = True
        self.trans_y = True


class TestMatmulInt8OpBroadcast2(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (3, 1, 12, 12)
        self.y_shape = (1, 2, 12, 12)
        self.trans_x = False
        self.trans_y = True


if __name__ == '__main__':
    unittest.main()
