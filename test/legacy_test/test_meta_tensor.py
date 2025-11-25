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

import unittest

import paddle
from paddle.base.libpaddle import IrMetaTensor, IrTensor
from paddle.static import MetaTensor


class TestIrTensor(unittest.TestCase):
    def test_basic_get_set(self):
        ir_tensor = IrTensor()

        ir_tensor.set_shape([4, 8192, 768])
        self.assertEqual(ir_tensor.shape, [4, 8192, 768])

        ir_tensor.set_dtype('bfloat16')
        self.assertEqual(ir_tensor.dtype, paddle.bfloat16)
        ir_tensor.set_dtype(paddle.uint8)
        self.assertEqual(ir_tensor.dtype, paddle.uint8)

    def test_eq(self):
        x_ir_meta = IrTensor()
        y_ir_meta = IrTensor()
        self.assertEqual(x_ir_meta, y_ir_meta)
        x_ir_meta.set_shape([4, 8192])
        y_ir_meta.set_shape([4, 8192])
        self.assertEqual(x_ir_meta, y_ir_meta)
        x_ir_meta.set_shape([4, 8193])
        self.assertNotEqual(x_ir_meta, y_ir_meta)
        y_ir_meta = IrTensor(x_ir_meta)
        self.assertEqual(x_ir_meta, y_ir_meta)


class TestIrMetaTensor(unittest.TestCase):
    def test_basic_get_set(self):
        ir_tensor = IrTensor()
        ir_meta_tensor = IrMetaTensor(ir_tensor)

        shape = [4, 8192, 768]
        ir_meta_tensor.set_shape(shape)
        self.assertEqual(ir_tensor.shape, shape)
        self.assertEqual(ir_meta_tensor.shape, shape)

        ir_meta_tensor.set_dtype('bfloat16')
        self.assertEqual(ir_tensor.dtype, paddle.bfloat16)
        self.assertEqual(ir_meta_tensor.dtype, paddle.bfloat16)
        ir_meta_tensor.set_dtype(paddle.uint8)
        self.assertEqual(ir_tensor.dtype, paddle.uint8)
        self.assertEqual(ir_meta_tensor.dtype, paddle.uint8)


def infer_meta_fn(x_meta: MetaTensor, y_meta: MetaTensor):
    z_meta = MetaTensor()
    z_meta.set_shape([x_meta.shape[0], y_meta.shape[-1]])
    if x_meta.dtype == paddle.bfloat16 or x_meta.dtype == paddle.float16:
        z_meta.set_dtype("float32")
    else:
        z_meta.set_dtype(x_meta.dtype)
    return z_meta


class TestMetaTensor(unittest.TestCase):
    def test_basic_get_set(self):
        meta_tensor = MetaTensor()

        meta_tensor.set_shape([4, 8192, 768])
        self.assertEqual(meta_tensor.shape, [4, 8192, 768])

        meta_tensor.set_dtype('bfloat16')
        self.assertEqual(meta_tensor.dtype, paddle.bfloat16)
        meta_tensor.set_dtype(paddle.uint8)
        self.assertEqual(meta_tensor.dtype, paddle.uint8)

    def test_eq(self):
        x_meta = MetaTensor()
        y_meta = MetaTensor()
        self.assertEqual(x_meta, y_meta)
        x_meta.set_shape([4, 8192])
        y_meta.set_shape([4, 8192])
        self.assertEqual(x_meta, y_meta)
        x_meta.set_shape([4])
        self.assertNotEqual(x_meta, y_meta)

    def test_infer_meta(self):
        x_meta = MetaTensor()
        x_meta.set_shape([4, 8192])
        x_meta.set_dtype('bfloat16')
        y_meta = MetaTensor()
        y_meta.set_shape([4, 8192, 768])
        z_meta = infer_meta_fn(x_meta, y_meta)
        self.assertEqual(z_meta.shape, [4, 768])
        self.assertEqual(z_meta.dtype, paddle.float32)


if __name__ == "__main__":
    unittest.main()
