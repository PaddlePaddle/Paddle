# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.static import InputSpec
from paddle.fluid.framework import core, convert_np_dtype_to_dtype_


class TestInputSpec(unittest.TestCase):
    def test_default(self):
        tensor_spec = InputSpec([3, 4])
        self.assertEqual(tensor_spec.dtype,
                         convert_np_dtype_to_dtype_('float32'))
        self.assertEqual(tensor_spec.name, None)

    def test_from_tensor(self):
        x_bool = fluid.layers.fill_constant(shape=[1], dtype='bool', value=True)
        bool_spec = InputSpec.from_tensor(x_bool)
        self.assertEqual(bool_spec.dtype, x_bool.dtype)
        self.assertEqual(bool_spec.shape, x_bool.shape)
        self.assertEqual(bool_spec.name, x_bool.name)

        bool_spec2 = InputSpec.from_tensor(x_bool, name='bool_spec')
        self.assertEqual(bool_spec2.name, bool_spec2.name)

    def test_from_numpy(self):
        x_numpy = np.ones([10, 12])
        x_np_spec = InputSpec.from_numpy(x_numpy)
        self.assertEqual(x_np_spec.dtype,
                         convert_np_dtype_to_dtype_(x_numpy.dtype))
        self.assertEqual(x_np_spec.shape, x_numpy.shape)
        self.assertEqual(x_np_spec.name, None)

        x_numpy2 = np.array([1, 2, 3, 4]).astype('int64')
        x_np_spec2 = InputSpec.from_numpy(x_numpy2, name='x_np_int64')
        self.assertEqual(x_np_spec2.dtype,
                         convert_np_dtype_to_dtype_(x_numpy2.dtype))
        self.assertEqual(x_np_spec2.shape, x_numpy2.shape)
        self.assertEqual(x_np_spec2.name, 'x_np_int64')

    def test_shape_with_none(self):
        tensor_spec = InputSpec([None, 4, None], dtype='int8', name='x_spec')
        self.assertEqual(tensor_spec.dtype, convert_np_dtype_to_dtype_('int8'))
        self.assertEqual(tensor_spec.name, 'x_spec')
        self.assertEqual(tensor_spec.shape, (-1, 4, -1))

    def test_shape_raise_error(self):
        # 1. shape should only contain int and None.
        with self.assertRaises(ValueError):
            tensor_spec = InputSpec(['None', 4, None], dtype='int8')

        # 2. shape should be type `list` or `tuple`
        with self.assertRaises(TypeError):
            tensor_spec = InputSpec(4, dtype='int8')

        # 3. len(shape) should be greater than 0.
        with self.assertRaises(ValueError):
            tensor_spec = InputSpec([], dtype='int8')

    def test_batch_and_unbatch(self):
        tensor_spec = InputSpec([10])
        # insert batch_size
        batch_tensor_spec = tensor_spec.batch(16)
        self.assertEqual(batch_tensor_spec.shape, (16, 10))

        # unbatch
        unbatch_spec = batch_tensor_spec.unbatch()
        self.assertEqual(unbatch_spec.shape, (10, ))

        # 1. `unbatch` requires len(shape) > 1
        with self.assertRaises(ValueError):
            unbatch_spec.unbatch()

        # 2. `batch` requires len(batch_size) == 1
        with self.assertRaises(ValueError):
            tensor_spec.batch([16, 12])

        # 3. `batch` requires type(batch_size) == int
        with self.assertRaises(TypeError):
            tensor_spec.batch('16')

    def test_eq_and_hash(self):
        tensor_spec_1 = InputSpec([10, 16], dtype='float32')
        tensor_spec_2 = InputSpec([10, 16], dtype='float32')
        tensor_spec_3 = InputSpec([10, 16], dtype='float32', name='x')
        tensor_spec_4 = InputSpec([16], dtype='float32', name='x')

        # override ``__eq__`` according to [shape, dtype, name]
        self.assertTrue(tensor_spec_1 == tensor_spec_2)
        self.assertTrue(tensor_spec_1 != tensor_spec_3)  # different name
        self.assertTrue(tensor_spec_3 != tensor_spec_4)  # different shape

        # override ``__hash__``  according to [shape, dtype]
        self.assertTrue(hash(tensor_spec_1) == hash(tensor_spec_2))
        self.assertTrue(hash(tensor_spec_1) == hash(tensor_spec_3))
        self.assertTrue(hash(tensor_spec_3) != hash(tensor_spec_4))


if __name__ == '__main__':
    unittest.main()
