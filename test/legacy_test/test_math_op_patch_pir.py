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

import inspect
import unittest

import paddle

paddle.enable_static()


class TestMathOpPatchesPir(unittest.TestCase):
    def test_math_exists(self):
        with paddle.pir_utils.IrGuard():
            a = paddle.to_tensor([[1, 1], [2, 2], [3, 3]])
            self.assertTrue(isinstance(a, paddle.pir.OpResult))
            self.assertTrue(inspect.ismethod(a.dot))
            self.assertTrue(inspect.ismethod(a.logsumexp))
            self.assertTrue(inspect.ismethod(a.multiplex))
            self.assertTrue(inspect.ismethod(a.prod))
            self.assertTrue(inspect.ismethod(a.scale))
            self.assertTrue(inspect.ismethod(a.stanh))
            self.assertTrue(inspect.ismethod(a.add_n))
            self.assertTrue(inspect.ismethod(a.max))
            self.assertTrue(inspect.ismethod(a.maximum))
            self.assertTrue(inspect.ismethod(a.min))
            self.assertTrue(inspect.ismethod(a.minimum))
            self.assertTrue(inspect.ismethod(a.floor_divide))
            self.assertTrue(inspect.ismethod(a.remainder))
            self.assertTrue(inspect.ismethod(a.floor_mod))
            self.assertTrue(inspect.ismethod(a.multiply))
            self.assertTrue(inspect.ismethod(a.inverse))
            self.assertTrue(inspect.ismethod(a.log1p))
            self.assertTrue(inspect.ismethod(a.erf))
            self.assertTrue(inspect.ismethod(a.addmm))
            self.assertTrue(inspect.ismethod(a.clip))
            self.assertTrue(inspect.ismethod(a.trace))
            self.assertTrue(inspect.ismethod(a.kron))
            self.assertTrue(inspect.ismethod(a.isinf))
            self.assertTrue(inspect.ismethod(a.isnan))
            self.assertTrue(inspect.ismethod(a.concat))
            self.assertTrue(inspect.ismethod(a.broadcast_to))
            self.assertTrue(inspect.ismethod(a.scatter_nd_add))
            self.assertTrue(inspect.ismethod(a.scatter_nd))
            self.assertTrue(inspect.ismethod(a.shard_index))
            self.assertTrue(inspect.ismethod(a.chunk))
            self.assertTrue(inspect.ismethod(a.stack))
            self.assertTrue(inspect.ismethod(a.strided_slice))
            self.assertTrue(inspect.ismethod(a.unsqueeze))
            self.assertTrue(inspect.ismethod(a.unstack))
            self.assertTrue(inspect.ismethod(a.argmax))
            self.assertTrue(inspect.ismethod(a.argmin))
            self.assertTrue(inspect.ismethod(a.argsort))
            self.assertTrue(inspect.ismethod(a.masked_select))
            self.assertTrue(inspect.ismethod(a.topk))
            self.assertTrue(inspect.ismethod(a.index_select))
            self.assertTrue(inspect.ismethod(a.nonzero))
            self.assertTrue(inspect.ismethod(a.sort))
            self.assertTrue(inspect.ismethod(a.index_sample))
            self.assertTrue(inspect.ismethod(a.mean))
            self.assertTrue(inspect.ismethod(a.std))
            self.assertTrue(inspect.ismethod(a.numel))
            self.assertTrue(inspect.ismethod(a.asin_))
            self.assertTrue(inspect.ismethod(a.atan2))
            self.assertTrue(inspect.ismethod(a.atanh_))
            self.assertTrue(inspect.ismethod(a.coalesce))
            self.assertTrue(inspect.ismethod(a.diagflat))
            self.assertTrue(inspect.ismethod(a.multinomial))
            self.assertTrue(inspect.ismethod(a.pinv))
            self.assertTrue(inspect.ismethod(a.renorm))
            self.assertTrue(inspect.ismethod(a.renorm_))
            self.assertTrue(inspect.ismethod(a.tan))
            self.assertTrue(inspect.ismethod(a.tan_))
            self.assertTrue(inspect.ismethod(a.tril))
            self.assertTrue(inspect.ismethod(a.tril_))
            self.assertTrue(inspect.ismethod(a.triu))
            self.assertTrue(inspect.ismethod(a.triu_))
            self.assertTrue(inspect.ismethod(a.stft))
            self.assertTrue(inspect.ismethod(a.istft))
            self.assertTrue(inspect.ismethod(a.abs_))
            self.assertTrue(inspect.ismethod(a.acos_))
            self.assertTrue(inspect.ismethod(a.atan_))
            self.assertTrue(inspect.ismethod(a.cos_))
            self.assertTrue(inspect.ismethod(a.cosh_))
            self.assertTrue(inspect.ismethod(a.sin_))
            self.assertTrue(inspect.ismethod(a.sinh_))
            self.assertTrue(inspect.ismethod(a.acosh_))
            self.assertTrue(inspect.ismethod(a.asinh_))
            self.assertTrue(inspect.ismethod(a.diag))
            self.assertTrue(inspect.ismethod(a.eye))
            self.assertTrue(inspect.ismethod(a.linspace))
            self.assertTrue(inspect.ismethod(a.fill_constant))
            self.assertTrue(inspect.ismethod(a.ones))
            self.assertTrue(inspect.ismethod(a.ones_like))
            self.assertTrue(inspect.ismethod(a.zeros))
            self.assertTrue(inspect.ismethod(a.zeros_like))
            self.assertTrue(inspect.ismethod(a.arange))
            self.assertTrue(inspect.ismethod(a.full))
            self.assertTrue(inspect.ismethod(a.full_like))
            self.assertTrue(inspect.ismethod(a.meshgrid))
            self.assertTrue(inspect.ismethod(a.empty))
            self.assertTrue(inspect.ismethod(a.empty_like))
            self.assertTrue(inspect.ismethod(a.complex))
            self.assertTrue(inspect.ismethod(a.eigh))
            self.assertTrue(inspect.ismethod(a.standard_normal))
            self.assertTrue(inspect.ismethod(a.normal))
            self.assertTrue(inspect.ismethod(a.uniform))
            self.assertTrue(inspect.ismethod(a.randn))
            self.assertTrue(inspect.ismethod(a.rand))
            self.assertTrue(inspect.ismethod(a.randint))
            self.assertTrue(inspect.ismethod(a.randint_like))
            self.assertTrue(inspect.ismethod(a.randperm))
            self.assertTrue(inspect.ismethod(a.poisson))
            self.assertTrue(inspect.ismethod(a.searchsorted))
            self.assertTrue(inspect.ismethod(a.set_printoptions))
            self.assertTrue(inspect.ismethod(a.array_length))
            self.assertTrue(inspect.ismethod(a.array_read))
            self.assertTrue(inspect.ismethod(a.array_write))
            self.assertTrue(inspect.ismethod(a.create_array))
            self.assertTrue(inspect.ismethod(a.einsum))


if __name__ == '__main__':
    unittest.main()
