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

# The GPU unique and sparse CSR kernels use thrust without stream sync.
# User-created streams are non-blocking, so any work these kernels put on
# the legacy default stream would race with the thrust calls. Run them on
# a user stream and compare with numpy.

import unittest

import numpy as np

import paddle


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(), "only test with CUDA stream"
)
class TestThrustNoSyncOnUserStream(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        paddle.set_device('gpu')
        self.stream = paddle.device.Stream()
        self.rng = np.random.default_rng(2026)

    def run_on_stream(self, func, *args):
        paddle.device.synchronize()
        with paddle.device.stream_guard(self.stream):
            out = func(*args)
        self.stream.synchronize()
        return out

    def test_unique_return_inverse(self):
        x_np = self.rng.integers(0, 1 << 20, size=[1 << 22], dtype=np.int64)
        x = paddle.to_tensor(x_np)
        ref_out, ref_index, ref_inverse, ref_counts = np.unique(
            x_np, return_index=True, return_inverse=True, return_counts=True
        )
        for _ in range(3):
            out, index, inverse, counts = self.run_on_stream(
                lambda t: paddle.unique(
                    t,
                    return_index=True,
                    return_inverse=True,
                    return_counts=True,
                ),
                x,
            )
            np.testing.assert_array_equal(out.numpy(), ref_out)
            np.testing.assert_array_equal(inverse.numpy(), ref_inverse)
            np.testing.assert_array_equal(counts.numpy(), ref_counts)
            np.testing.assert_array_equal(x_np[index.numpy()], ref_out)

    def test_unique_axis_return_inverse(self):
        x_np = self.rng.integers(0, 64, size=[1 << 16, 2], dtype=np.int64)
        x = paddle.to_tensor(x_np)
        ref_out, ref_inverse, ref_counts = np.unique(
            x_np, return_inverse=True, return_counts=True, axis=0
        )
        out, inverse, counts = self.run_on_stream(
            lambda t: paddle.unique(
                t, return_inverse=True, return_counts=True, axis=0
            ),
            x,
        )
        np.testing.assert_array_equal(out.numpy(), ref_out)
        np.testing.assert_array_equal(
            inverse.numpy().reshape(-1), ref_inverse.reshape(-1)
        )
        np.testing.assert_array_equal(counts.numpy(), ref_counts)

    def gen_masked(self, shape):
        x_np = self.rng.standard_normal(shape).astype(np.float32)
        mask_np = self.rng.standard_normal(shape).astype(np.float32)
        mask_np[self.rng.random(shape) < 0.7] = 0
        return x_np, mask_np

    def check_mask_as_csr(self, shape):
        x_np, mask_np = self.gen_masked(shape)
        x = paddle.to_tensor(x_np)
        mask = paddle.to_tensor(mask_np).to_sparse_csr()
        out = self.run_on_stream(paddle.sparse.mask_as, x, mask)
        np.testing.assert_allclose(
            out.to_dense().numpy(), x_np * (mask_np != 0)
        )

    def test_mask_as_csr_2d(self):
        self.check_mask_as_csr([1024, 512])

    def test_mask_as_csr_3d(self):
        self.check_mask_as_csr([16, 512, 256])

    def test_csr_to_coo_3d(self):
        _, mask_np = self.gen_masked([16, 512, 256])
        csr = paddle.to_tensor(mask_np).to_sparse_csr()
        coo = self.run_on_stream(lambda t: t.to_sparse_coo(3), csr)
        np.testing.assert_array_equal(
            coo.indices().numpy(), np.stack(np.nonzero(mask_np))
        )
        np.testing.assert_array_equal(
            coo.values().numpy(), mask_np[np.nonzero(mask_np)]
        )


if __name__ == '__main__':
    unittest.main()
