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

import os
import pathlib
import shutil
import unittest

import numpy as np

import paddle
from paddle.device.cuda.graphs import CUDAGraph


def can_use_cuda_graph():
    return paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm()


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or float(paddle.version.cuda()) < 11.0,
    "only support cuda >= 11.0",
)
class TestCUDAGraphInDygraphMode(unittest.TestCase):
    def setUp(self):
        if can_use_cuda_graph():
            paddle.set_flags(
                {
                    'FLAGS_allocator_strategy': 'auto_growth',
                    'FLAGS_sync_nccl_allreduce': False,
                    'FLAGS_cudnn_deterministic': True,
                    'FLAGS_use_stream_safe_cuda_allocator': False,
                }
            )

    def random_tensor(self, shape):
        return paddle.to_tensor(
            np.random.randint(low=0, high=10, size=shape).astype("float32")
        )

    def test_cuda_graph_dynamic_graph(self):
        if not can_use_cuda_graph():
            return

        shape = [2, 3]
        x = self.random_tensor(shape)
        z = self.random_tensor(shape)

        g = CUDAGraph()
        g.capture_begin()
        y = x + 10
        z.add_(x)
        g.capture_end()

        for _ in range(10):
            z_np_init = z.numpy()
            x_new = self.random_tensor(shape)
            x.copy_(x_new, False)
            g.replay()
            x_np = x_new.numpy()
            y_np = y.numpy()
            z_np = z.numpy()
            self.assertTrue((y_np - x_np == 10).all())
            self.assertTrue((z_np - z_np_init == x_np).all())

        g.reset()

    def test_concat_and_split(self):
        if not can_use_cuda_graph():
            return

        concat_num = 100
        xs = []
        xs_np = []

        for i in range(concat_num):
            x_np = np.random.random(size=[1]).astype(np.float32)
            xs.append(paddle.to_tensor(x_np))
            xs_np.append(x_np)

        graph = CUDAGraph()
        graph.capture_begin()
        y = paddle.concat(xs)
        zs = paddle.split(y, len(xs))
        graph.capture_end()
        graph.replay()

        y_np = y.numpy()
        y_np_expected = np.concatenate(xs_np)
        np.testing.assert_array_equal(y_np, y_np_expected)
        self.assertEqual(len(zs), len(xs_np))
        for i, z in enumerate(zs):
            np.testing.assert_array_equal(z.numpy(), xs_np[i])

        output_dir = f'cuda_graph_dot_{os.getpid()}'
        try:
            graph.print_to_dot_files(pathlib.Path(output_dir))
            graph.reset()
            shutil.rmtree(output_dir)
        except Exception as e:
            msg = str(e)
            sub_msg = "The print_to_dot_files() method is only supported when CUDA version >= 11.3"
            self.assertTrue(sub_msg in msg)
        finally:
            graph.reset()

    def test_dataloader(self):
        if not can_use_cuda_graph():
            return

        class AutoIncDataset(paddle.io.Dataset):
            def __init__(self, n, dtype):
                self.n = n
                self.dtype = dtype

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return np.array([idx]).astype(self.dtype)

        n = 100
        dtype = 'int64'
        dataset = AutoIncDataset(n, dtype)
        data_loader = paddle.io.DataLoader(
            dataset, batch_size=1, num_workers=2, use_buffer_reader=True
        )
        x = None
        y = None

        graph = None
        for i, data in enumerate(data_loader):
            if graph is None:
                x = data
                x = x.cuda()
                graph = CUDAGraph()
                graph.capture_begin()
                y = x * x
                graph.capture_end()
            else:
                x.copy_(data, False)
                x = x.cuda()

            graph.replay()
            actual_x = np.array([[i]]).astype(dtype)
            actual_y = np.array([[i * i]]).astype(dtype)
            np.testing.assert_array_equal(actual_x, x.numpy())
            np.testing.assert_array_equal(actual_y, y.numpy())

    def test_dev_ctx_alloc(self):
        if not can_use_cuda_graph():
            return

        x = paddle.to_tensor([2], dtype='float32')
        graph = CUDAGraph()
        graph.capture_begin()
        y = paddle.cast(x, dtype='float16')
        graph.capture_end()


if __name__ == "__main__":
    unittest.main()
