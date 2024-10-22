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
import unittest

from legacy_test.test_parallel_dygraph_dataparallel import (
    TestMultipleAccelerators,
)

import paddle


class TestHybridPipeParallel(TestMultipleAccelerators):
    def test_hybrid_parallel_pp_layer(self):
        self.run_mnist_2accelerators(
            os.path.abspath('../../legacy_test/hybrid_parallel_pp_layer.py')
        )

    def test_hybrid_parallel_pp_tuple_inputs(self):
        self.run_mnist_2accelerators('hybrid_parallel_pp_embedding.py')

    def test_hybrid_parallel_shared_weight(self):
        self.run_mnist_2accelerators('hybrid_parallel_shared_weight.py')

    def test_pipeline_parallel_amp(self):
        self.run_mnist_2accelerators('hybrid_parallel_pp_amp.py')

    def test_pipeline_parallel_fp16(self):
        self.run_mnist_2accelerators('hybrid_parallel_pp_fp16.py')

    def test_pipeline_parallel_bf16(self):
        self.run_mnist_2accelerators('hybrid_parallel_pp_bf16.py')

    def test_hybrid_parallel_transformer(self):
        self.run_mnist_2accelerators('hybrid_parallel_pp_transformer.py')

    def test_hybrid_parallel_save_load(self):
        self.run_mnist_2accelerators('hybrid_parallel_pp_save_load.py')

    def test_hybrid_parallel_recompute(self):
        self.run_mnist_2accelerators('hybrid_parallel_pp_recompute.py')

    def test_hybrid_parallel_pp_clip_grad(self):
        self.run_mnist_2accelerators('hybrid_parallel_pp_clip_grad.py')

    def test_hybrid_parallel_transformer_unbalanced_data(self):
        self.run_mnist_2accelerators(
            'hybrid_parallel_pp_transformer_unbalanced_data.py'
        )

    def test_hybrid_parallel_pp_return_micro_batch_loss(self):
        self.run_mnist_2accelerators(
            'hybrid_parallel_pp_return_micro_batch_loss.py'
        )

    def test_hybrid_parallel_pp_with_eager_connect(self):
        os.environ["FLAGS_eager_communication_connection"] = "1"
        self.run_mnist_2accelerators(
            'hybrid_parallel_pp_return_micro_batch_loss.py'
        )
        os.environ["FLAGS_eager_communication_connection"] = "0"


class TestFakeMicroDataSet(unittest.TestCase):
    def test_fake_micro_data_set(self):
        import numpy as np

        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        batch_size = 4
        micro_batch_size = 2
        acc_step = 2
        length = 4
        x_data = np.random.randint(0, batch_size, size=[batch_size, length])
        data1 = paddle.to_tensor(x_data)
        data1.stop_gradient = True

        data2 = [
            data1[
                (i * micro_batch_size) : ((i + 1) * micro_batch_size), :
            ].detach()
            for i in range(acc_step)
        ]

        data3 = None

        batch = [(data1, data2, data3), None]

        for micro_batch in FakeMicroDataset(
            batch, True, False, acc_step, micro_batch_size
        ):
            x, y = micro_batch
            self.assertEqual(len(x), 3)
            for e in [x[0], x[1]]:
                self.assertEqual(e.shape[0], micro_batch_size)
                self.assertEqual(e.shape[1], length)
            self.assertTrue(x[2] is None)
            self.assertTrue(y is None)

        # not first stage or last stage
        micro_batches = FakeMicroDataset(
            batch, False, False, acc_step, micro_batch_size
        )
        x, y = micro_batches._load_micro_batch(0)
        self.assertTrue(x is None)
        self.assertTrue(y is None)


if __name__ == "__main__":
    unittest.main()
