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
import unittest

import paddle


class TestUtilsAttrError(unittest.TestCase):
    def test_error(self):
        with self.assertRaises(AttributeError):
            type(paddle.utils.nonexist)


class TestAlias(unittest.TestCase):
    def setUp(self):
        self.api_map = [
            (
                paddle.io.Dataset,
                paddle.utils.data.Dataset,
                paddle.utils.data.dataset.Dataset,
                None,
            ),
            (
                paddle.io.ChainDataset,
                paddle.utils.data.ChainDataset,
                paddle.utils.data.dataset.ChainDataset,
                None,
            ),
            (
                paddle.io.ConcatDataset,
                paddle.utils.data.ConcatDataset,
                paddle.utils.data.dataset.ConcatDataset,
                None,
            ),
            (
                paddle.io.IterableDataset,
                paddle.utils.data.IterableDataset,
                paddle.utils.data.dataset.IterableDataset,
                None,
            ),
            (
                paddle.io.Sampler,
                paddle.utils.data.Sampler,
                paddle.utils.data.sampler.Sampler,
                None,
            ),
            (
                paddle.io.SequenceSampler,
                paddle.utils.data.SequentialSampler,
                paddle.utils.data.sampler.SequentialSampler,
                None,
            ),
            (
                paddle.io.Subset,
                paddle.utils.data.Subset,
                paddle.utils.data.dataset.Subset,
                None,
            ),
            (
                paddle.io.get_worker_info,
                paddle.utils.data.get_worker_info,
                paddle.utils.data.dataloader.get_worker_info,
                paddle.utils.data._utils.worker.get_worker_info,
            ),
            (
                paddle.io.random_split,
                paddle.utils.data.random_split,
                paddle.utils.data.dataset.random_split,
                None,
            ),
            (
                paddle.io.dataloader.collate.default_collate_fn,
                paddle.utils.data.default_collate,
                paddle.utils.data.dataloader.default_collate,
                paddle.utils.data._utils.collate.default_collate,
            ),
            (
                paddle.io.BatchSampler,
                paddle.utils.data.BatchSampler,
                paddle.utils.data.sampler.BatchSampler,
                None,
            ),
            (
                paddle.io.RandomSampler,
                paddle.utils.data.RandomSampler,
                paddle.utils.data.sampler.RandomSampler,
                None,
            ),
            (
                paddle.io.TensorDataset,
                paddle.utils.data.TensorDataset,
                paddle.utils.data.dataset.TensorDataset,
                None,
            ),
            (
                paddle.optimizer.Adadelta,
                paddle.optim.Adadelta,
                paddle.optim.adadelta.Adadelta,
                None,
            ),
            (
                paddle.optimizer.Adagrad,
                paddle.optim.Adagrad,
                paddle.optim.adagrad.Adagrad,
                None,
            ),
            (
                paddle.optimizer.Adam,
                paddle.optim.Adam,
                paddle.optim.adam.Adam,
                None,
            ),
            (
                paddle.optimizer.Adamax,
                paddle.optim.Adamax,
                paddle.optim.adamax.Adamax,
                None,
            ),
            (
                paddle.optimizer.AdamW,
                paddle.optim.AdamW,
                paddle.optim.adamw.AdamW,
                None,
            ),
            (
                paddle.optimizer.ASGD,
                paddle.optim.ASGD,
                paddle.optim.asgd.ASGD,
                None,
            ),
            (
                paddle.optimizer.LBFGS,
                paddle.optim.LBFGS,
                paddle.optim.lbfgs.LBFGS,
                None,
            ),
            (
                paddle.optimizer.Muon,
                paddle.optim.Muon,
                paddle.optim.muon.Muon,
                None,
            ),
            (
                paddle.optimizer.NAdam,
                paddle.optim.NAdam,
                paddle.optim.nadam.NAdam,
                None,
            ),
            (
                paddle.optimizer.Optimizer,
                paddle.optim.Optimizer,
                paddle.optim.optimizer.Optimizer,
                None,
            ),
            (
                paddle.optimizer.RAdam,
                paddle.optim.RAdam,
                paddle.optim.radam.RAdam,
                None,
            ),
            (
                paddle.optimizer.RMSProp,
                paddle.optim.RMSProp,
                paddle.optim.rmsprop.RMSProp,
                None,
            ),
            (
                paddle.optimizer.Rprop,
                paddle.optim.Rprop,
                paddle.optim.rprop.Rprop,
                None,
            ),
            (
                paddle.optimizer.SGD,
                paddle.optim.SGD,
                paddle.optim.sgd.SGD,
                None,
            ),
        ]

    def test_compatibility(self):
        for pairs in self.api_map:
            self.assertTrue(pairs[0], pairs[1])
            self.assertTrue(pairs[0], pairs[2])
            if pairs[3] is not None:
                self.assertTrue(pairs[0], pairs[3])


if __name__ == "__main__":
    unittest.main()
