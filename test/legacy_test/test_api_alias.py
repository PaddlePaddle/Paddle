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
import importlib
import unittest

import paddle


class TestUtilsAttrError(unittest.TestCase):
    def test_error(self):
        with self.assertRaises(AttributeError):
            type(paddle.utils.nonexist)


class TestAlias(unittest.TestCase):
    utils_data_aliases = [
        (
            'paddle.io.Dataset',
            'paddle.utils.data.Dataset',
            'paddle.utils.data.dataset.Dataset',
        ),
        (
            'paddle.io.ChainDataset',
            'paddle.utils.data.ChainDataset',
            'paddle.utils.data.dataset.ChainDataset',
        ),
        (
            'paddle.io.ConcatDataset',
            'paddle.utils.data.ConcatDataset',
            'paddle.utils.data.dataset.ConcatDataset',
        ),
        (
            'paddle.io.IterableDataset',
            'paddle.utils.data.IterableDataset',
            'paddle.utils.data.dataset.IterableDataset',
        ),
        (
            'paddle.io.Sampler',
            'paddle.utils.data.Sampler',
            'paddle.utils.data.sampler.Sampler',
        ),
        (
            'paddle.io.SequenceSampler',
            'paddle.utils.data.SequentialSampler',
            'paddle.utils.data.sampler.SequentialSampler',
        ),
        (
            'paddle.io.Subset',
            'paddle.utils.data.Subset',
            'paddle.utils.data.dataset.Subset',
        ),
        (
            'paddle.io.get_worker_info',
            'paddle.utils.data.get_worker_info',
            'paddle.utils.data.dataloader.get_worker_info',
            'paddle.utils.data._utils.worker.get_worker_info',
        ),
        (
            'paddle.io.random_split',
            'paddle.utils.data.random_split',
            'paddle.utils.data.dataset.random_split',
        ),
        (
            'paddle.io.dataloader.collate.default_collate_fn',
            'paddle.utils.data.default_collate',
            'paddle.utils.data.dataloader.default_collate',
            'paddle.utils.data._utils.collate.default_collate',
        ),
        (
            'paddle.io.BatchSampler',
            'paddle.utils.data.BatchSampler',
            'paddle.utils.data.sampler.BatchSampler',
        ),
        (
            'paddle.io.RandomSampler',
            'paddle.utils.data.RandomSampler',
            'paddle.utils.data.sampler.RandomSampler',
        ),
        (
            'paddle.io.TensorDataset',
            'paddle.utils.data.TensorDataset',
            'paddle.utils.data.dataset.TensorDataset',
        ),
    ]
    optim_module_names = [
        'adadelta',
        'adagrad',
        'adam',
        'adamax',
        'adamw',
        'asgd',
        'lamb',
        'lbfgs',
        'lr',
        'momentum',
        'muon',
        'nadam',
        'optimizer',
        'radam',
        'rmsprop',
        'rprop',
        'sgd',
    ]
    optim_api_names = [
        'Adadelta',
        'Adagrad',
        'Adam',
        'Adamax',
        'AdamW',
        'ASGD',
        'Lamb',
        'LBFGS',
        'Momentum',
        'Muon',
        'NAdam',
        'Optimizer',
        'RAdam',
        'RMSProp',
        'Rprop',
        'SGD',
    ]
    optim_submodule_apis = [
        ('adadelta', 'Adadelta'),
        ('adagrad', 'Adagrad'),
        ('adam', 'Adam'),
        ('adamax', 'Adamax'),
        ('adamw', 'AdamW'),
        ('asgd', 'ASGD'),
        ('lamb', 'Lamb'),
        ('lbfgs', 'LBFGS'),
        ('momentum', 'Momentum'),
        ('muon', 'Muon'),
        ('nadam', 'NAdam'),
        ('optimizer', 'Optimizer'),
        ('radam', 'RAdam'),
        ('rmsprop', 'RMSProp'),
        ('rprop', 'Rprop'),
        ('sgd', 'SGD'),
    ]
    distribution_submodule_apis = [
        ('distribution', 'Distribution'),
        ('normal', 'Normal'),
    ]

    def assert_api_alias(self, canonical_path, *alias_paths):
        canonical_module_path, canonical_name = canonical_path.rsplit('.', 1)
        canonical_module = importlib.import_module(canonical_module_path)
        canonical_api = getattr(canonical_module, canonical_name)

        for alias_path in alias_paths:
            alias_module_path, alias_name = alias_path.rsplit('.', 1)
            alias_module = importlib.import_module(alias_module_path)
            self.assertIs(canonical_api, getattr(alias_module, alias_name))

    def assert_module_alias(self, canonical_path, alias_path):
        self.assertIs(
            importlib.import_module(canonical_path),
            importlib.import_module(alias_path),
        )

    def test_utils_data_api_alias(self):
        for canonical_path, *alias_paths in self.utils_data_aliases:
            self.assert_api_alias(canonical_path, *alias_paths)

    def test_optim_module_alias(self):
        self.assert_module_alias('paddle.optimizer', 'paddle.optim')
        for name in self.optim_module_names:
            self.assert_module_alias(
                f'paddle.optimizer.{name}', f'paddle.optim.{name}'
            )

    def test_optim_api_alias(self):
        for name in self.optim_api_names:
            self.assert_api_alias(
                f'paddle.optimizer.{name}', f'paddle.optim.{name}'
            )

        for module_name, api_name in self.optim_submodule_apis:
            self.assert_api_alias(
                f'paddle.optimizer.{module_name}.{api_name}',
                f'paddle.optim.{module_name}.{api_name}',
            )

    def test_distributions_module_alias(self):
        self.assert_module_alias('paddle.distribution', 'paddle.distributions')

    def test_distributions_api_alias(self):
        for module_name, api_name in self.distribution_submodule_apis:
            self.assert_api_alias(
                f'paddle.distribution.{module_name}.{api_name}',
                f'paddle.distributions.{module_name}.{api_name}',
            )


if __name__ == "__main__":
    unittest.main()
