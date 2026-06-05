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
    def test_utils_data_api_alias(self):
        api_map = [
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
        ]
        self.assert_api_map(api_map)

    def test_optimizer_import_usages(self):
        import paddle.optim.adadelta
        import paddle.optim.lr_scheduler
        from paddle import optim
        from paddle.optim import lr_scheduler
        from paddle.optim.adadelta import Adadelta
        from paddle.optim.lr_scheduler import ConstantLR

        self.assertIs(paddle.optim, optim)
        api_map = [
            (
                paddle.optimizer.Adadelta,
                paddle.optim.Adadelta,
                paddle.optim.adadelta.Adadelta,
                Adadelta,
            ),
            (paddle.optimizer.Adagrad, paddle.optim.Adagrad),
            (paddle.optimizer.Adam, paddle.optim.Adam),
            (paddle.optimizer.Adamax, paddle.optim.Adamax),
            (paddle.optimizer.AdamW, paddle.optim.AdamW),
            (paddle.optimizer.ASGD, paddle.optim.ASGD),
            (paddle.optimizer.LBFGS, paddle.optim.LBFGS),
            (paddle.optimizer.Muon, paddle.optim.Muon),
            (paddle.optimizer.NAdam, paddle.optim.NAdam),
            (paddle.optimizer.Optimizer, paddle.optim.Optimizer),
            (paddle.optimizer.RAdam, paddle.optim.RAdam),
            (paddle.optimizer.RMSProp, paddle.optim.RMSProp),
            (paddle.optimizer.Rprop, paddle.optim.Rprop),
            (paddle.optimizer.SGD, paddle.optim.SGD),
            (
                paddle.optimizer.lr.PiecewiseDecay,
                paddle.optim.lr_scheduler.ConstantLR,
                lr_scheduler.ConstantLR,
                ConstantLR,
            ),
        ]
        self.assertIs(paddle.optim.lr_scheduler, lr_scheduler)
        self.assert_api_map(api_map)

    def test_lr_scheduler_api_alias(self):
        import paddle.optim.lr_scheduler
        import paddle.optimizer.lr
        from paddle.optim import lr_scheduler
        from paddle.optim.lr_scheduler import (
            ConstantLR,
            CosineAnnealingLR,
            CosineAnnealingWarmRestarts,
            CyclicLR,
            ExponentialLR,
            LambdaLR,
            LinearLR,
            LRScheduler,
            MultiplicativeLR,
            MultiStepLR,
            OneCycleLR,
            ReduceLROnPlateau,
            StepLR,
        )

        api_map = [
            (
                paddle.optimizer.lr.PiecewiseDecay,
                paddle.optim.lr_scheduler.ConstantLR,
                lr_scheduler.ConstantLR,
                ConstantLR,
            ),
            (
                paddle.optimizer.lr.CosineAnnealingDecay,
                paddle.optim.lr_scheduler.CosineAnnealingLR,
                lr_scheduler.CosineAnnealingLR,
                CosineAnnealingLR,
            ),
            (
                paddle.optimizer.lr.CosineAnnealingWarmRestarts,
                paddle.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                lr_scheduler.CosineAnnealingWarmRestarts,
                CosineAnnealingWarmRestarts,
            ),
            (
                paddle.optimizer.lr.CyclicLR,
                paddle.optim.lr_scheduler.CyclicLR,
                lr_scheduler.CyclicLR,
                CyclicLR,
            ),
            (
                paddle.optimizer.lr.ExponentialDecay,
                paddle.optim.lr_scheduler.ExponentialLR,
                lr_scheduler.ExponentialLR,
                ExponentialLR,
            ),
            (
                paddle.optimizer.lr.LRScheduler,
                paddle.optim.lr_scheduler.LRScheduler,
                lr_scheduler.LRScheduler,
                LRScheduler,
            ),
            (
                paddle.optimizer.lr.LambdaDecay,
                paddle.optim.lr_scheduler.LambdaLR,
                lr_scheduler.LambdaLR,
                LambdaLR,
            ),
            (
                paddle.optimizer.lr.LinearLR,
                paddle.optim.lr_scheduler.LinearLR,
                lr_scheduler.LinearLR,
                LinearLR,
            ),
            (
                paddle.optimizer.lr.MultiStepDecay,
                paddle.optim.lr_scheduler.MultiStepLR,
                lr_scheduler.MultiStepLR,
                MultiStepLR,
            ),
            (
                paddle.optimizer.lr.MultiplicativeDecay,
                paddle.optim.lr_scheduler.MultiplicativeLR,
                lr_scheduler.MultiplicativeLR,
                MultiplicativeLR,
            ),
            (
                paddle.optimizer.lr.OneCycleLR,
                paddle.optim.lr_scheduler.OneCycleLR,
                lr_scheduler.OneCycleLR,
                OneCycleLR,
            ),
            (
                paddle.optimizer.lr.ReduceOnPlateau,
                paddle.optim.lr_scheduler.ReduceLROnPlateau,
                lr_scheduler.ReduceLROnPlateau,
                ReduceLROnPlateau,
            ),
            (
                paddle.optimizer.lr.StepDecay,
                paddle.optim.lr_scheduler.StepLR,
                lr_scheduler.StepLR,
                StepLR,
            ),
        ]
        self.assertIs(paddle.optim.lr_scheduler, lr_scheduler)
        self.assert_api_map(api_map)

    def test_distribution_import_usages(self):
        import importlib
        import sys

        import paddle.distribution
        import paddle.distribution.normal
        import paddle.distributions
        import paddle.distributions.normal
        from paddle import distribution, distributions
        from paddle.distribution import Normal as DistributionNormal
        from paddle.distribution.normal import Normal as DistributionSubNormal
        from paddle.distributions import Normal as DistributionsNormal
        from paddle.distributions.normal import Normal as DistributionsSubNormal

        self.assertIs(paddle.distribution, distribution)
        self.assertIs(paddle.distributions, distributions)
        self.assertIs(paddle.distribution, paddle.distributions)
        self.assertIs(
            sys.modules["paddle.distribution"],
            sys.modules["paddle.distributions"],
        )

        self.assert_distribution_api_aliases()

        submodule_api_map = [
            ("bernoulli", "Bernoulli"),
            ("beta", "Beta"),
            ("binomial", "Binomial"),
            ("categorical", "Categorical"),
            ("cauchy", "Cauchy"),
            ("chi2", "Chi2"),
            ("constraint", "Constraint"),
            ("continuous_bernoulli", "ContinuousBernoulli"),
            ("dirichlet", "Dirichlet"),
            ("distribution", "Distribution"),
            ("exponential", "Exponential"),
            ("exponential_family", "ExponentialFamily"),
            ("gamma", "Gamma"),
            ("geometric", "Geometric"),
            ("gumbel", "Gumbel"),
            ("independent", "Independent"),
            ("laplace", "Laplace"),
            ("lkj_cholesky", "LKJCholesky"),
            ("lognormal", "LogNormal"),
            ("multinomial", "Multinomial"),
            ("multivariate_normal", "MultivariateNormal"),
            ("normal", "Normal"),
            ("poisson", "Poisson"),
            ("student_t", "StudentT"),
            ("transform", "Transform"),
            ("transformed_distribution", "TransformedDistribution"),
            ("uniform", "Uniform"),
            ("variable", "Variable"),
        ]
        for module_name, api_name in submodule_api_map:
            self.assert_distribution_submodule_import(
                importlib, module_name, api_name
            )

        normal_usages = [
            DistributionNormal,
            DistributionsNormal,
            DistributionSubNormal,
            DistributionsSubNormal,
            paddle.distribution.Normal,
            paddle.distributions.Normal,
            paddle.distribution.normal.Normal,
            paddle.distributions.normal.Normal,
        ]
        self.assert_normal_usages_equal(normal_usages)

    def test_random_api_alias(self):
        self.assertIs(paddle.random.initial_seed, paddle.initial_seed)

    def assert_distribution_api_aliases(self):
        api_names = [
            "Bernoulli",
            "Beta",
            "Binomial",
            "Categorical",
            "Cauchy",
            "Chi2",
            "ContinuousBernoulli",
            "Dirichlet",
            "Distribution",
            "Exponential",
            "ExponentialFamily",
            "Gamma",
            "Geometric",
            "Gumbel",
            "Independent",
            "Laplace",
            "LKJCholesky",
            "LogNormal",
            "Multinomial",
            "MultivariateNormal",
            "Normal",
            "Poisson",
            "StudentT",
            "Transform",
            "TransformedDistribution",
            "Uniform",
        ]
        for api_name in api_names:
            self.assertIs(
                getattr(paddle.distribution, api_name),
                getattr(paddle.distributions, api_name),
            )

    def assert_distribution_submodule_import(
        self, importlib, module_name, api_name
    ):
        distribution_module = importlib.import_module(
            f"paddle.distribution.{module_name}"
        )
        distributions_module = importlib.import_module(
            f"paddle.distributions.{module_name}"
        )

        self.assertEqual(
            distribution_module.__file__, distributions_module.__file__
        )
        self.assertTrue(callable(getattr(distribution_module, api_name)))
        self.assertTrue(callable(getattr(distributions_module, api_name)))

    def assert_normal_usages_equal(self, normal_usages):
        expected = self.get_normal_usage_outputs(normal_usages[0])
        for normal in normal_usages[1:]:
            self.assertEqual(normal.__name__, normal_usages[0].__name__)
            actual = self.get_normal_usage_outputs(normal)
            for actual_value, expected_value in zip(actual, expected):
                self.assert_tensor_equal(actual_value, expected_value)

    def get_normal_usage_outputs(self, normal):
        value = paddle.to_tensor([0.25, 1.5], dtype="float32")
        dist = normal([0.0, 1.0], [1.0, 2.0], validate_args=False)
        return (
            dist.mean,
            dist.variance,
            dist.entropy(),
            dist.log_prob(value),
            dist.probs(value),
        )

    def assert_tensor_equal(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(bool(paddle.allclose(actual, expected).item()))

    def assert_api_map(self, api_map):
        for pairs in api_map:
            for alias in pairs[1:]:
                if alias is not None:
                    self.assertIs(pairs[0], alias)


if __name__ == "__main__":
    unittest.main()
