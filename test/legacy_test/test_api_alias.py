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
        import sys

        import paddle.distribution
        import paddle.distribution.normal

        distribution_normal_module = paddle.distribution.normal
        import paddle.distributions
        import paddle.distributions.bernoulli
        import paddle.distributions.beta
        import paddle.distributions.binomial
        import paddle.distributions.categorical
        import paddle.distributions.cauchy
        import paddle.distributions.chi2
        import paddle.distributions.constraint
        import paddle.distributions.continuous_bernoulli
        import paddle.distributions.dirichlet
        import paddle.distributions.distribution
        import paddle.distributions.exponential
        import paddle.distributions.exponential_family
        import paddle.distributions.gamma
        import paddle.distributions.geometric
        import paddle.distributions.gumbel
        import paddle.distributions.independent
        import paddle.distributions.kl
        import paddle.distributions.laplace
        import paddle.distributions.lkj_cholesky
        import paddle.distributions.lognormal
        import paddle.distributions.multinomial
        import paddle.distributions.multivariate_normal
        import paddle.distributions.normal
        import paddle.distributions.poisson
        import paddle.distributions.student_t
        import paddle.distributions.transform
        import paddle.distributions.transformed_distribution
        import paddle.distributions.uniform
        import paddle.distributions.variable
        from paddle import distributions
        from paddle.distribution.normal import Normal as DistributionNormal
        from paddle.distributions import (
            bernoulli,
            beta,
            binomial,
            categorical,
            cauchy,
            chi2,
            constraint,
            continuous_bernoulli,
            dirichlet,
            distribution,
            exponential,
            exponential_family,
            gamma,
            geometric,
            gumbel,
            independent,
            kl,
            laplace,
            lkj_cholesky,
            lognormal,
            multinomial,
            multivariate_normal,
            normal,
            poisson,
            student_t,
            transform,
            transformed_distribution,
            uniform,
            variable,
        )
        from paddle.distributions.normal import Normal

        self.assertIs(paddle.distributions, distributions)
        self.assertIs(
            sys.modules["paddle.distribution.normal"],
            sys.modules["paddle.distributions.normal"],
        )
        self.assertIs(distribution_normal_module, paddle.distribution.normal)
        self.assertIs(distribution_normal_module, paddle.distributions.normal)
        self.assertIs(paddle.distribution.normal, paddle.distributions.normal)
        self.assertIs(DistributionNormal, Normal)
        self.assertIs(paddle.distribution.normal.Normal, DistributionNormal)
        self.assertIs(paddle.distributions.normal.Normal, Normal)
        self.assertIs(
            paddle.distribution.normal.Normal,
            paddle.distributions.normal.Normal,
        )
        self.assertTrue(callable(bernoulli.Bernoulli))
        self.assertTrue(callable(beta.Beta))
        self.assertTrue(callable(binomial.Binomial))
        self.assertTrue(callable(categorical.Categorical))
        self.assertTrue(callable(cauchy.Cauchy))
        self.assertTrue(callable(chi2.Chi2))
        self.assertTrue(callable(continuous_bernoulli.ContinuousBernoulli))
        self.assertTrue(callable(dirichlet.Dirichlet))
        self.assertTrue(callable(distribution.Distribution))
        self.assertTrue(callable(exponential.Exponential))
        self.assertTrue(callable(exponential_family.ExponentialFamily))
        self.assertTrue(callable(gamma.Gamma))
        self.assertTrue(callable(geometric.Geometric))
        self.assertTrue(callable(gumbel.Gumbel))
        self.assertTrue(callable(independent.Independent))
        self.assertTrue(callable(laplace.Laplace))
        self.assertTrue(callable(lkj_cholesky.LKJCholesky))
        self.assertTrue(callable(lognormal.LogNormal))
        self.assertTrue(callable(multinomial.Multinomial))
        self.assertTrue(callable(multivariate_normal.MultivariateNormal))
        self.assertTrue(callable(paddle.distributions.normal.Normal))
        self.assertTrue(callable(distributions.normal.Normal))
        self.assertTrue(callable(normal.Normal))
        self.assertTrue(callable(Normal))
        self.assertTrue(callable(paddle.distribution.normal.Normal))
        self.assertTrue(callable(DistributionNormal))
        self.assertTrue(callable(poisson.Poisson))
        self.assertTrue(callable(student_t.StudentT))
        self.assertTrue(
            callable(transformed_distribution.TransformedDistribution)
        )
        self.assertTrue(callable(uniform.Uniform))
        self.assertTrue(callable(constraint.Constraint))
        self.assertTrue(callable(kl.kl_divergence))
        self.assertTrue(callable(kl.register_kl))
        self.assertTrue(callable(transform.Transform))
        self.assertTrue(callable(variable.Variable))

    def test_random_api_alias(self):
        self.assertIs(paddle.random.initial_seed, paddle.initial_seed)

    def assert_api_map(self, api_map):
        for pairs in api_map:
            for alias in pairs[1:]:
                if alias is not None:
                    self.assertIs(pairs[0], alias)


if __name__ == "__main__":
    unittest.main()
