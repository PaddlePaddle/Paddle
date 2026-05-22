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
    def test_optim_module_alias(self):
        from paddle import optim, optimizer

        self.assertIs(optimizer, optim)

    def test_optim_api_alias(self):
        from paddle import optim, optimizer

        for name in [
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
        ]:
            self.assertIs(getattr(optimizer, name), getattr(optim, name))

    def test_optim_submodule_alias(self):
        import paddle.optim.adadelta as optim_adadelta
        import paddle.optim.adagrad as optim_adagrad
        import paddle.optim.adam as optim_adam
        import paddle.optim.adamax as optim_adamax
        import paddle.optim.adamw as optim_adamw
        import paddle.optim.asgd as optim_asgd
        import paddle.optim.lamb as optim_lamb
        import paddle.optim.lbfgs as optim_lbfgs
        import paddle.optim.lr as optim_lr
        import paddle.optim.momentum as optim_momentum
        import paddle.optim.muon as optim_muon
        import paddle.optim.nadam as optim_nadam
        import paddle.optim.optimizer as optim_optimizer
        import paddle.optim.radam as optim_radam
        import paddle.optim.rmsprop as optim_rmsprop
        import paddle.optim.rprop as optim_rprop
        import paddle.optim.sgd as optim_sgd
        import paddle.optimizer.adadelta as optimizer_adadelta
        import paddle.optimizer.adagrad as optimizer_adagrad
        import paddle.optimizer.adam as optimizer_adam
        import paddle.optimizer.adamax as optimizer_adamax
        import paddle.optimizer.adamw as optimizer_adamw
        import paddle.optimizer.asgd as optimizer_asgd
        import paddle.optimizer.lamb as optimizer_lamb
        import paddle.optimizer.lbfgs as optimizer_lbfgs
        import paddle.optimizer.lr as optimizer_lr
        import paddle.optimizer.momentum as optimizer_momentum
        import paddle.optimizer.muon as optimizer_muon
        import paddle.optimizer.nadam as optimizer_nadam
        import paddle.optimizer.optimizer as optimizer_optimizer
        import paddle.optimizer.radam as optimizer_radam
        import paddle.optimizer.rmsprop as optimizer_rmsprop
        import paddle.optimizer.rprop as optimizer_rprop
        import paddle.optimizer.sgd as optimizer_sgd

        self.assertIs(optimizer_adadelta, optim_adadelta)
        self.assertIs(optimizer_adagrad, optim_adagrad)
        self.assertIs(optimizer_adam, optim_adam)
        self.assertIs(optimizer_adamax, optim_adamax)
        self.assertIs(optimizer_adamw, optim_adamw)
        self.assertIs(optimizer_asgd, optim_asgd)
        self.assertIs(optimizer_lamb, optim_lamb)
        self.assertIs(optimizer_lbfgs, optim_lbfgs)
        self.assertIs(optimizer_lr, optim_lr)
        self.assertIs(optimizer_momentum, optim_momentum)
        self.assertIs(optimizer_muon, optim_muon)
        self.assertIs(optimizer_nadam, optim_nadam)
        self.assertIs(optimizer_optimizer, optim_optimizer)
        self.assertIs(optimizer_radam, optim_radam)
        self.assertIs(optimizer_rmsprop, optim_rmsprop)
        self.assertIs(optimizer_rprop, optim_rprop)
        self.assertIs(optimizer_sgd, optim_sgd)

    def test_distributions_module_alias(self):
        from paddle import distribution, distributions

        self.assertIs(distribution, distributions)

    def test_distributions_api_alias(self):
        from paddle import distribution, distributions

        for name in [
            'Bernoulli',
            'Beta',
            'Binomial',
            'Categorical',
            'Cauchy',
            'Chi2',
            'ContinuousBernoulli',
            'Dirichlet',
            'Distribution',
            'Exponential',
            'ExponentialFamily',
            'Gamma',
            'Geometric',
            'Gumbel',
            'Independent',
            'Laplace',
            'LKJCholesky',
            'LogNormal',
            'Multinomial',
            'MultivariateNormal',
            'Normal',
            'Poisson',
            'StudentT',
            'kl_divergence',
            'register_kl',
            'AbsTransform',
            'AffineTransform',
            'ChainTransform',
            'ExpTransform',
            'IndependentTransform',
            'PowerTransform',
            'ReshapeTransform',
            'SigmoidTransform',
            'SoftmaxTransform',
            'StackTransform',
            'StickBreakingTransform',
            'TanhTransform',
            'Transform',
            'TransformedDistribution',
            'Uniform',
        ]:
            self.assertIs(
                getattr(distribution, name), getattr(distributions, name)
            )

    def test_distributions_submodule_alias(self):
        import paddle.distribution.bernoulli as distribution_bernoulli
        import paddle.distribution.beta as distribution_beta
        import paddle.distribution.binomial as distribution_binomial
        import paddle.distribution.categorical as distribution_categorical
        import paddle.distribution.cauchy as distribution_cauchy
        import paddle.distribution.chi2 as distribution_chi2
        import paddle.distribution.constraint as distribution_constraint
        import paddle.distribution.dirichlet as distribution_dirichlet
        import paddle.distribution.distribution as distribution_distribution
        import paddle.distribution.exponential as distribution_exponential
        import paddle.distribution.gamma as distribution_gamma
        import paddle.distribution.geometric as distribution_geometric
        import paddle.distribution.gumbel as distribution_gumbel
        import paddle.distribution.independent as distribution_independent
        import paddle.distribution.kl as distribution_kl
        import paddle.distribution.laplace as distribution_laplace
        import paddle.distribution.lkj_cholesky as distribution_lkj_cholesky
        import paddle.distribution.lognormal as distribution_lognormal
        import paddle.distribution.multinomial as distribution_multinomial
        import paddle.distribution.normal as distribution_normal
        import paddle.distribution.poisson as distribution_poisson
        import paddle.distribution.student_t as distribution_student_t
        import paddle.distribution.transform as distribution_transform
        import paddle.distribution.uniform as distribution_uniform
        import paddle.distribution.variable as distribution_variable
        import paddle.distributions.bernoulli as distributions_bernoulli
        import paddle.distributions.beta as distributions_beta
        import paddle.distributions.binomial as distributions_binomial
        import paddle.distributions.categorical as distributions_categorical
        import paddle.distributions.cauchy as distributions_cauchy
        import paddle.distributions.chi2 as distributions_chi2
        import paddle.distributions.constraint as distributions_constraint
        import paddle.distributions.dirichlet as distributions_dirichlet
        import paddle.distributions.distribution as distributions_distribution
        import paddle.distributions.exponential as distributions_exponential
        import paddle.distributions.gamma as distributions_gamma
        import paddle.distributions.geometric as distributions_geometric
        import paddle.distributions.gumbel as distributions_gumbel
        import paddle.distributions.independent as distributions_independent
        import paddle.distributions.kl as distributions_kl
        import paddle.distributions.laplace as distributions_laplace
        import paddle.distributions.lkj_cholesky as distributions_lkj_cholesky
        import paddle.distributions.lognormal as distributions_lognormal
        import paddle.distributions.multinomial as distributions_multinomial
        import paddle.distributions.normal as distributions_normal
        import paddle.distributions.poisson as distributions_poisson
        import paddle.distributions.student_t as distributions_student_t
        import paddle.distributions.transform as distributions_transform
        import paddle.distributions.uniform as distributions_uniform
        import paddle.distributions.variable as distributions_variable
        from paddle.distribution import (
            continuous_bernoulli as distribution_continuous_bernoulli,
            exponential_family as distribution_exponential_family,
            multivariate_normal as distribution_multivariate_normal,
            transformed_distribution as distribution_transformed_distribution,
        )
        from paddle.distributions import (
            continuous_bernoulli as distributions_continuous_bernoulli,
            exponential_family as distributions_exponential_family,
            multivariate_normal as distributions_multivariate_normal,
            transformed_distribution as distributions_transformed_distribution,
        )

        self.assertEqual(
            distribution_bernoulli.__file__, distributions_bernoulli.__file__
        )
        self.assertEqual(
            distribution_beta.__file__, distributions_beta.__file__
        )
        self.assertEqual(
            distribution_binomial.__file__, distributions_binomial.__file__
        )
        self.assertEqual(
            distribution_categorical.__file__,
            distributions_categorical.__file__,
        )
        self.assertEqual(
            distribution_cauchy.__file__, distributions_cauchy.__file__
        )
        self.assertEqual(
            distribution_chi2.__file__, distributions_chi2.__file__
        )
        self.assertEqual(
            distribution_constraint.__file__, distributions_constraint.__file__
        )
        self.assertEqual(
            distribution_continuous_bernoulli.__file__,
            distributions_continuous_bernoulli.__file__,
        )
        self.assertEqual(
            distribution_dirichlet.__file__, distributions_dirichlet.__file__
        )
        self.assertEqual(
            distribution_distribution.__file__,
            distributions_distribution.__file__,
        )
        self.assertEqual(
            distribution_exponential.__file__,
            distributions_exponential.__file__,
        )
        self.assertEqual(
            distribution_exponential_family.__file__,
            distributions_exponential_family.__file__,
        )
        self.assertEqual(
            distribution_gamma.__file__, distributions_gamma.__file__
        )
        self.assertEqual(
            distribution_geometric.__file__, distributions_geometric.__file__
        )
        self.assertEqual(
            distribution_gumbel.__file__, distributions_gumbel.__file__
        )
        self.assertEqual(
            distribution_independent.__file__,
            distributions_independent.__file__,
        )
        self.assertEqual(distribution_kl.__file__, distributions_kl.__file__)
        self.assertEqual(
            distribution_laplace.__file__, distributions_laplace.__file__
        )
        self.assertEqual(
            distribution_lkj_cholesky.__file__,
            distributions_lkj_cholesky.__file__,
        )
        self.assertEqual(
            distribution_lognormal.__file__, distributions_lognormal.__file__
        )
        self.assertEqual(
            distribution_multinomial.__file__,
            distributions_multinomial.__file__,
        )
        self.assertEqual(
            distribution_multivariate_normal.__file__,
            distributions_multivariate_normal.__file__,
        )
        self.assertEqual(
            distribution_normal.__file__, distributions_normal.__file__
        )
        self.assertEqual(
            distribution_poisson.__file__, distributions_poisson.__file__
        )
        self.assertEqual(
            distribution_student_t.__file__, distributions_student_t.__file__
        )
        self.assertEqual(
            distribution_transform.__file__, distributions_transform.__file__
        )
        self.assertEqual(
            distribution_transformed_distribution.__file__,
            distributions_transformed_distribution.__file__,
        )
        self.assertEqual(
            distribution_uniform.__file__, distributions_uniform.__file__
        )
        self.assertEqual(
            distribution_variable.__file__, distributions_variable.__file__
        )


if __name__ == "__main__":
    unittest.main()
