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
import functools
import warnings

import paddle
from paddle.distribution.beta import Beta
from paddle.distribution.categorical import Categorical
from paddle.distribution.dirichlet import Dirichlet
from paddle.distribution.distribution import Distribution
from paddle.distribution.exponential_family import ExponentialFamily
from paddle.distribution.normal import Normal
from paddle.distribution.uniform import Uniform
from paddle.fluid.framework import _non_static_mode, in_dygraph_mode

__all__ = ["register_kl", "kl_divergence"]

_REGISTER_TABLE = {}


def kl_divergence(p, q):
    r"""
    Kullback-Leibler divergence between distribution p and q.

    .. math::

        KL(p||q) = \int p(x)log\frac{p(x)}{q(x)} \mathrm{d}x

    Args:
        p (Distribution): ``Distribution`` object. Inherits from the Distribution Base class.
        q (Distribution): ``Distribution`` object. Inherits from the Distribution Base class.

    Returns:
        Tensor, Batchwise KL-divergence between distribution p and q.

    Examples:

        .. code-block:: python

            import paddle

            p = paddle.distribution.Beta(alpha=0.5, beta=0.5)
            q = paddle.distribution.Beta(alpha=0.3, beta=0.7)

            print(paddle.distribution.kl_divergence(p, q))
            # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [0.21193528])

    """
    return _dispatch(type(p), type(q))(p, q)


def register_kl(cls_p, cls_q):
    """Decorator for register a KL divergence implemention function.

    The ``kl_divergence(p, q)`` function will search concrete implemention
    functions registered by ``register_kl``, according to multi-dispatch pattern.
    If an implemention function is found, it will return the result, otherwise,
    it will raise ``NotImplementError`` exception. Users can register
    implemention funciton by the decorator.

    Args:
        cls_p (Distribution): The Distribution type of Instance p. Subclass derived from ``Distribution``.
        cls_q (Distribution): The Distribution type of Instance q. Subclass derived from ``Distribution``.

    Examples:
        .. code-block:: python

            import paddle

            @paddle.distribution.register_kl(paddle.distribution.Beta, paddle.distribution.Beta)
            def kl_beta_beta():
                pass # insert implementation here
    """
    if (not issubclass(cls_p, Distribution)
            or not issubclass(cls_q, Distribution)):
        raise TypeError('cls_p and cls_q must be subclass of Distribution')

    def decorator(f):
        _REGISTER_TABLE[cls_p, cls_q] = f
        return f

    return decorator


def _dispatch(cls_p, cls_q):
    """Multiple dispatch into concrete implement function"""

    # find all matched super class pair of p and q
    matchs = [(super_p, super_q) for super_p, super_q in _REGISTER_TABLE
              if issubclass(cls_p, super_p) and issubclass(cls_q, super_q)]
    if not matchs:
        raise NotImplementedError

    left_p, left_q = min(_Compare(*m) for m in matchs).classes
    right_p, right_q = min(_Compare(*reversed(m)) for m in matchs).classes

    if _REGISTER_TABLE[left_p, left_q] is not _REGISTER_TABLE[right_p, right_q]:
        warnings.warn(
            'Ambiguous kl_divergence({}, {}). Please register_kl({}, {})'.
            format(cls_p.__name__, cls_q.__name__, left_p.__name__,
                   right_q.__name__), RuntimeWarning)

    return _REGISTER_TABLE[left_p, left_q]


@functools.total_ordering
class _Compare(object):

    def __init__(self, *classes):
        self.classes = classes

    def __eq__(self, other):
        return self.classes == other.classes

    def __le__(self, other):
        for cls_x, cls_y in zip(self.classes, other.classes):
            if not issubclass(cls_x, cls_y):
                return False
            if cls_x is not cls_y:
                break
        return True


@register_kl(Beta, Beta)
def _kl_beta_beta(p, q):
    return ((q.alpha.lgamma() + q.beta.lgamma() + (p.alpha + p.beta).lgamma()) -
            (p.alpha.lgamma() + p.beta.lgamma() + (q.alpha + q.beta).lgamma()) +
            ((p.alpha - q.alpha) * p.alpha.digamma()) +
            ((p.beta - q.beta) * p.beta.digamma()) +
            (((q.alpha + q.beta) - (p.alpha + p.beta)) *
             (p.alpha + p.beta).digamma()))


@register_kl(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(p, q):
    return (
        (p.concentration.sum(-1).lgamma() - q.concentration.sum(-1).lgamma()) -
        ((p.concentration.lgamma() - q.concentration.lgamma()).sum(-1)) +
        (((p.concentration - q.concentration) *
          (p.concentration.digamma() -
           p.concentration.sum(-1).digamma().unsqueeze(-1))).sum(-1)))


@register_kl(Categorical, Categorical)
def _kl_categorical_categorical(p, q):
    return p.kl_divergence(q)


@register_kl(Normal, Normal)
def _kl_normal_normal(p, q):
    return p.kl_divergence(q)


@register_kl(Uniform, Uniform)
def _kl_uniform_uniform(p, q):
    return p.kl_divergence(q)


@register_kl(ExponentialFamily, ExponentialFamily)
def _kl_expfamily_expfamily(p, q):
    """Compute kl-divergence using `Bregman divergences <https://www.lix.polytechnique.fr/~nielsen/EntropyEF-ICIP2010.pdf>`_
    """
    if not type(p) == type(q):
        raise NotImplementedError

    p_natural_params = []
    for param in p._natural_parameters:
        param = param.detach()
        param.stop_gradient = False
        p_natural_params.append(param)

    q_natural_params = q._natural_parameters

    p_log_norm = p._log_normalizer(*p_natural_params)

    try:
        if _non_static_mode():
            p_grads = paddle.grad(p_log_norm,
                                  p_natural_params,
                                  create_graph=True)
        else:
            p_grads = paddle.static.gradients(p_log_norm, p_natural_params)
    except RuntimeError as e:
        raise TypeError(
            "Cann't compute kl_divergence({cls_p}, {cls_q}) use bregman divergence. Please register_kl({cls_p}, {cls_q})."
            .format(cls_p=type(p).__name__, cls_q=type(q).__name__)) from e

    kl = q._log_normalizer(*q_natural_params) - p_log_norm
    for p_param, q_param, p_grad in zip(p_natural_params, q_natural_params,
                                        p_grads):
        term = (q_param - p_param) * p_grad
        kl -= _sum_rightmost(term, len(q.event_shape))

    return kl


def _sum_rightmost(value, n):
    return value.sum(list(range(-n, 0))) if n > 0 else value
