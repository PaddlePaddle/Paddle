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

from .beta import Beta
from .categorical import Categorical
from .dirichlet import Dirichlet
from .distribution import Distribution
from .normal import Normal
from .uniform import Uniform

__all__ = ["register_kl", "kl_divergence"]

_REGISTER_TABLE = {}


def kl_divergence(p, q):
    """Kullback-Leibler divergence between distribution p and q.

    .. math::

        KL(p||q) = \int p(x)log\frac{p(x)}{q(x)} \mathrm{d}x

    Args:
        p (Distribution): ``Distribution`` object.
        q (Distribution): ``Distribution`` object.

    Returns:
        Tensor: batchwise KL-divergence between distribution p and q.

    Raises:
        NotImplementedError: can't find register function for KL(p||Q).

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

    Args:
        cls_p(Distribution): subclass derived from ``Distribution``.
        cls_q(Distribution): subclass derived from ``Distribution``.

    Examples:
        .. code-block:: python

            import paddle

            @paddle.distribution.register_kl(paddle.distribution.Beta, paddle.distribution.Beta)
            def kl_beta_beta():
                pass # insert implementation here
    """
    if (not issubclass(cls_p, Distribution) or
            not issubclass(cls_q, Distribution)):
        raise TypeError('cls_p and cls_q must be subclass of Distribution')

    def decorator(f):
        _REGISTER_TABLE[cls_p, cls_q] = f
        return f

    return decorator


def _dispatch(cls_p, cls_q):
    """multiple dispatch into concrete implement function"""

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

    return left_fun


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
            ((p.alpha - q.alpha) * p.alpha.digamma()) + (
                (p.beta - q.beta) * p.beta.digamma()) + (
                    ((q.alpha + q.beta) -
                     (p.alpha + p.beta)) * (p.alpha + p.beta).digamma()))


@register_kl(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(p, q):
    return (
        (p.concentration.sum(-1).lgamma() - q.concentration.sum(-1).lgamma()) -
        ((p.concentration.lgamma() - q.concentration.lgamma()).sum(-1)) + (
            ((p.concentration - q.concentration) *
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


def _lbeta(x, y):
    return x.lgamma() + y.lgamma() - (x + y).lgamma()
