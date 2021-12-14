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
    """Compute kl divergence between distribution p and q

    Args:
        p ([type]): [description]
        q ([type]): [description]
    """
    fun = _dispatch(type(p), type(q))

    if fun is NotImplemented:
        raise NotImplementedError

    return fun(p, q)


def _dispatch(cls_p, cls_q):
    """multiple dispatch into concrete implement function"""
    matchs = []
    for super_p, super_q in _REGISTER_TABLE:
        if issubclass(cls_p, super_p) and issubclass(cls_q, super_q):
            matchs.append((super_p, super_q))

    if not matchs:
        return NotImplemented

    left_p, left_q = min(_Compare(*m) for m in matchs).types
    right_p, right_q = min(_Compare(*reversed(m)) for m in matchs).types

    left_fun = _REGISTER_TABLE[left_p, left_q]
    right_fun = _REGISTER_TABLE[right_p, right_q]

    if left_fun is not right_fun:
        warnings.warn(
            'Ambiguous kl_divergence({}, {}). Please register_kl({}, {})'.
            format(cls_p.__name__, cls_q.__name__, left_p.__name__,
                   right_q.__name__), RuntimeWarning)

    return left_fun


@functools.total_ordering
class _Compare(object):
    def __init__(self, *types):
        self.types = types

    def __eq__(self, other):
        return self.types == other.types

    def __le__(self, other):
        for x, y in zip(self.types, other.types):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True


def register_kl(cls_p, cls_q):
    if (not issubclass(cls_p, Distribution) or
            not issubclass(cls_q, Distribution)):
        raise TypeError('cls_p and cls_q must be subclass of Distribution')

    def decorator(f):
        _REGISTER_TABLE[cls_p, cls_q] = f
        return f

    return decorator


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
