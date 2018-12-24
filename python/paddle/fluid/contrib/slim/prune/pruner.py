# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import collections
from .... import layers

__all__ = ['Pruner', 'MagnitudePruner', 'RatioPruner']


class Pruner(object):
    """
    Base class of all pruners.
    """

    def __init__(self):
        pass

    def prune(self, param):
        pass


class MagnitudePruner(Pruner):
    """
    Pruner used to pruning a parameter by threshold.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def prune(self, param, threshold=None):
        if threshold is None:
            thres = layers.fill_constant(
                shape=[1], dtype='float32', value=self.threshold)
        else:
            thres = threshold
        zeros_mask = layers.less_than(x=param, y=thres)
        return zeros_mask


class StructurePruner(Pruner):
    """
    Pruner used to pruning parameters by groups.
    """

    def __init__(self, ratios, group_dims, criterions):
        self.ratios = ratios
        self.group_dims = group_dims
        self.criterions = criterions

    def prune(self, params):
        params = params if isinstance(params,
                                      collections.Iterable) else [params]
        masks = []
        for param in params:
            name = param.name
            ratio = self.ratios[name] if name in self.ratios else self.ratios[
                '*']
            group_dim = self.group_dims[
                name] if name in self.group_dims else self.group_dims['*']
            criterion = self.criterions[
                name] if name in self.criterions else self.criterions['*']
            if criterion == 'l1_norm':
                l1_norms = fluid.layers.reduce_sum(
                    fluid.layers.abs(param), dim=group_dim, keep_dim=True)
                tmp = fluid.layers.reshape(l1_norms, shape=[-1])
                tmp = fluid.layers.topk(tmp, k=tmp.shape[0] * (1 - ratio))
                threhold = tmp[-1]
                zero_mask = l1_norms > threhold
                masks.append(zero_mask)
            else:
                raise NotImplementedError(
                    "criterion {} is not implemented!".format(criterion))
        return masks


class RatioPruner(Pruner):
    """
    Pruner used to pruning a parameter by ratio.
    """

    def __init__(self, ratios=None):
        """
        Args:
            ratios: dict with pair (paramer_name, pruned_ratio). 
        """
        self.ratios = ratios

    def prune(self, param, ratio=None):
        """
        Args:
            ratio: `ratio=40%` means pruning (1 - 40%) weights to zero.
        """
        if ratio is None:
            rat = self.ratios[
                param.name] if param.name in self.ratios else self.ratios['*']
        else:
            rat = ratio
        if rat < 1.0:
            k = max(int(rat * np.prod(param.shape)), 1)
            param_vec = layers.reshape(x=param, shape=[1, -1])
            param_topk, _ = layers.topk(param_vec, k=k)
            threshold = layers.slice(
                param_topk, axes=[1], starts=[-1], ends=[k])
            threshold = layers.reshape(x=threshold, shape=[1])
            zeros_mask = layers.less_than(x=param, y=threshold)
        else:
            zeros_mask = layers.ones(param.shape)
        return zeros_mask
