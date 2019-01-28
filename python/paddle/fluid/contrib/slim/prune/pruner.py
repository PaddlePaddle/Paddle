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

__all__ = ['Pruner', 'MagnitudePruner', 'RatioPruner', 'StructurePruner']


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

    def __init__(self, pruning_axis, criterions):
        self.pruning_axis = pruning_axis
        self.criterions = criterions

    def cal_pruned_idx(self, name, param, ratio, axis):
        criterion = self.criterions[
            name] if name in self.criterions else self.criterions['*']
        prune_num = int(round(param.shape[axis] * ratio))
        reduce_dims = [i for i in range(len(param.shape)) if i != axis]
        if criterion == 'l1_norm':
            criterions = np.sum(np.abs(param), axis=tuple(reduce_dims))
        pruned_idx = criterions.argsort()[:prune_num]
        return pruned_idx

    def prune_tensor(self, tensor, pruned_idx, pruned_axis, lazy=False):
        mask = np.zeros(tensor.shape[pruned_axis], dtype=bool)
        mask[pruned_idx] = True

        def func(data):
            return data[~mask]

        def lazy_func(data):
            data[mask] = 0
            return data

        if lazy:
            return np.apply_along_axis(lazy_func, pruned_axis, tensor)
        else:
            return np.apply_along_axis(func, pruned_axis, tensor)


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
