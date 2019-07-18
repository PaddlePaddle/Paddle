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

__all__ = ['Pruner', 'StructurePruner']


class Pruner(object):
    """
    Base class of all pruners.
    """

    def __init__(self):
        pass

    def prune(self, param):
        pass


class StructurePruner(Pruner):
    """
    Pruner used to pruning parameters by groups.
    """

    def __init__(self, pruning_axis, criterions):
        """
        Args:
            pruning_axis(dict): The key is the name of parameter to be pruned,
                                '*' means all the parameters.
                                The value is the axis to be used. Given a parameter
                                with shape [3, 4], the result of pruning 50% on aixs 1
                                is a parameter with shape [3, 2].
            criterions(dict): The key is the name of parameter to be pruned,
                              '*' means all the parameters.
                              The value is the criterion used to sort groups for pruning.
                              It only supports 'l1_norm' currently.
        """
        self.pruning_axis = pruning_axis
        self.criterions = criterions

    def cal_pruned_idx(self, name, param, ratio, axis=None):
        """
        Calculate the index to be pruned on axis by given pruning ratio.
        Args:
            name(str): The name of parameter to be pruned.
            param(np.array): The data of parameter to be pruned.
            ratio(float): The ratio to be pruned.
            axis(int): The axis to be used for pruning given parameter.
                       If it is None, the value in self.pruning_axis will be used.
                       default: None.
        Returns:
            list<int>: The indexes to be pruned on axis.
        """
        criterion = self.criterions[
            name] if name in self.criterions else self.criterions['*']
        if axis is None:
            assert self.pruning_axis is not None, "pruning_axis should set if axis is None."
            axis = self.pruning_axis[
                name] if name in self.pruning_axis else self.pruning_axis['*']
        prune_num = int(round(param.shape[axis] * ratio))
        reduce_dims = [i for i in range(len(param.shape)) if i != axis]
        if criterion == 'l1_norm':
            criterions = np.sum(np.abs(param), axis=tuple(reduce_dims))
        pruned_idx = criterions.argsort()[:prune_num]
        return pruned_idx

    def prune_tensor(self, tensor, pruned_idx, pruned_axis, lazy=False):
        """
        Pruning a array by indexes on given axis.
        Args:
            tensor(numpy.array): The target array to be pruned.
            pruned_idx(list<int>): The indexes to be pruned.
            pruned_axis(int): The axis of given array to be pruned on. 
            lazy(bool): True means setting the pruned elements to zero.
                        False means remove the pruned elements from memory.
                        default: False.
        Returns:
            numpy.array: The pruned array.
        """
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
