# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import warnings
import logging
import numpy as np


class Converter(object):
    """
    Parameters Merge and Slice

    Args:
        params_dict(dict): parameters' value of all ranks that to be converted. 
                           key is param's name(str), value is all ranks' data(list(numpy.ndarray))
        pre_strategy(dict): parameters' dist_attr of last training process.
        cur_strategy(dict): parameters' dist_attr of current rank training process.
    """

    def __init__(self, params_dict, pre_strategy, cur_strategy):
        self._params_dict = params_dict
        self._pre_strategy = pre_strategy
        self._cur_strategy = cur_strategy

    @property
    def params_dict(self):
        return self._params_dict

    @property
    def pre_strategy(self):
        return self._pre_strategy

    @property
    def cur_strategy(self):
        return self._cur_strategy

    @params_dict.setter
    def params_dict(self, params_dict):
        if not params_dict:
            raise ValueError("'params_dict' is None, "
                             "the parameters to be converted cannot be None.")
        if not isinstance(params_dict, dict):
            raise TypeError(
                "The type of 'params_dict' should be 'dict', but got {}.".
                format(str(type(params_dict))))
        self._params_dict = params_dict

    @pre_strategy.setter
    def pre_strategy(self, pre_strategy):
        if not pre_strategy:
            raise ValueError("'pre_strategy' is None, "
                             "there are not parameter in pre process.")
        if not isinstance(pre_strategy, dict):
            raise TypeError("The type of 'pre_strategy' should be 'dict', "
                            "but got '{}'.".format(str(type(pre_strategy))))
        self._pre_strategy = pre_strategy

    @cur_strategy.setter
    def cur_strategy(self, cur_strategy):
        if not cur_strategy:
            warnings.warn("'cur_strategy' is None, "
                          "there are not parameter in cur process")
        if not isinstance(cur_strategy, dict):
            raise TypeError("The type of 'cur_strategy' should be 'dict', "
                            "but got '{}'.".format(str(type(cur_strategy))))
        self._cur_strategy = cur_strategy

    def convert(self, prefix_match=False):
        params_dict = {}
        # the name which is in cur_process but not in pre_process
        param_not_in_pre = []
        # the name which is in pre_process but not in cur_process
        param_not_in_cur = []
        # the name which is in strategy but not in ckpt files
        param_not_in_ckpt = []
        logging.info("Start to merge and slice parameters.")
        for param_name in self._cur_strategy:
            if param_name not in self._pre_strategy:
                param_not_in_pre.append(param_name)
                continue
            if param_name not in self._params_dict:
                param_not_in_ckpt.append(param_name)
                continue

            param_list = self._params_dict[param_name]
            pre_dist_attr = self._pre_strategy[param_name]
            cur_dist_attr = self._cur_strategy[param_name]
            params_dict[param_name] = Converter.merge_and_slice(
                param_list, pre_dist_attr, cur_dist_attr)

        for param_name in self._pre_strategy:
            if param_name not in self._cur_strategy:
                param_not_in_cur.append(param_name)

        if prefix_match:
            params_dict, param_match_with_pre, param_match_with_cur = self.convert_with_prefix_match(
                params_dict, param_not_in_pre, param_not_in_cur)
        else:
            params_dict, param_match_with_pre, param_match_with_cur = params_dict, [], []

        if param_not_in_pre:
            warnings.warn(
                "parameters '{}' are not found in last training strategy."
                .format(
                    str(set(param_not_in_pre) - set(param_match_with_pre))))
        if param_not_in_cur:
            warnings.warn(
                "parameters '{}' are not found in current training strategy."
                .format(
                    str(set(param_not_in_cur) - set(param_match_with_cur))))
        if param_not_in_ckpt:
            warnings.warn(
                "parameters '{}' are found in pre_strategy, but are not found"
                "in checkpoint files, please check your checkpoint files."
                .format(str(param_not_in_ckpt)))

        return params_dict

    def convert_with_prefix_match(self, params_dict, param_not_in_pre,
                                  param_not_in_cur):
        # the name which in cur_process and can match with pre_process
        param_match_with_pre = []
        # the name which in pre_process and can match with cur_process
        param_match_with_cur = []
        for cur_name in param_not_in_pre:
            prefix_name = cur_name
            while prefix_name.find("_") != -1:
                prefix_name = prefix_name[:prefix_name.rfind("_")]
                for pre_name in param_not_in_cur:
                    if prefix_name in pre_name:
                        # 'cur_name' of cur_process can match with 'pre_name' of pre_process
                        pre_param_list = self._params_dict[pre_name]
                        pre_dist_attr = self._pre_strategy[pre_name]
                        cur_dist_attr = self._cur_strategy[cur_name]
                        params_dict[cur_name] = Converter.merge_and_slice(
                            pre_param_list, pre_dist_attr, cur_dist_attr)
                        logging.info("param {} is placed with param {}".format(
                            cur_name, pre_name))
                        param_match_with_pre.append(cur_name)
                        param_match_with_cur.append(pre_name)
                        break
                break

        return params_dict, param_match_with_pre, param_match_with_cur

    @staticmethod
    def merge_and_slice(param_list, pre_dist_attr, cur_dist_attr):
        """
        Merge parameters with previous dist_attr and slice parameters with current dist_attr

        Returns:
            param(numpy.narray): a parameters's value of current rank.
        """
        assert isinstance(param_list, list)
        assert all(isinstance(p, np.ndarray) for p in param_list)

        if pre_dist_attr == cur_dist_attr:
            # skip merge and slice param
            rank_id = paddle.distributed.get_rank()
            index = cur_dist_attr["process_group"].index(rank_id)
            param = param_list[index]
        else:
            pre_dims_mapping = pre_dist_attr["dims_mapping"]
            cur_dims_mapping = cur_dist_attr["dims_mapping"]
            if len(set(pre_dims_mapping)) > 1 or -1 not in pre_dims_mapping:
                # merge param
                param = Converter.merge_with_dist_attr(param_list,
                                                       pre_dist_attr)
            else:
                # skip merge param
                param = param_list[0]

            if len(set(cur_dims_mapping)) > 1 or -1 not in cur_dims_mapping:
                # slice param
                param = Converter.slice_with_dist_attr(param, cur_dist_attr)

        return param

    @staticmethod
    def merge_with_dist_attr(param_list, dist_attr):
        """ Merge param with distributed attribute """
        from .reshard import _compute_complete_shape, _compute_partition_index

        dims_mapping = dist_attr["dims_mapping"]
        process_shape = dist_attr["process_shape"]
        process_group = dist_attr["process_group"]
        # get the complete shape of the parameter
        complete_shape = _compute_complete_shape(param_list[0].shape,
                                                 process_shape, dims_mapping)
        # merge the parameter with dist_attr
        partition_param_list = []
        merged_partiton = []
        for process in process_group:
            partition_index = _compute_partition_index(
                process, complete_shape, dims_mapping, process_shape,
                process_group)
            index = process_group.index(process)
            if partition_index not in merged_partiton:
                merged_partiton.append(partition_index)
                Converter.merge(partition_param_list, param_list[index],
                                partition_index, complete_shape)

        assert len(partition_param_list) == 1 or not partition_param_list, \
            "Fail to merge parameter"
        complete_param = partition_param_list[0][0]
        return complete_param

    @staticmethod
    def slice_with_dist_attr(param, dist_attr):
        """ Slice parameter with distributed attribute """
        dims_mapping = dist_attr["dims_mapping"]
        process_shape = dist_attr["process_shape"]
        process_group = dist_attr["process_group"]
        # slice the parameter with dist_attr
        partition_index_list = Converter._get_split_indices(
            param.shape, dims_mapping, process_shape, process_group)
        sliced_param_list = Converter.split(param, partition_index_list,
                                            len(partition_index_list))
        # get the current parameter's index in sliced_param_list
        rank_id = paddle.distributed.get_rank()
        sliced_param_index = Converter._get_sliced_index(
            rank_id, param.shape, dims_mapping, process_shape, process_group)
        sliced_param = sliced_param_list[sliced_param_index]
        return sliced_param

    @staticmethod
    def merge(partition_param_list, param, partition_index, complete_shape):
        """
        Merge partitial parameters to a complete one.

        Returns:
            None

        Examples:
            .. code-block:: python

                import numpy as np
                partition_param_list = [(np.array([[[1.11, 1.12]]]), [[0,1],[0,1],[0,2]])]
                param = np.array([[[1.13, 1.14]]])
                partition_index = [[0,1],[0,1],[2,4]]

                _merge_parameter(partition_param_list, param, partition_index)
                # partition_param_list: [(np.array([[[1.11, 1.12, 1.13, 1.14]]]), [[0,1],[0,1],[0,4]])]
        """
        from .reshard import _compute_concat_info

        if len(partition_param_list) == 1:
            is_complete_data = True
            for idx, item in enumerate(partition_param_list[0][1]):
                if item[0] != 0 or item[1] != complete_shape[idx]:
                    is_complete_data = False
                    break
            if is_complete_data:
                return

        if not partition_param_list:
            partition_param_list.append((param, partition_index))
        else:
            i = 0
            while i < len(partition_param_list):
                concat_axis, first_order, new_partition = _compute_concat_info(
                    partition_param_list[i][1], partition_index)
                if concat_axis != -1:
                    if first_order == 0:
                        new_param = np.concatenate(
                            (partition_param_list[i][0], param),
                            axis=concat_axis)
                    else:
                        new_param = np.concatenate(
                            (param, partition_param_list[i][0]),
                            axis=concat_axis)

                    partition_param_list.pop(i)
                    Converter.merge(partition_param_list, new_param,
                                    new_partition, complete_shape)
                    break
                i += 1

    @staticmethod
    def split(complete_param, partition_index_list, length):
        """
        Slice a complete parameter.

        Returns:
            sliced_param_list(list): sliced parameters with 'partition_index_list'

        Examples:
            .. code-block:: python

                import numpy as np
                complete_param = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
                rank = 2
                complete_shape = [1, 1, 6]
                dims_mapping = [-1, -1, 0]
                process_shape = [3]
                process_group = [0, 1, 2]

                sliced_param_list = split(complete_param, [[], [], [2, 4]], 3)
                # [array([[[1.11, 1.12]]]), array([[[1.13, 1.14]]]), array([[[1.15, 1.16]]])]
        """
        sliced_param_list = []
        axis = len(complete_param.shape) - length
        sliced_param = np.split(
            complete_param, partition_index_list[axis], axis=axis)
        if length == 1:
            return sliced_param
        for param in sliced_param:
            sliced_param_list.extend(
                Converter.split(param, partition_index_list, length - 1))
        return sliced_param_list

    @staticmethod
    def _get_split_indices(complete_shape, dims_mapping, process_shape,
                           process_group):
        """
        Get split indices of every dimension.

        Returns:
            split_indices_list(list): the split indices of every dimension of the parameter

        Examples:
            .. code-block:: python

                import numpy as np
                complete_param = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
                complete_shape = [1, 1, 6]
                dims_mapping = [-1, -1, 0]
                process_shape = [3]
                process_group = [0, 1, 2]

                index = _get_split_indices(complete_shape, dims_mapping, process_shape, process_group)
                # index: [[], [], [2, 4]]
        """
        from .reshard import _compute_partition_index

        split_indices_list = []
        for process in process_group:
            partition_index = _compute_partition_index(
                process, complete_shape, dims_mapping, process_shape,
                process_group)
            if split_indices_list:
                for dim in range(len(partition_index)):
                    split_indices_list[dim].extend(partition_index[dim])
            else:
                split_indices_list = partition_index
        split_indices_list = list(
            map(lambda x, y: list(set(x) - set([y]) - set([0])),
                split_indices_list, complete_shape))
        split_indices_list = [sorted(x) for x in split_indices_list]
        return split_indices_list

    @staticmethod
    def _get_sliced_index(rank_id, complete_shape, dims_mapping, process_shape,
                          process_group):
        """
        Get sliced_param's index of current rank in all sliced parameters list.

        Returns:
            sliced_param_index(int): the index of sliced param in sliced_param_list

        Examples:
            .. code-block:: python

                import numpy as np
                complete_param = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
                rank = 2
                complete_shape = [1, 1, 6]
                dims_mapping = [-1, -1, 0]
                process_shape = [3]
                process_group = [0, 1, 2]

                slice_param = _slice_parameter(complete_param, [[], [], [2, 4]], 3)
                # slice_param: 
                # [array([[[1.11, 1.12]]]), array([[[1.13, 1.14]]]), array([[[1.15, 1.16]]])]

                index = _get_sliced_index(rank, complete_shape, dims_mapping
                                                process_shape, process_group)
                # index: 2
        """
        from .reshard import _compute_partition_index

        partition_index = _compute_partition_index(
            rank_id, complete_shape, dims_mapping, process_shape, process_group)
        sliced_index = 0
        for i, shape in enumerate(complete_shape):
            if dims_mapping[i] == -1:
                slice_shape = shape
            else:
                slice_shape = shape // process_shape[dims_mapping[i]]
            if shape == 1:
                index = 0
            else:
                index = (partition_index[i][0] + 1) // slice_shape
            sliced_index = sliced_index * (shape // slice_shape) + index
        return sliced_index
