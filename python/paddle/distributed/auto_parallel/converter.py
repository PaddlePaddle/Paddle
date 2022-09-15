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
from ..utils import get_logger


class Converter(object):
    """
    Converter is a class object for auto parallel to convert tensors from
    one parallel strategy to another one. Tensors will merge and slice value
    with their strategy when strategies are different.
    """

    def __init__(self, tensors_dict, pre_strategy, cur_strategy):
        """
        Args:
            tensors_dict(dict): tensors' value of all ranks that to be converted.
                key is tensor's name(str), value is all ranks' data(list(numpy.ndarray))
            pre_strategy(dict): tensors' distributed attribute of last training process.
                key is tensor's name(str), value is tensor's distributed attribute in last
                training process.
            cur_strategy(dict): tensors' distributed attribute of current rank.
                key is tensor's name(str), value is tensor's distributed attribute in current
                rank.
        """
        self._tensors_dict = self._check_tensor_dict(tensors_dict)
        self._pre_strategy = self._check_pre_strategy(pre_strategy)
        self._cur_strategy = self._check_cur_strategy(cur_strategy)
        self._logger = get_logger(logging.INFO)

    def _check_tensor_dict(self, tensors_dict):
        if not tensors_dict:
            raise ValueError("'tensors_dict' is None, "
                             "the tensors to be converted cannot be None.")
        if not isinstance(tensors_dict, dict):
            raise TypeError(
                "The type of 'tensors_dict' should be 'dict', but got '{}'.".
                format(str(type(tensors_dict))))
        return tensors_dict

    def _check_pre_strategy(self, pre_strategy):
        if not pre_strategy:
            raise ValueError("'pre_strategy' is None, "
                             "there are not tensors in pre process.")
        if not isinstance(pre_strategy, dict):
            raise TypeError("The type of 'pre_strategy' should be 'dict', "
                            "but got '{}'.".format(str(type(pre_strategy))))
        return pre_strategy

    def _check_cur_strategy(self, cur_strategy):
        if not cur_strategy:
            warnings.warn("'cur_strategy' is None, "
                          "there are not tensors in cur process")
        if not isinstance(cur_strategy, dict):
            raise TypeError("The type of 'cur_strategy' should be 'dict', "
                            "but got '{}'.".format(str(type(cur_strategy))))
        return cur_strategy

    def convert(self, strict=True):
        """
        Convert tensors

        Args:
            strict(bool): whether to strict convert tensor with tensor's name. If False, it will
            convert tensors by prefix matching. Otherwise, tensors will be converted with
            their name strictly.

        Returns:
            converted tensors(dict)

        Examples:
            .. code-block:: python

                import numpy as np
                complete_tensors = np.arange(4).reshape([2, 2])
                partitial_tensors = np.split(complete_tensors, 2, axis=0)
                name = "tmp_0"
                tensors_dict = {name: partitial_tensors}
                strategy_1 = {
                    name: {
                        "process_shape": [2],
                        "process_group": [0, 1],
                        "dims_mapping": [0, -1]
                    }
                }
                strategy_2 = {
                    name: {
                        "process_shape": [2],
                        "process_group": [0, 1],
                        "dims_mapping": [-1, -1]
                    }
                }
                converter = Converter(tensors_dict, strategy_1, strategy_2)
                result = converter.convert()
                # the result's value is equal to `complete_tensors`
        """
        tensors_dict = {}
        # the name which is in cur_process but not in pre_process
        tensor_not_in_pre = []
        # the name which is in pre_process but not in cur_process
        tensor_not_in_cur = []
        # the name which is in strategy but not in ckpt files
        tensor_not_in_ckpt = []
        self._logger.info("Start to convert tensors.")
        for tensor_name in self._cur_strategy:
            if tensor_name not in self._pre_strategy:
                tensor_not_in_pre.append(tensor_name)
                continue
            if tensor_name not in self._tensors_dict:
                tensor_not_in_ckpt.append(tensor_name)
                continue
            self._pre_name = tensor_name
            self._cur_name = tensor_name
            tensor_list = self._tensors_dict[tensor_name]
            pre_dist_attr = self._pre_strategy[tensor_name]
            cur_dist_attr = self._cur_strategy[tensor_name]
            try:
                tensors_dict[tensor_name] = Converter.merge_and_slice(
                    tensor_list, pre_dist_attr, cur_dist_attr)
            except ValueError as err:
                raise ValueError(
                    "Fail to convert tensor '{}'. ".format(str(tensor_name)) +
                    str(err))

        for tensor_name in self._pre_strategy:
            if tensor_name not in self._cur_strategy:
                tensor_not_in_cur.append(tensor_name)

        if not strict:
            tensors_dict, tensor_match_with_pre, tensor_match_with_cur = self.convert_with_prefix_match(
                tensors_dict, tensor_not_in_pre, tensor_not_in_cur)
        else:
            tensors_dict, tensor_match_with_pre, tensor_match_with_cur = tensors_dict, [], []

        tensor_not_in_pre = set(tensor_not_in_pre) - set(tensor_match_with_pre)
        tensor_not_in_cur = set(tensor_not_in_cur) - set(tensor_match_with_cur)
        if tensor_not_in_pre:
            warnings.warn(
                "tensors [{}] are not found in last training strategy.".format(
                    str(tensor_not_in_pre)))
        if tensor_not_in_cur:
            warnings.warn(
                "tensors [{}] are not found in current training strategy.".
                format(str(tensor_not_in_cur)))
        if tensor_not_in_ckpt:
            warnings.warn(
                "tensors [{}] are found in pre_strategy, but are not found"
                "in checkpoint files, please check your checkpoint files.".
                format(str(tensor_not_in_ckpt)))

        return tensors_dict

    def convert_with_prefix_match(self, tensors_dict, tensor_not_in_pre,
                                  tensor_not_in_cur):
        # the name which in cur_process and can match with pre_process
        tensor_match_with_pre = []
        # the name which in pre_process and can match with cur_process
        tensor_match_with_cur = []
        for cur_name in tensor_not_in_pre:
            prefix_name = cur_name
            while prefix_name.find("_") != -1:
                prefix_name = prefix_name[:prefix_name.rfind("_")]
                for pre_name in tensor_not_in_cur:
                    if prefix_name in pre_name:
                        # 'cur_name' of cur_process can match with 'pre_name' of pre_process
                        self._pre_name = pre_name
                        self._cur_name = cur_name
                        pre_tensor_list = self._tensors_dict[pre_name]
                        pre_dist_attr = self._pre_strategy[pre_name]
                        cur_dist_attr = self._cur_strategy[cur_name]
                        try:
                            tensors_dict[cur_name] = Converter.merge_and_slice(
                                pre_tensor_list, pre_dist_attr, cur_dist_attr)
                        except ValueError as err:
                            raise ValueError(
                                "Fail to convert tensor '{}' by '{}'. ".format(
                                    str(cur_name), str(pre_name)) + str(err))
                        self._logger.info(
                            "tensor [{}] is matched with tensor [{}]".format(
                                cur_name, pre_name))
                        tensor_match_with_pre.append(cur_name)
                        tensor_match_with_cur.append(pre_name)
                        break
                break

        return tensors_dict, tensor_match_with_pre, tensor_match_with_cur

    @staticmethod
    def merge_and_slice(tensor_list, pre_dist_attr, cur_dist_attr):
        """
        Merge tensors with previous dist_attr and slice tensors with current dist_attr

        Returns:
            tensor(numpy.narray): a tensor's value of current rank.
        """
        assert isinstance(tensor_list, list)
        assert all(isinstance(p, np.ndarray) for p in tensor_list)

        if pre_dist_attr == cur_dist_attr:
            # skip merge and slice tensor
            rank_id = paddle.distributed.get_rank()
            index = cur_dist_attr["process_group"].index(rank_id)
            tensor = tensor_list[index]
        else:
            pre_dims_mapping = pre_dist_attr["dims_mapping"]
            cur_dims_mapping = cur_dist_attr["dims_mapping"]
            if len(set(pre_dims_mapping)) > 1 or -1 not in pre_dims_mapping:
                # merge tensor
                tensor = Converter.merge_with_dist_attr(tensor_list,
                                                        pre_dist_attr)
            else:
                # skip merge tensor
                tensor = tensor_list[0]

            if len(set(cur_dims_mapping)) > 1 or -1 not in cur_dims_mapping:
                # slice tensor
                tensor = Converter.slice_with_dist_attr(tensor, cur_dist_attr)

        return tensor

    @staticmethod
    def merge_with_dist_attr(tensor_list, dist_attr):
        """ Merge tensor with distributed attribute """
        from .reshard import Resharder

        dims_mapping = dist_attr["dims_mapping"]
        process_shape = dist_attr["process_shape"]
        process_group = dist_attr["process_group"]
        # get the complete shape of the tensor
        complete_shape = Resharder.compute_complete_shape(
            tensor_list[0].shape, process_shape, dims_mapping)
        # merge the tensor with dist_attr
        partition_tensor_list = []
        merged_partiton = []
        for process in process_group:
            partition_index = Resharder.compute_partition_index(
                process, complete_shape, dims_mapping, process_shape,
                process_group)
            index = process_group.index(process)
            if partition_index not in merged_partiton:
                merged_partiton.append(partition_index)
                Converter.merge(partition_tensor_list, tensor_list[index],
                                partition_index, complete_shape)

        if len(partition_tensor_list) != 1:
            raise ValueError("Fail to merge tensor with dist_attr '{}'.".format(
                str(dist_attr)))
        complete_tensor = partition_tensor_list[0][0]
        return complete_tensor

    @staticmethod
    def slice_with_dist_attr(tensor, dist_attr):
        """ Slice tensor with distributed attribute """
        dims_mapping = dist_attr["dims_mapping"]
        process_shape = dist_attr["process_shape"]
        process_group = dist_attr["process_group"]
        # slice the tensor with dist_attr
        partition_index_list = Converter._get_split_indices(
            tensor.shape, dims_mapping, process_shape, process_group)
        sliced_tensor_list = Converter.split(tensor, partition_index_list,
                                             len(partition_index_list))
        # get the current tensor's index in sliced_tensor_list
        rank_id = paddle.distributed.get_rank()
        sliced_tensor_index = Converter._get_sliced_index(
            rank_id, tensor.shape, dims_mapping, process_shape, process_group)
        if sliced_tensor_index not in range(len(sliced_tensor_list)):
            raise ValueError("Fail to slice tensor with dist_attr '{}'.".format(
                str(dist_attr)))
        sliced_tensor = sliced_tensor_list[sliced_tensor_index]
        return sliced_tensor

    @staticmethod
    def merge(partition_tensor_list, tensor, partition_index, complete_shape):
        """
        Merge partitial tensors to a complete.

        Returns:
            None

        Examples:
            .. code-block:: python

                import numpy as np
                partition_tensor_list = [(np.array([[[1.11, 1.12]]]), [[0,1],[0,1],[0,2]])]
                tensor = np.array([[[1.13, 1.14]]])
                partition_index = [[0,1],[0,1],[2,4]]

                _merge_tensor(partition_tensor_list, tensor, partition_index)
                # partition_tensor_list: [(np.array([[[1.11, 1.12, 1.13, 1.14]]]), [[0,1],[0,1],[0,4]])]
        """
        from .reshard import Resharder

        if len(partition_tensor_list) == 1:
            is_complete_data = True
            for idx, item in enumerate(partition_tensor_list[0][1]):
                if item[0] != 0 or item[1] != complete_shape[idx]:
                    is_complete_data = False
                    break
            if is_complete_data:
                return

        if not partition_tensor_list:
            partition_tensor_list.append((tensor, partition_index))
        else:
            i = 0
            while i < len(partition_tensor_list):
                concat_axis, first_order, new_partition = Resharder.compute_concat_info(
                    partition_tensor_list[i][1], partition_index)
                if concat_axis != -1:
                    if first_order == 0:
                        new_tensor = np.concatenate(
                            (partition_tensor_list[i][0], tensor),
                            axis=concat_axis)
                    else:
                        new_tensor = np.concatenate(
                            (tensor, partition_tensor_list[i][0]),
                            axis=concat_axis)

                    partition_tensor_list.pop(i)
                    Converter.merge(partition_tensor_list, new_tensor,
                                    new_partition, complete_shape)
                    break
                i += 1

    @staticmethod
    def split(complete_tensor, partition_index_list, length):
        """
        Slice a complete tensor.

        Returns:
            sliced_tensor_list(list): sliced tensors with 'partition_index_list'

        Examples:
            .. code-block:: python

                import numpy as np
                complete_tensor = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
                rank = 2
                complete_shape = [1, 1, 6]
                dims_mapping = [-1, -1, 0]
                process_shape = [3]
                process_group = [0, 1, 2]

                sliced_tensor_list = split(complete_tensor, [[], [], [2, 4]], 3)
                # [array([[[1.11, 1.12]]]), array([[[1.13, 1.14]]]), array([[[1.15, 1.16]]])]
        """
        sliced_tensor_list = []
        axis = len(complete_tensor.shape) - length
        sliced_tensor = np.split(complete_tensor,
                                 partition_index_list[axis],
                                 axis=axis)
        if length == 1:
            return sliced_tensor
        for tensor in sliced_tensor:
            sliced_tensor_list.extend(
                Converter.split(tensor, partition_index_list, length - 1))
        return sliced_tensor_list

    @staticmethod
    def _get_split_indices(complete_shape, dims_mapping, process_shape,
                           process_group):
        """
        Get split indices of every dimension.

        Returns:
            split_indices_list(list): the split indices of every dimension of the tensor

        Examples:
            .. code-block:: python

                import numpy as np
                complete_tensor = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
                complete_shape = [1, 1, 6]
                dims_mapping = [-1, -1, 0]
                process_shape = [3]
                process_group = [0, 1, 2]

                index = _get_split_indices(complete_shape, dims_mapping, process_shape, process_group)
                # index: [[], [], [2, 4]]
        """
        from .reshard import Resharder

        split_indices_list = []
        for process in process_group:
            partition_index = Resharder.compute_partition_index(
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
        Get sliced_tensor's index of current rank in all sliced tensors list.

        Returns:
            sliced_tensor_index(int): the index of sliced tensor in sliced_tensor_list

        Examples:
            .. code-block:: python

                import numpy as np
                complete_tensor = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
                rank = 2
                complete_shape = [1, 1, 6]
                dims_mapping = [-1, -1, 0]
                process_shape = [3]
                process_group = [0, 1, 2]

                slice_tensor = _slice_tensor(complete_tensor, [[], [], [2, 4]], 3)
                # slice_tensor:
                # [array([[[1.11, 1.12]]]), array([[[1.13, 1.14]]]), array([[[1.15, 1.16]]])]

                index = _get_sliced_index(rank, complete_shape, dims_mapping
                                                process_shape, process_group)
                # index: 2
        """
        from .reshard import Resharder

        partition_index = Resharder.compute_partition_index(
            rank_id, complete_shape, dims_mapping, process_shape, process_group)
        sliced_index = 0
        for i, shape in enumerate(complete_shape):
            if dims_mapping[i] == -1:
                slice_shape = shape
            else:
                slice_shape = shape // process_shape[dims_mapping[i]]
            if slice_shape == 1:
                index = partition_index[i][0]
            else:
                index = (partition_index[i][0] + 1) // slice_shape
            sliced_index = sliced_index * (shape // slice_shape) + index
        return sliced_index
