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
import numpy as np


class Converter(object):
    """
    Merge and Slice

    Args:
        variables(list[varname]): all variables of current process
        vars_dict(dict): variables' value of all ranks that to be converted. 
                         key is var's name(str), value is all ranks' data(list(numpy.ndarray))
        pre_dist_attr(dict): variables' dist_attr of last training process.
        cur_dist_attr(dict): variables' dist_attr of current training process.
    """

    def __init__(self, variables, vars_dict, pre_dist_attr, cur_dist_attr):
        self.variables = variables
        self.vars_dict = vars_dict
        self.pre_dist_attr = pre_dist_attr
        self.cur_dist_attr = cur_dist_attr

    @variables.setter
    def variables(self, variables):
        if not isinstance(variables, list):
            raise TypeError("The type of variables should be 'list', "
                            "but got {}".format(str(type(variables))))
        if not all(isinstance(v, str) for v in variables):
            raise TypeError("The type of variables should be 'list[str]', "
                            "but got {}".format(str(type(variables))))
        self.variables = variables

    @vars_dict.setter
    def vars_dict(self, vars_dict):
        assert isinstance(vars_dict, dict), \
            "The type of 'vars_dict' should be 'dict', but got {}.".format(
                str(type(vars_dict)))
        for name, value in vars_dict.items():
            if not isinstance(name, str):
                raise TypeError("The key of 'vars_dict' is var's name, "
                                "and its type should be 'str', but got {}."
                                .format(str(type(name))))
            if not isinstance(value, list) or not all(
                    isinstance(v, np.ndarray) for v in value):
                raise TypeError(
                    "The value of 'vars_dict' is var's value of all ranks, "
                    "and its type should be 'list(numpy.ndarray)'.")
        self.vars_dict = vars_dict

    @pre_dist_attr.setter
    def pre_dist_attr(self, pre_dist_attr):
        if not pre_dist_attr:
            raise ValueError("'pre_dist_attr' can not be None.")
        elif not isinstance(pre_dist_attr, dict):
            raise TypeError("The type of 'pre_dist_attr' should be 'dict', "
                            "but got '{}'.".format(str(type(pre_dist_attr))))
        else:
            for name, value in pre_dist_attr.items():
                if not isinstance(name, str):
                    raise TypeError("The key of 'pre_dist_attr' is var's name, "
                                    "and its type should be 'str', "
                                    "but got '{}'.".format(str(type(name))))
                if not isinstance(value, dict):
                    raise TypeError(
                        "The type of distributed attribute should be 'dict', "
                        "but got '{}'".format(str(type(value))))
                attr = ['process_shape', 'process_group', 'dims_mapping']
                if list(value.keys()) != attr:
                    raise ValueError(
                        "The key of distributed attribute should be "
                        "'['process_shape', 'process_group', 'dims_mapping']', "
                        "but got {}.".format(str(value.keys())))
        self.pre_dist_attr = pre_dist_attr

    @cur_dist_attr.setter
    def cur_dist_attr(self, cur_dist_attr):
        if not cur_dist_attr:
            raise ValueError("'cur_dist_attr' can not be None.")
        elif not isinstance(cur_dist_attr, dict):
            raise TypeError("The type of 'cur_dist_attr' should be 'dict', "
                            "but got '{}'.".format(str(type(cur_dist_attr))))
        else:
            for name, value in cur_dist_attr.items():
                if not isinstance(name, str):
                    raise TypeError("The key of 'cur_dist_attr' is var's name, "
                                    "and its type should be 'str', "
                                    "but got '{}'.".format(str(type(name))))
                if not isinstance(value, dict):
                    raise TypeError(
                        "The type of distributed attribute should be 'dict', "
                        "but got '{}'".format(str(type(value))))
                attr = ['process_shape', 'process_group', 'dims_mapping']
                if list(value.keys()) != attr:
                    raise ValueError(
                        "The key of distributed attribute should be "
                        "'['process_shape', 'process_group', 'dims_mapping']', "
                        "but got {}.".format(str(value.keys())))
        self.cur_dist_attr = cur_dist_attr

    def convert(self):
        var_not_in_pre = []
        var_not_in_cur = []
        var_not_in_ckpt = []
        # logging.info("Start to merge and slice variables.")
        for varname in self.variables:
            if varname not in self.pre_dist_attr:
                var_not_in_pre.append(varname)
                continue
            if varname not in self.cur_dist_attr:
                var_not_in_cur.append(varname)
                self.vars_dict.pop(varname)
                continue
            if varname not in self.vars_dict:
                var_not_in_ckpt.append(varname)
                continue

            pre_attr = self.pre_dist_attr[varname]
            cur_attr = self.cur_dist_attr[varname]
            var_list = self.vars_dict[varname]
            self.vars_dict[varname] = self.merge_and_slice(var_list, pre_attr,
                                                           cur_attr)

        if var_not_in_pre:
            warnings.warn(
                "variables '{}' are not found in last training process."
                .format(str(var_not_in_pre)))
        if var_not_in_cur:
            warnings.warn(
                "variables '{}' are not found in current training process."
                .format(str(var_not_in_cur)))
        if var_not_in_ckpt:
            warnings.warn("variables '{}' are not found in checkpoint files."
                          .format(str(var_not_in_ckpt)))
        return self.vars_dict

    def merge_and_slice(self, var, pre_attr, cur_attr):
        """
        Merge variable with previous dist_attr and slice variable with current dist_attr

        Returns:
            var(numpy.narray): a variable's value of current rank.
        """
        if pre_attr == cur_attr:
            # skip merge and slice var
            rank_id = paddle.distributed.get_rank()
            index = cur_attr["process_group"].index(rank_id)
            var = var[index]
        else:
            pre_dims_mapping = pre_attr["dims_mapping"]
            cur_dims_mapping = cur_attr["dims_mapping"]
            if len(set(pre_dims_mapping)) > 1 or -1 not in pre_dims_mapping:
                # merge var
                var = self.merge_with_dist_attr(var, pre_attr)
            else:
                # skip merge var
                var = var[0]

            if len(set(cur_dims_mapping)) > 1 or -1 not in cur_dims_mapping:
                # slice var
                var = self.slice_with_dist_attr(var, cur_attr)

        return var

    def merge_with_dist_attr(self, var_list, dist_attr):
        """ Merge variable with distributed attribute """
        from .reshard import _compute_complete_shape, _compute_partition_index

        dims_mapping = dist_attr["dims_mapping"]
        process_shape = dist_attr["process_shape"]
        process_group = dist_attr["process_group"]
        # get the complete shape of the variable
        complete_shape = _compute_complete_shape(var_list[0].shape,
                                                 process_shape, dims_mapping)
        # merge the variable with dist_attr
        partition_var_list = []
        merged_partiton = []
        for process in process_group:
            partition_index = _compute_partition_index(
                process, complete_shape, dims_mapping, process_shape,
                process_group)
            index = process_group.index(process)
            if partition_index not in merged_partiton:
                merged_partiton.append(partition_index)
                self.merge(partition_var_list, var_list[index], partition_index,
                           complete_shape)

        assert len(partition_var_list) == 1 or not partition_var_list, \
            "Fail to merge variable"
        complete_var = partition_var_list[0][0]
        return complete_var

    def slice_with_dist_attr(self, var, dist_attr):
        """ Slice variable with distributed attribute """
        var = np.array(var) if isinstance(var, paddle.fluid.LoDTensor) else var
        dims_mapping = dist_attr["dims_mapping"]
        process_shape = dist_attr["process_shape"]
        process_group = dist_attr["process_group"]
        # slice the variable with dist_attr
        partition_index_list = self._get_split_indices(
            var.shape, dims_mapping, process_shape, process_group)
        sliced_var_list = self.split(var, partition_index_list,
                                     len(partition_index_list))
        # get the current variable's index in sliced_var_list
        rank_id = paddle.distributed.get_rank()
        sliced_var_index = self._get_sliced_index(
            rank_id, var.shape, dims_mapping, process_shape, process_group)
        sliced_var = sliced_var_list[sliced_var_index]
        return sliced_var

    def merge(self, partition_var_list, var, partition_index, complete_shape):
        """
        Merge partitial variables to a complete one.

        Returns:
            None

        Examples:
            .. code-block:: python

                import numpy as np
                partition_var_list = [(np.array([[[1.11, 1.12]]]), [[0,1],[0,1],[0,2]])]
                var = np.array([[[1.13, 1.14]]])
                partition_index = [[0,1],[0,1],[2,4]]

                _merge_variable(partition_var_list, var, partition_index)
                # partition_var_list: [(np.array([[[1.11, 1.12, 1.13, 1.14]]]), [[0,1],[0,1],[0,4]])]
        """
        from .reshard import _compute_concat_info

        if len(partition_var_list) == 1:
            is_complete_data = True
            for idx, item in enumerate(partition_var_list[0][1]):
                if item[0] != 0 or item[1] != complete_shape[idx]:
                    is_complete_data = False
                    break
            if is_complete_data:
                return

        if not partition_var_list:
            partition_var_list.append((var, partition_index))
        else:
            i = 0
            while i < len(partition_var_list):
                concat_axis, first_order, new_partition = _compute_concat_info(
                    partition_var_list[i][1], partition_index)
                if concat_axis != -1:
                    if first_order == 0:
                        new_var = np.concatenate(
                            (partition_var_list[i][0], var), axis=concat_axis)
                    else:
                        new_var = np.concatenate(
                            (var, partition_var_list[i][0]), axis=concat_axis)

                    partition_var_list.pop(i)
                    self.merge(partition_var_list, new_var, new_partition,
                               complete_shape)
                    break
                i += 1

    def split(self, complete_var, partition_index_list, length):
        """
        Slice a complete variable.

        Returns:
            sliced_var_list(list): sliced variables with 'partition_index_list'

        Examples:
            .. code-block:: python

                import numpy as np
                complete_var = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
                rank = 2
                complete_shape = [1, 1, 6]
                dims_mapping = [-1, -1, 0]
                process_shape = [3]
                process_group = [0, 1, 2]

                sliced_var_list = split(complete_var, [[], [], [2, 4]], 3)
                # [array([[[1.11, 1.12]]]), array([[[1.13, 1.14]]]), array([[[1.15, 1.16]]])]
        """
        sliced_var_list = []
        axis = len(complete_var.shape) - length
        sliced_var = np.split(
            complete_var, partition_index_list[axis], axis=axis)
        if length == 1:
            return sliced_var
        for var in sliced_var:
            sliced_var_list.extend(
                self.split(var, partition_index_list, length - 1))
        return sliced_var_list

    def _get_split_indices(self, complete_shape, dims_mapping, process_shape,
                           process_group):
        """
        Get split indices of every dimension.

        Returns:
            split_indices_list(list): the split indices of every dimension of the variable

        Examples:
            .. code-block:: python

                import numpy as np
                complete_var = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
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

    def _get_sliced_index(self, rank_id, complete_shape, dims_mapping,
                          process_shape, process_group):
        """
        Get sliced_var's index of current rank in all sliced variables list.

        Returns:
            sliced_var_index(int): the index of sliced var in sliced_var_list

        Examples:
            .. code-block:: python

                import numpy as np
                complete_var = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
                rank = 2
                complete_shape = [1, 1, 6]
                dims_mapping = [-1, -1, 0]
                process_shape = [3]
                process_group = [0, 1, 2]

                slice_var = _slice_variable(complete_var, [[], [], [2, 4]], 3)
                # slice_var: 
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
