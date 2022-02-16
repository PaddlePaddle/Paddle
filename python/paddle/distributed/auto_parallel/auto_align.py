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

import os
import pickle
from collections import OrderedDict

import numpy as np

import paddle
from paddle.fluid import core
from paddle.fluid.framework import Program
from .dist_context import get_default_distributed_context
from .utils import _merge_parameter, _load_distributed_attribute
from .reshard import _compute_complete_shape, _compute_partition_index


class AutoAlign:
    """
    Locate the first step with different loss and op with different values.
    
    Args:
        directory (str): Directory to save parameters
        program (Program): The program for alignment
    """

    def __init__(self, program, directory='./'):
        if not os.path.exists(directory):
            directory = './'
        if not isinstance(directory, (str)):
            raise TypeError("directory must be string, got %s here" %
                            type(directory))
        if not isinstance(program, (Program)):
            raise TypeError(
                "program must be paddle.fluid.framework.program, got %s here" %
                type(program))
        self._directory = directory
        self._program = program

    def _save_diff_info(self, diff_info):
        """Save different information of step and location(forward or backward)."""
        path = os.path.join(self._directory, "diff_info.pkl")
        with open(path, 'wb') as f:
            pickle.dump(diff_info, f)

    def save_serial_tensors(self, tensor_dict):
        """Save serial tensors."""
        path = os.path.join(self._directory, "tensors.pkl")
        with open(path, 'wb') as f:
            pickle.dump(tensor_dict, f)

    def save_dist_tensors(self, tensor_dict):
        """Save all rank tensors."""
        path = os.path.join(
            self._directory,
            "tensors_" + str(paddle.distributed.get_rank()) + ".pkl")
        with open(path, 'wb') as f:
            pickle.dump(tensor_dict, f)

    def save_dist_attr(self, dist_context=None):
        """ Set distributed attribute of every rank."""
        if dist_context is None:
            dist_context = get_default_distributed_context()
        dist_attr = {}
        for var in self._program.list_vars():
            tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                var)
            if tensor_dist_attr is None:
                continue
            process_mesh = tensor_dist_attr.process_mesh
            dims_mapping = tensor_dist_attr.dims_mapping
            dist_attr[var.name] = {
                "process_shape": process_mesh.topology,
                "process_group": process_mesh.processes,
                "dims_mapping": dims_mapping
            }
        path = os.path.join(
            self._directory,
            "dist_attr_" + str(paddle.distributed.get_rank()) + ".pkl")
        with open(path, 'wb') as f:
            pickle.dump(dist_attr, f)
        return dist_attr

    def get_grad_vars(self):
        """Get grad vars."""
        grad_var = set()
        vars = self._program.global_block().vars
        ops = self._program.global_block().ops
        for op in ops:
            for var_name in op.output_arg_names:
                if var_name in vars:
                    if "@GRAD" in var_name and vars[var_name].desc.type(
                    ) != core.VarDesc.VarType.READER:
                        grad_var.add(var_name)
        return list(grad_var)

    def get_forward_var(self):
        """Get forward vars."""
        forward_var = set()
        for idx, op in enumerate(distributed_main_program.global_block().ops):
            if op.type == "fill_constant" and "@GRAD" in op.output_arg_names[0]:
                break
            for var_name in op.input_arg_names:
                if op.type == 'create_py_reader' or op.type == 'create_double_buffer_reader' or op.type == 'read':
                    continue
                forward_var.add(var_name)
        return list(forward_var)

    def check_loss(self, dist_tensor_file_path_list, serial_path,
                   dist_attr_file_path):
        """
        Compare average dist losses with serial losses.
        1. load serial losses
        2. load dist losses
        3. compare serial losses with average dist losses

        Args:
            dist_tensor_file_path_list (list): used to hold every rank losses 
            serial_path (str): the path of serial losses
            dist_attr_file_path (str): the path of dist losses

        Examples:
            dist_tensor_file_path_list = ["./tensors_0.pkl", "./tensors_1.pkl"]
            autoalign.check_loss(dist_tensor_file_path_list, "./tensors.pkl", "dist_attr_0.pkl")
        """
        dist_dict = OrderedDict()
        with open(dist_attr_file_path, 'rb') as f:
            dist_attr_dict = pickle.load(f)
        for file_path in dist_tensor_file_path_list:
            with open(file_path, 'rb') as f:
                tensor_dict = pickle.load(f)
                for step in tensor_dict.keys():
                    if step not in dist_dict.keys():
                        dist_dict[step] = {}
                    for tensor_name in tensor_dict[step].keys():
                        if tensor_name not in dist_dict[step].keys():
                            dist_dict[step][tensor_name] = []
                        dist_dict[step][tensor_name].append(tensor_dict[step][
                            tensor_name])

        with open(serial_path, 'rb') as f:
            serial_dict = pickle.load(f)

        diff_step = None
        check_step = -1
        result = True
        for step in dist_dict.keys():
            first_step = next(iter(dist_dict.keys()))
            if dist_dict[step]["loss"]:
                avg_loss = sum(dist_dict[step]["loss"]) / len(dist_dict[step][
                    "loss"])
                serial_loss = serial_dict[step]["loss"]
                print('Step: {}, serial loss: {}, average loss: {}'.format(
                    step, serial_loss, avg_loss))
                if avg_loss != serial_loss:
                    diff_step = step
                    print("The step {} loss is different".format(step))
                    if diff_step == first_step:
                        print(
                            'Check whether the parallel mode is DP, because DP may make a little different.'
                        )
                        check_step = first_step
                        print('Check step {} forward'.format(check_step))
                        diff_info = []
                        diff_info.append(check_step)
                        diff_info.append('forward')
                        self._save_diff_info(diff_info)
                    else:
                        check_step = diff_step - 1
                        print('Check step {} backward'.format(check_step))
                        diff_info = []
                        diff_info.append(check_step)
                        diff_info.append('backward')
                        print(diff_info)
                        self._save_diff_info(diff_info)
                    result = False
                    break
        if result:
            print("Loss is the same.")

        return result, check_step

    def _merge_parameter_with_dist_attr(self, param_list, dist_attr):
        """ Merge parameter with distributed attribute. """
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
                _merge_parameter(partition_param_list, param_list[index],
                                 partition_index, complete_shape)

        assert len(partition_param_list) == 1 or not partition_param_list, \
            "Fail to merge parameter"
        return partition_param_list[0][0]

    def find_diff_info(self, dist_tensor_file_path_list, serial_path,
                       dist_attr_file_path_list, diff_step):
        """
        Find different step and tensor.

        Args:
            dist_tensor_file_path_list(list): used to hold every rank losses  

        Examples:
            dist_tensor_file_path_list(list) = ["./tensors_0.pkl", "./tensors_1.pkl"]
            autoalign.find_diff_info(dist_tensor_file_path_list, "./tensors.pkl", "dist_attr_0.pkl")
        """
        with open(serial_path, 'rb') as f:
            serial_dict = pickle.load(f)
        dist_dict = OrderedDict()

        dist_attr_dict = {}
        for file_path in dist_attr_file_path_list:
            with open(file_path, 'rb') as f:
                tensor_dict = pickle.load(f)
                for tensor_name in tensor_dict.keys():
                    dist_attr_dict[tensor_name] = tensor_dict[tensor_name]

        dist_dict[diff_step] = {}
        for file_path in dist_tensor_file_path_list:
            with open(file_path, 'rb') as f:
                tensor_dict = pickle.load(f)
                for tensor_name in tensor_dict[diff_step].keys():
                    if tensor_name not in dist_dict[diff_step].keys():
                        dist_dict[diff_step][tensor_name] = []
                        dist_dict[diff_step][tensor_name].append(tensor_dict[
                            diff_step][tensor_name])

        for tensor_name in serial_dict[diff_step].keys():
            if tensor_name == "loss":
                continue
            serial_tensor = serial_dict[diff_step][tensor_name]
            if tensor_name in dist_dict[diff_step]:
                dist_tensor_list = dist_dict[diff_step][tensor_name]
                dist_attr = dist_attr_dict[tensor_name]
                merged_tensor = self._merge_parameter_with_dist_attr(
                    dist_tensor_list, dist_attr)
                serial_tensor = serial_dict[diff_step][tensor_name]

                if not np.allclose(merged_tensor, serial_tensor):
                    print(
                        'The tensor {} is different at step {}, serial tensor: {}, merged tensor: {}'.
                        format(tensor_name, diff_step, serial_tensor,
                               merged_tensor))
