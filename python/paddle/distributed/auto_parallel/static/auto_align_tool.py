# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import copy
import os
import pickle

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.base import core
from paddle.base.framework import Program
from paddle.distributed.auto_parallel.static.converter import Converter
from paddle.distributed.auto_parallel.static.dist_context import (
    get_default_distributed_context,
)
from paddle.distributed.auto_parallel.static.utils import (
    is_backward_op,
    is_forward_op,
    is_loss_op,
)
from paddle.static.io import deserialize_program

_valid_types = [
    core.VarDesc.VarType.DENSE_TENSOR,
    core.VarDesc.VarType.SELECTED_ROWS,
    core.VarDesc.VarType.DENSE_TENSOR_ARRAY,
]

paddle.enable_static()


class AutoAlignTool:
    """
    This is an automatic parallel precision alignment tool。
    """

    def __init__(self, program: Program, step=1, fetch_list=None):
        """Set some initialization information of the tool.
        step: Step when returning a specific variable name。
        fetch_list: initialization fetch_list.When a specific step is not reached, return this.
                 It can combine with Engine class。
                 example:in Engine.fit function,like this
                         try:
                             fetch_list = []
                             align_tool = AutoAlignTool(self.main_program, 0, fetch_names)
                             level = 0
                             fetch_list = align_tool.get_var(level, step)
                             outs = self._executor.run(
                                 self.main_program,
                                 fetch_list=fetch_list,
                                 use_program_cache=self._strategy.use_cache,
                                 return_numpy=self._strategy.return_numpy,
                             )
                             if fetch_list != fetch_names:
                                 align_tool.save(dir_path, outs, fetch_list, self._dist_contexts["train"], self.serial)
                                 exit(0)
                         except core.EOFException:
                             break
        """
        assert isinstance(program, Program)
        self._program = program
        self._blocks = program.blocks
        self._step = step
        self._fetch_list = fetch_list
        assert self._blocks is not None

    def set_step(self, step):
        self._step = step

    def get_var(self, level, step):
        """
        level must be in [0,1,2,3,4,5].
        """
        if step != self._step or step == -1:
            return self._fetch_list
        if level == 0:
            return self.get_loss_lr_var()
        elif level == 1:
            return self.get_data_var()
        elif level == 2:
            return self.get_param_var()
        elif level == 3:
            return self.get_param_grad_var()
        elif level == 4:
            return self.get_forward_tmp_var()
        elif level == 5:
            return self.get_backward_tmp_var()
        else:
            raise ValueError

    def set_program(self, program: Program):
        assert isinstance(program, Program)
        self._program = program
        self._blocks = program.blocks
        assert self._blocks is not None

    def get_loss_lr_var(self):
        """
        Returns the variable name of learning rate and loss
        """
        fetch_set = set()
        loss_ops = []
        for block in self._blocks:
            for op in block.ops:
                if is_loss_op(op):
                    assert (
                        len(op.desc.output_arg_names()) == 1
                    ), "loss op should only output loss var"
                    loss_ops.append(op)

        for block in self._blocks:
            for varname in block.vars:
                var = block._find_var_recursive(varname)

                if var is None or var.type not in _valid_types:
                    continue

                if "learning_rate" in var.name:
                    fetch_set.add(var.name)

        for loss_op in loss_ops:
            fetch_set.add(loss_op.output_arg_names[0])

        return list(fetch_set)

    def get_data_var(self):
        """
        Returns the variable name of data.
        """
        fetch_set = set()
        for block in self._blocks:
            for varname in block.vars:
                var = block._find_var_recursive(varname)

                if var is None or var.type not in _valid_types:
                    continue

                if var.is_data:
                    fetch_set.add(var.name)
        return list(fetch_set)

    def get_param_var(self):
        """
        Returns the variable name of parameters.
        """
        fetch_set = set()
        for block in self._blocks:
            for op in block.ops:
                if is_backward_op(op):
                    break
                for varname in op.input_arg_names + op.output_arg_names:
                    var = block._find_var_recursive(varname)
                    if var is None or var.type not in _valid_types:
                        continue
                    if var.is_parameter:
                        fetch_set.add(varname)

        return list(fetch_set)

    def get_param_grad_var(self):
        """
        Returns the variable name of parameters' gradient.
        """
        fetch_set = set()
        for block in self._blocks:
            for op in block.ops:
                if is_forward_op(op):
                    continue
                for varname in op.input_arg_names + op.output_arg_names:
                    if "@GRAD" not in varname:
                        continue
                    fwd_varname = varname.split("@GRAD")[0]
                    fwd_var = block._find_var_recursive(fwd_varname)
                    if fwd_var is None or fwd_var.type not in _valid_types:
                        continue
                    if fwd_var.is_parameter is False:
                        continue
                    var = block._find_var_recursive(varname)
                    if var is None or var.type not in _valid_types:
                        continue
                    fetch_set.add(varname)

        return list(fetch_set)

    def get_forward_tmp_var(self):
        """
        Returns the name of the temporary variable in the forward propagation
        """
        fetch_set = set()
        loss_lr_list = self.get_loss_lr_var()
        for block in self._blocks:
            for op in block.ops:
                if is_backward_op(op):
                    break
                for varname in op.input_arg_names + op.output_arg_names:
                    if varname in loss_lr_list:
                        continue
                    var = block._find_var_recursive(varname)
                    if var is None or var.type not in _valid_types:
                        continue
                    if var.is_data or var.is_parameter:
                        continue
                    fetch_set.add(varname)

        return list(fetch_set)

    def get_backward_tmp_var(self):
        """
        Returns the name of a temporary variable in back-propagation
        """
        fetch_set = set()
        loss_lr_list = self.get_loss_lr_var()
        forward_tmp_list = self.get_forward_tmp_var()
        for block in self._blocks:
            for op in block.ops:
                if is_backward_op(op):
                    for varname in op.input_arg_names + op.output_arg_names:
                        if (
                            varname in loss_lr_list
                            or varname in forward_tmp_list
                        ):
                            continue
                        if "@GRAD" in varname:
                            fwd_varname = varname.split("@GRAD")[0]
                            fwd_var = block._find_var_recursive(fwd_varname)
                            if (
                                fwd_var is not None
                                and fwd_var.type in _valid_types
                            ):
                                if fwd_var.is_parameter:
                                    continue
                        var = block._find_var_recursive(varname)
                        if var is None or var.type not in _valid_types:
                            continue
                        if var.is_data or var.is_parameter:
                            continue
                        fetch_set.add(varname)

        return list(fetch_set)

    def save(self, save_dir, vars, fetch_list, dist_context=None):
        """
        save fetch variables, distributed properties of variables and program.
        """
        if os.path.exists(save_dir) is False:
            os.mkdir(save_dir)
        if dist_context is None:
            dist_context = get_default_distributed_context()
        assert os.path.exists(save_dir)
        if dist.get_world_size() == 1:
            vars_path = os.path.join(save_dir, "vars.pkl")
            program_path = os.path.join(save_dir, "program.pdmodel")
            dist_attr_path = os.path.join(save_dir, "dist_attr.pkl")
        else:
            vars_path = os.path.join(
                save_dir, f"vars_rank{dist.get_rank()}.pkl"
            )
            program_path = os.path.join(
                save_dir, f"program_rank{dist.get_rank()}.pdmodel"
            )
            dist_attr_path = os.path.join(
                save_dir, f"dist_attr_rank{dist.get_rank()}.pkl"
            )
        if vars is not None:
            vars_dict = {}
            assert len(fetch_list) == len(vars)
            for i in range(len(fetch_list)):
                if vars[i] is None:
                    continue
                vars_dict[fetch_list[i]] = vars[i]
            with open(vars_path, "wb") as f:
                pickle.dump(vars_dict, f)
            dist_attr = {}
            for var in self._program.list_vars():
                if var.name not in fetch_list:
                    continue
                tensor_dist_attr = (
                    dist_context.get_tensor_dist_attr_for_program(var)
                )
                if tensor_dist_attr is None:
                    continue
                process_mesh = tensor_dist_attr.process_mesh
                dims_mapping = tensor_dist_attr.dims_mapping
                dist_attr[var.name] = {
                    "process_shape": process_mesh.shape,
                    "process_group": process_mesh.process_ids,
                    "dims_mapping": dims_mapping,
                }
            if len(dist_attr) > 0:
                with open(dist_attr_path, "wb") as f:
                    pickle.dump(dist_attr, f)
        if self._program is not None:
            with open(program_path, "wb") as f:
                f.write(self._program.desc.serialize_to_string())

    @staticmethod
    def load(save_dir):
        assert os.path.exists(save_dir)
        filename_list = sorted(os.listdir(save_dir))
        vars_list = []
        program_list = []
        dist_attr_list = []
        for filename in filename_list:
            filepath = os.path.join(save_dir, filename)
            assert os.path.isfile(filepath)
            if "vars" in filename:
                assert filename.endswith("pkl")
                with open(filepath, "rb") as f:
                    vars_list.append(pickle.load(f))
            elif "program" in filename:
                assert filename.endswith("pdmodel")
                with open(filepath, "rb") as f:
                    program_string = f.read()
                program_list.append(deserialize_program(program_string))
            elif "dist_attr" in filename:
                assert filename.endswith("pkl")
                with open(filepath, "rb") as f:
                    dist_attr_list.append(pickle.load(f))

        dist_attr_map = {}
        for dist_attrs in dist_attr_list:
            for dist_attr_name in dist_attrs.keys():
                if dist_attr_name not in dist_attr_map:
                    dist_attr_map[dist_attr_name] = dist_attrs[dist_attr_name]
        assert len(vars_list) == len(program_list)
        return vars_list, program_list, dist_attr_map

    @staticmethod
    def convert_src_tensor_2_dst_tensor(vars_list, src_attr_map, dst_attr_map):
        """
        Converter is a class object for auto parallel to convert tensors from
        one parallel strategy to another one. Tensors will merge and slice value
        with their strategy when strategies are different.
        But like dp to pp or dp to serial is not supported.
        """
        assert len(vars_list) >= 1
        # if dist_attr_map is None or len(dist_attr_map) == 0 or len(vars_list) == 1:
        if src_attr_map is None or len(src_attr_map) == 0:
            return vars_list[0]

        dst_strategies = {}
        src_strategies = {}
        tensors_dict = {}

        convert_tensor_dict = None
        for var_name in src_attr_map.keys():
            assert var_name not in dst_strategies
            dist_vars = []
            for vars in vars_list:
                if var_name in vars.keys():
                    dist_vars.append(vars[var_name])
            if len(dist_vars) == 0:
                continue

            if var_name in dst_attr_map and var_name in src_attr_map:
                dst_strategies[var_name] = copy.deepcopy(dst_attr_map[var_name])
                src_strategies[var_name] = copy.deepcopy(src_attr_map[var_name])
                tensors_dict[var_name] = dist_vars

        if src_attr_map == dst_attr_map:
            return tensors_dict
        converter = Converter(tensors_dict, src_strategies, dst_strategies)
        convert_tensor_dict = converter.convert()

        return convert_tensor_dict

    @staticmethod
    def find_diff_vars(fixed_vars_map, query_vars_map):
        """
        Found two variable names with different variable lists
        """
        diff_var_name_list = set()
        for var_name in fixed_vars_map.keys():
            if var_name in query_vars_map:
                fixed_vars = fixed_vars_map[var_name]
                query_vars = query_vars_map[var_name]
                if isinstance(fixed_vars, np.ndarray):
                    fixed_vars = [fixed_vars]
                if isinstance(query_vars, np.ndarray):
                    query_vars = [query_vars]

                length = min(len(fixed_vars), len(query_vars))
                if len(fixed_vars) != len(query_vars):
                    print()
                for i in range(length):
                    if not np.allclose(fixed_vars[i], query_vars[i]):
                        diff_var_name_list.add(var_name)
        return diff_var_name_list

    @staticmethod
    def diff_information(right_dir, wrong_dir):
        """
        Find the corresponding operator according to the variable name.
        """
        (
            right_vars_list,
            right_program_list,
            right_dist_attr_map,
        ) = AutoAlignTool.load(right_dir)
        (
            wrong_vars_list,
            wrong_program_list,
            wrong_dist_attr_map,
        ) = AutoAlignTool.load(wrong_dir)
        right_tensors_dict = AutoAlignTool.convert_src_tensor_2_dst_tensor(
            right_vars_list, right_dist_attr_map, right_dist_attr_map
        )
        wrong_tensors_dict = AutoAlignTool.convert_src_tensor_2_dst_tensor(
            wrong_vars_list, wrong_dist_attr_map, right_dist_attr_map
        )

        diff_var_name_list = AutoAlignTool.find_diff_vars(
            right_tensors_dict, wrong_tensors_dict
        )

        diff_ops_varname_dict = collections.OrderedDict()

        for program in wrong_program_list:
            for block in program.blocks:
                for op in block.ops:
                    for varname in op.input_arg_names + op.output_arg_names:
                        if varname in diff_var_name_list:
                            if len(diff_ops_varname_dict) == 0:
                                print(
                                    "first different op:\n",
                                    op,
                                    f"\ndifferent varname is:{varname}",
                                )
                            if op not in diff_ops_varname_dict:
                                diff_ops_varname_dict[op] = [varname]
                            else:
                                diff_ops_varname_dict[op].append(varname)

        return diff_ops_varname_dict

    @staticmethod
    def diff_information_from_dirs(right_dirs, wrong_dirs):
        right_vars_list = []
        right_program_list = []
        right_dist_attr_map = {}
        for right_dir in right_dirs:
            (
                tmp_vars_list,
                right_program_list,
                tmp_dist_attr_map,
            ) = AutoAlignTool.load(right_dir)
            if len(right_vars_list) == 0:
                right_vars_list = tmp_vars_list
            else:
                for i in range(len(tmp_vars_list)):
                    vars_list = tmp_vars_list[i]
                    for key in vars_list.keys():
                        if key not in right_vars_list[i].keys():
                            right_vars_list[i][key] = vars_list[key]

            for key in tmp_dist_attr_map.keys():
                if key not in right_dist_attr_map:
                    right_dist_attr_map[key] = tmp_dist_attr_map[key]

        wrong_vars_list = []
        wrong_program_list = []
        wrong_dist_attr_map = {}
        for wrong_dir in wrong_dirs:
            (
                tmp_vars_list,
                wrong_program_list,
                tmp_dist_attr_map,
            ) = AutoAlignTool.load(wrong_dir)
            if len(wrong_vars_list) == 0:
                wrong_vars_list = tmp_vars_list
            else:
                for i in range(len(tmp_vars_list)):
                    vars_list = tmp_vars_list[i]
                    for key in vars_list.keys():
                        if key not in wrong_vars_list[i].keys():
                            wrong_vars_list[i][key] = vars_list[key]

            for key in tmp_dist_attr_map.keys():
                if key not in wrong_dist_attr_map:
                    wrong_dist_attr_map[key] = tmp_dist_attr_map[key]

        right_tensors_dict = AutoAlignTool.convert_src_tensor_2_dst_tensor(
            right_vars_list, right_dist_attr_map, right_dist_attr_map
        )
        wrong_tensors_dict = AutoAlignTool.convert_src_tensor_2_dst_tensor(
            wrong_vars_list, wrong_dist_attr_map, right_dist_attr_map
        )
        diff_var_name_list = AutoAlignTool.find_diff_vars(
            right_tensors_dict, wrong_tensors_dict
        )

        diff_ops_varname_dict = collections.OrderedDict()

        for program in wrong_program_list:
            for block in program.blocks:
                for op in block.ops:
                    for varname in op.input_arg_names + op.output_arg_names:
                        if varname in diff_var_name_list:
                            if len(diff_ops_varname_dict) == 0:
                                print(
                                    "first different op:\n",
                                    op,
                                    f"\ndifferent varname is:{varname}",
                                )
                            if op not in diff_ops_varname_dict:
                                diff_ops_varname_dict[op] = [varname]
                            else:
                                diff_ops_varname_dict[op].append(varname)

        return diff_ops_varname_dict
