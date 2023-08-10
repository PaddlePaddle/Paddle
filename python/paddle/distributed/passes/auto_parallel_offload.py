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

import logging

from paddle.framework import core
from paddle.utils import unique_name

from ..auto_parallel.static.dist_attribute import OperatorDistAttr
from ..auto_parallel.static.utils import set_var_dist_attr
from .auto_parallel_recompute import insert_dependencies_for_two_ops
from .pass_base import PassBase, register_pass

# logging.basicConfig(
#     format='%(levelname)s - %(asctime)s - %(pathname)s: %(lineno)s - %(message)s',
#     level=logging.DEBUG,
# )


@register_pass("auto_parallel_recom_offload")
class RecomOffloadPass(PassBase):
    def __init__(self):
        super().__init__()
        self.set_attr("loss", None)
        self.set_attr("dist_context", None)
        self.set_attr("offload_points", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        if self.get_attr("loss") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _creat_vars(self, varname, ref_var_dist_attr):
        """
        Create new variables for offload and fetch.
        """
        pinned_var_name = unique_name.generate(varname + "@Pinned")
        fetched_var_name = unique_name.generate(varname + "@Fetch")

        pinned_var = self._main_block.create_var(
            name=pinned_var_name,
            shape=self._main_block.var(varname).shape,
            dtype=self._main_block.var(varname).dtype,
            persistable=False,
            stop_gradient=False,
        )
        # set the dist_attr of new var according to origin var
        ref_mesh = ref_var_dist_attr.process_mesh
        ref_mapping = ref_var_dist_attr.dims_mapping
        set_var_dist_attr(self._dist_context, pinned_var, ref_mapping, ref_mesh)

        fetched_var = self._main_block.create_var(
            name=fetched_var_name,
            shape=self._main_block.var(varname).shape,
            dtype=self._main_block.var(varname).dtype,
            persistable=False,
            stop_gradient=False,
        )

        set_var_dist_attr(
            self._dist_context, fetched_var, ref_mapping, ref_mesh
        )

        return pinned_var_name, fetched_var_name

    def _reset_op_dist_attr(self, op, new_var_name, is_input=True):
        """
        Reset the dist_attr of the input and output of the operator.
        """
        op_dist_attr = self._dist_context.get_op_dist_attr_for_program(op)
        var_dist_attr = self._dist_context.get_tensor_dist_attr_for_program(
            self._main_block.var(new_var_name)
        )
        assert (
            op_dist_attr is not None
        ), "Reset op {}'s dist_attr, but its dist_attr is None".format(
            op.desc.type()
        )
        if is_input:
            op_dist_attr.set_input_dist_attr(new_var_name, var_dist_attr)
        if not is_input:
            op_dist_attr.set_output_dist_attr(new_var_name, var_dist_attr)

    def _record_op_to_insert(self, idx, checkpoint_name, op_type):
        """
        Record op which needs be insterted in main_block
        """
        if op_type == "offload":
            assert (
                checkpoint_name in self.unoffload_checkpoint_names
            ), "{} not in unoffload_checkpoint_names"
            self.unoffload_checkpoint_names.remove(checkpoint_name)
            logging.debug(f"Record offload {checkpoint_name}")
            if idx in self.pos_insert_op_map:
                self.pos_insert_op_map[idx].append(("offload", checkpoint_name))
            else:
                self.pos_insert_op_map[idx] = [("offload", checkpoint_name)]
            return checkpoint_name
        elif op_type == "fetch":
            assert (
                checkpoint_name in self.unoffload_checkpoint_names
            ), "{} not in unoffload_checkpoint_names"
            self.unoffload_checkpoint_names.remove(checkpoint_name)
            logging.debug(f"Record fetch {checkpoint_name}")
            if idx in self.pos_insert_op_map:
                self.pos_insert_op_map[idx].append(("fetch", checkpoint_name))
            else:
                self.pos_insert_op_map[idx] = [("fetch", checkpoint_name)]
            return checkpoint_name
        else:
            raise ValueError(
                "Only support op_type 'offload' and 'fetch', but received {}".format(
                    op_type
                )
            )

    def _get_offload_pos(self):
        """
        Parse and get the variables's information that need to be offloaded during forward pass.
        """
        # such as: "origin_idx : ["fetch", var_name, "fetch", var_name]"
        self.pos_insert_op_map = {}
        self.unoffload_checkpoint_names = self._offload_points[:]
        self.need_offload_checkpoint_names = self.unoffload_checkpoint_names[:]

        # record the checkpoints after they are generated
        for idx, op in enumerate(
            self._main_block.ops[self.fw_start_op_idx : self.bw_start_op_idx]
        ):
            output_var_names = op.desc.output_arg_names()

            # record the ops which need be offloaded
            for output_var_name in output_var_names:
                if output_var_name in self.need_offload_checkpoint_names:
                    if output_var_name in self.unoffload_checkpoint_names:
                        self._record_op_to_insert(
                            idx + 1, output_var_name, "offload"
                        )
                    else:
                        raise ValueError(
                            "The variable {} has already been offload, but it is used as output of op {} again.".format(
                                output_var_name, op.desc.type()
                            )
                        )
                else:
                    pass
        logging.debug(
            "There are some checkpoints unoffloaded: {}, maybe it is wrong".format(
                self.unoffload_checkpoint_names
            )
        )

    def _get_fetch_pos(self):
        """
        Parse and get the variables's information that need to be fetched during backward pass.
        """
        # such as: "origin_idx : ["fetch", var_name, "fetch", var_name]"
        self.pos_insert_op_map = {}
        self.unoffload_checkpoint_names = self.need_offload_checkpoint_names[:]
        # find the first forward op idx and the first backward op idx
        all_ops_num = len(self._main_block.ops)
        self.bw_start_op_idx = all_ops_num
        for idx, op in enumerate(self._main_block.ops):
            if int(op.desc.attr('op_role')) == 1:  # 1 means 'backward op'
                self.bw_start_op_idx = idx
                break
            else:
                pass
        assert self.bw_start_op_idx < all_ops_num, "BW op should exist"

        for i, op in enumerate(self._main_block.ops[self.bw_start_op_idx :]):
            current_idx = self.bw_start_op_idx + i
            input_varnames = op.desc.input_arg_names()
            output_varnames = op.desc.output_arg_names()  # for nop op
            for input_varname in input_varnames:
                if input_varname in self.need_offload_checkpoint_names:
                    if input_varname in self.unoffload_checkpoint_names:
                        insert_idx = (
                            current_idx - 1
                            if current_idx - 1 >= self.bw_start_op_idx
                            else self.bw_start_op_idx
                        )
                        self._record_op_to_insert(
                            insert_idx, input_varname, "fetch"
                        )
                    else:
                        logging.debug(
                            f"The var {input_varname} has already been fetched and is reused by op {op.desc.type()}"
                        )
                    # deal origin ops inputs
                    self._reset_op_dist_attr(
                        self._main_block.ops[current_idx],
                        self._varnames2fetch_names[input_varname],
                    )
                    self._main_block.ops[current_idx]._rename_input(
                        input_varname,
                        self._varnames2fetch_names[input_varname],
                    )
                else:
                    pass

            for output_varname in output_varnames:
                if output_varname in self.need_offload_checkpoint_names:
                    self._reset_op_dist_attr(
                        self._main_block.ops[current_idx],
                        self._varnames2fetch_names[output_varname],
                        is_input=False,
                    )
                    self._main_block.ops[current_idx]._rename_output(
                        output_varname,
                        self._varnames2fetch_names[output_varname],
                    )
        logging.debug(
            "There are some checkpoints unfetched {}, maybe it is wrong!".format(
                self.unoffload_checkpoint_names
            )
        )

    def _update_main_program(self, start_op_idx, end_op_idx):
        if len(self.pos_insert_op_map) == 0:
            return
        for op_idx in reversed(range(start_op_idx, end_op_idx)):
            if op_idx in self.pos_insert_op_map:
                op_varnames = self.pos_insert_op_map[op_idx]
                for i, op_var in enumerate(op_varnames):
                    if op_var[0] == "offload":
                        assert (
                            op_var[1] in self._varnames2pinned_names
                        ), "The variable {} is not in _varnames2pinned_names".format(
                            op_var[1]
                        )
                        dst_varname = self._varnames2pinned_names[op_var[1]]
                        self._insert_async_memcpy_op(
                            op_idx + i, op_var[1], dst_varname, 0, 1
                        )
                    elif op_var[0] == "fetch":
                        assert (
                            op_var[1] in self._varnames2fetch_names
                        ), "The variable {} is not in _varnames2fetch_names".format(
                            op_var[1]
                        )
                        src_varname = self._varnames2pinned_names[op_var[1]]
                        dst_varname = self._varnames2fetch_names[op_var[1]]
                        self._insert_async_memcpy_op(
                            op_idx + i, src_varname, dst_varname, 1, 0
                        )
                    else:
                        raise ValueError(
                            "Only support op_type 'offload' and 'fetch', but reveived {}".format(
                                op_var[0]
                            )
                        )
        self._main_block._sync_with_cpp()

    def _insert_async_memcpy_op(
        self, insert_idx, src_varname, dst_varname, op_role, dst_place_type
    ):
        OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
        op_type = 'memcpy_d2h' if op_role == 0 else 'memcpy_h2d'
        new_op = self._main_block._insert_op_without_sync(
            insert_idx,
            type=op_type,
            inputs={'X': [self._main_block.var(src_varname)]},
            outputs={'Out': [self._main_block.var(dst_varname)]},
            attrs={"dst_place_type": int(dst_place_type), OP_ROLE_KEY: op_role},
        )
        new_op_dist_attr = OperatorDistAttr(new_op.desc)
        input_dist_attr = self._dist_context.get_tensor_dist_attr_for_program(
            self._main_block.var(src_varname)
        )
        new_op_dist_attr.process_mesh = input_dist_attr.process_mesh
        output_dist_attr = self._dist_context.get_tensor_dist_attr_for_program(
            self._main_block.var(dst_varname)
        )
        new_op_dist_attr.set_input_dist_attr(src_varname, input_dist_attr)
        new_op_dist_attr.set_output_dist_attr(dst_varname, output_dist_attr)
        self._dist_context.set_op_dist_attr_for_program(
            new_op, new_op_dist_attr
        )
        # insert nop op to add a dependency between "fetch" op and "grad op"
        if op_role == 1:
            prior_op = self._main_block.ops[insert_idx - 1]
            prior_mesh = self._dist_context.get_op_dist_attr_for_program(
                prior_op
            ).process_mesh
            cur_mesh = self._dist_context.get_op_dist_attr_for_program(
                new_op
            ).process_mesh
            if prior_mesh == cur_mesh:
                insert_dependencies_for_two_ops(
                    self._main_block,
                    insert_idx,
                    prior_op,
                    new_op,
                    self._dist_context,
                )

    def _apply_single_impl(self, main_program, startup_program, context):
        # paddle.save(main_program, "orgin_prog.pdmodle")
        self._offload_points = self.get_attr("offload_points")
        logging.debug(f"The origin checkpints is {self._offload_points}")
        # loss = self.get_attr("loss")
        # no_grad_set = self.get_attr("no_grad_set")
        self._dist_context = self.get_attr("dist_context")
        self._main_block = main_program.global_block()

        # 1. We need to pop some tensors those shouldn't be offload
        # 1.1 Pop the tensors those don't have memory such as Xshape of 'unsqueese2'.
        ckpts_without_memory = []
        for var_name in self._offload_points:
            var = self._main_block.var(var_name)
            if 0 in var.shape:
                ckpts_without_memory.append(var_name)
        logging.debug(f"Those vars {ckpts_without_memory} don't have memory.")
        for var_name in ckpts_without_memory:
            self._offload_points.remove(var_name)

        # 1.2 Pop the checkpoints that are the last op in forward pass
        # find the first forward op idx and the first backward op idx
        self.fw_start_op_idx = len(self._main_block.ops)
        self.bw_start_op_idx = -1
        for idx, op in enumerate(self._main_block.ops):
            if int(
                op.desc.attr('op_role')
            ) == 0 and self.fw_start_op_idx == len(
                self._main_block.ops
            ):  # 0 means 'forward op'
                self.fw_start_op_idx = idx
            elif int(op.desc.attr('op_role')) == 1:  # 1 means 'backward op'
                self.bw_start_op_idx = idx
                break
            else:
                pass
        logging.debug(
            "fw_start_op_idx:{} and bw_first_op_idx: {}".format(
                self.fw_start_op_idx, self.bw_start_op_idx
            )
        )
        assert self.fw_start_op_idx < len(
            self._main_block.ops
        ), "FW op should exist"
        assert self.bw_start_op_idx < len(
            self._main_block.ops
        ), "BW op should exist"
        assert (
            self.fw_start_op_idx < self.bw_start_op_idx
        ), "BW start op must be behind th FW start op"

        has_pop_last_outputs = False
        for op in reversed(
            self._main_block.ops[self.fw_start_op_idx : self.bw_start_op_idx]
        ):
            output_var_names = op.desc.output_arg_names()
            output_count = len(output_var_names)
            pop_output_count = 0
            for output_var_name in output_var_names:
                if pop_output_count != 0:
                    assert (
                        output_var_name in self._offload_points
                    ), "The operator {} has more than one outputs, but not all the outputs are checkpoints, such as {}".format(
                        op.desc.type(), output_var_name
                    )
                if output_var_name in self._offload_points:
                    self._offload_points.remove(output_var_name)
                    pop_output_count += 1
                    if pop_output_count == output_count:
                        has_pop_last_outputs = True
                if has_pop_last_outputs:
                    break
            if has_pop_last_outputs:
                break

        assert len(self._offload_points) > 0, " points must not empty"
        logging.debug(
            f"The checkpints without the last op's checkpoints is {self._offload_points}"
        )
        self._varnames2pinned_names = {}
        self._varnames2fetch_names = {}

        # 2. ctreat auxiliary vars
        for _, op in enumerate(self._main_block.ops):
            if len(self._varnames2fetch_names) == len(self._offload_points):
                break
            output_vars = op.desc.output_arg_names()
            for output_var in output_vars:
                if output_var in self._offload_points:
                    ref_op_dist_attr = (
                        self._dist_context.get_op_dist_attr_for_program(op)
                    )
                    ref_var_dist_attr = ref_op_dist_attr.get_output_dist_attr(
                        output_var
                    )
                    assert ref_var_dist_attr is not None
                    pinned_var_name, fetched_var_name = self._creat_vars(
                        output_var, ref_var_dist_attr
                    )
                    self._varnames2pinned_names[output_var] = pinned_var_name
                    self._varnames2fetch_names[output_var] = fetched_var_name

        # 3. add offload op(memcpy_d2h) in forward pass
        self._get_offload_pos()
        self._update_main_program(self.fw_start_op_idx, self.bw_start_op_idx)

        # 4. add fetch op(memcpy_h2d) in backward pass
        self._get_fetch_pos()
        self._update_main_program(
            self.bw_start_op_idx, len(self._main_block.ops)
        )
