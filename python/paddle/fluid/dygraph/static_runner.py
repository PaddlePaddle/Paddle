# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import collections
import logging
import numpy as np
import os
import six

from . import base
from . import layers
from .. import core
from .. import executor
from .. import framework
from .. import backward

from ..proto import framework_pb2
from ... import compat as cpt

from ..transpiler.details import program_utils

__all__ = ["StaticModelRunner"]

# Set Log level
logging.getLogger().setLevel(logging.ERROR)

# DESIGN IDEA: Add an special operator, execute static program inside operator.
#
# Op‘s Inputs:
#   - the input variable of the user feed
#   - the necessary parameters of the network
# Op's Outputs:
#   - the output variable of fetch
# 
# This op receives a complete program desc, internally creates scope
# and executor, executes this program. Key points:
#
# 1. Data Sharing: 
#   The varBase of the dynamic graph is not in the scope, so before the op
#   executes the program internally, create persistent variables with the
#   same name as feed, parameters, and fetch in the scope, and share the
#   LoDTensor of the op input.
# 
# 2. Forward and Backward Separation:
#   Because the dynamic graph op performs the forward and backward separately,
#   the forward program is used as the execution object of the forward op,
#   and the reverse program is used as the execution object of the grad op.
#
# TODO:
# - add prefix for all params
# - insert feed, fetch into bwd_program
# - sort var in feed and fetch
# - self and non-self design
# - 


class StaticModelRunner(layers.Layer):
    def __init__(self, model_dir, model_filename=None, params_filename=None):
        super(StaticModelRunner, self).__init__()

        # Step 0. key variable definitions
        # the variable name of the feed & fetch itself
        # self._feed_var_name = None
        # self._fetch_var_name = None
        # the layer outputs var desc
        self._output_descs = []
        # input, output, params name list
        self._input_names = []
        self._output_names = []
        self._param_names = []

        # Step 1. load program desc from disk
        # the saved model hold feed, fetch op, no need, can be remove
        self._program_desc = self._load_static_model(model_dir, model_filename)
        # self._print_program_desc(self._program_desc)
        # logging.info("feed: %s" % self._feed_var_name)
        # logging.info("fetch: %s" % self._fetch_var_name)

        # Step 2. load all parameters
        self._load_persisitable_dict(model_dir, params_filename)

        # Step 3. generate backwar program desc
        self._bwd_program_desc = self._generate_backward(self._program_desc,
                                                         self._output_descs)
        # self._bwd_program_desc = self._generate_backward_desc()
        # self._print_program_desc(self._bwd_program_desc)

    def forward(self, inputs):
        # Step 1. check inputs
        if not isinstance(inputs, dict):
            raise TypeError(
                "The type of inputs in StaticModelRunner.forward must be dict, but received %s."
                % type(inputs))
        for key, value in inputs.items():
            if not isinstance(key, six.string_types):
                raise TypeError(
                    "The type of inputs.key in StaticModelRunner.forward must be str, but received %s."
                    % type(key))
            # key should be valid name
            if key not in self._all_var_names():
                raise ValueError(
                    "The variable name %s is not in the loaded program." % key)
            if not isinstance(value, (np.ndarray, core.VarBase)):
                raise TypeError(
                    "The type of inputs.value in StaticModelRunner.forward must be numpy array or Variable(VarBase), but received %s."
                    % type(value))
            # NOTE: In order to unify the API, firstly convert the input to VarBase
            if isinstance(value, np.ndarray):
                var = core.VarBase(
                    value=value,
                    name=key,
                    persistable=False,
                    place=framework._current_expected_place(),
                    zero_copy=True)
                var.stop_gradient = True
                inputs[key] = var

        # Step 2. run prorgam by op
        # build inputs, outputs and attrs
        input_vars = []
        params = []
        output_vars = []
        for var_name, var in inputs.items():
            self._input_names.append(var_name)
            input_vars.append(var)
        for param_name, param in self._parameters.items():
            params.append(param)
        for var_desc in self._output_descs:
            var = core.VarBase(var_desc.dtype(),
                               var_desc.shape(),
                               var_desc.name(), var_desc.type(), False)
            var.stop_gradient = False
            output_vars.append(var)
        # hold forward variables
        out_scope = core.VarBase(core.VarDesc.VarType.FP32, [],
                                 "program_out_scope",
                                 core.VarDesc.VarType.STEP_SCOPES, True)

        # NOTE: TraceOp attr checker receive attr=[] check failed
        op_attrs = {}
        op_attrs['fwd_block'] = self._program_desc.block(0)
        op_attrs['bwd_block'] = self._bwd_program_desc.block(0)
        if len(self._input_names) > 0:
            op_attrs['input_var_names'] = self._input_names
        if len(self._param_names) > 0:
            op_attrs['param_names'] = self._param_names
        if len(self._output_names) > 0:
            op_attrs['output_var_names'] = self._output_names

        # run op
        framework._dygraph_tracer().trace_op(
            type='run_program',
            inputs={'X': input_vars,
                    'Params': params},
            outputs={'Out': output_vars,
                     'OutScope': [out_scope]},
            attrs=op_attrs)

        outs = output_vars
        if len(output_vars) == 1:
            outs = output_vars[0]
        return outs

    def _generate_backward_desc(self):
        with framework._static_graph_guard():
            # Step 1. create backward program desc
            assert self._program_desc is not None
            bwd_program_desc = core.ProgramDesc()
            for i in six.moves.range(1, self._program_desc.num_blocks()):
                parent_idx = self._program_desc.block(i).parent
                bwd_program_desc.append_block(
                    bwd_program_desc.block(parent_idx))

            # Step 2. prepare program and related var
            # NOTE: To reuse backward interfaces, build Program firstly.
            # Originally, there is no need to build a program, but need to almost
            # rewrite a series of methods for append_backward for program_desc. 
            # Therefore, in order to reuse the method of backward.py, build the program here.
            fwd_program = self._build_program_by_desc(self._program_desc)
            bwd_program = self._build_program_by_desc(bwd_program_desc)

            # targets = []
            # for var in fwd_program.list_vars():
            #     if var.name in self._output_names:
            #         targets.append(var)

            targets = []
            for out in self._output_descs:
                targets.append(fwd_program.global_block().var(out.name()))

            # Step 3. calc gradients
            block = targets[0].block
            block_idx = block.idx
            target_block = bwd_program.block(block_idx)

            # target_gradients = [None] * len(targets)
            no_grad_dict = backward._get_stop_gradients_(fwd_program)

            # target_grad_map = {}
            # for i, grad in enumerate(target_gradients):
            #     target = targets[i]
            #     grad_name = backward._append_grad_suffix_(target.name)
            #     target_shape = target.name + '_shape'
            #     block.desc.append_op().copy_from(
            #         backward._create_op_desc_("shape", {'Input': [target.name]},
            #                         {"Out": [target_shape]}, {}))
            #     op_desc = backward._create_op_desc_("fill_constant",
            #                             {"ShapeTensor": [target_shape]},
            #                             {"Out": [grad_name]}, {
            #                                 "shape": target.shape,
            #                                 "value": 1.0,
            #                                 "dtype": target.dtype,
            #                             })
            #     block.desc.append_op().copy_from(op_desc)

            block_no_grad_set = set(
                map(backward._strip_grad_suffix_, no_grad_dict[0]))
            op_path = backward._find_op_path_(block, targets, [],
                                              block_no_grad_set)
            no_grad_dict[0].update(
                list(map(backward._append_grad_suffix_, block_no_grad_set)))

            grad_to_var = dict()
            grad_info_map = dict()
            backward._append_backward_ops_(block, op_path, target_block,
                                           no_grad_dict, grad_to_var)

            backward._rename_grad_(target_block, 0, grad_to_var, {})

            self._generate_backward_vars(target_block, 0, grad_to_var,
                                         grad_info_map)

            self._append_output_grad_vars(bwd_program)

            # Step 5. sync two program
            fwd_program._sync_with_cpp()
            bwd_program._sync_with_cpp()

            return bwd_program.desc

    def _generate_backward(self, fwd_program_desc, outputs):
        with framework._static_graph_guard():
            # Step 1. create backward program desc
            bwd_program_desc = core.ProgramDesc()
            # for i in six.moves.range(1, fwd_program_desc.num_blocks()):
            #     parent_idx = fwd_program_desc.block(i).parent
            #     bwd_program_desc.append_block(bwd_program_desc.block(parent_idx))

            # Step 2. prepare program and related var
            # NOTE: To reuse backward interfaces, build Program firstly.
            # Originally, there is no need to build a program, but need to almost
            # rewrite a series of methods for append_backward for program_desc. 
            # Therefore, in order to reuse the method of backward.py, build the program here.
            fwd_program = self._build_program_by_desc(fwd_program_desc)
            bwd_program = self._build_program_by_desc(bwd_program_desc)

            out_vars = []
            for out in outputs:
                out_vars.append(fwd_program.global_block().var(out.name()))

            block = fwd_program.block(0)
            target_block = bwd_program.block(0)

            # TODO: no_grad_dict? stop_gradient attr is not in desc 
            no_grad_dict = backward._get_stop_gradients_(fwd_program)

            #logging.info("no grad dict: {}".format(no_grad_dict))

            # Step 3. generate backward program
            block_no_grad_set = set(
                map(backward._strip_grad_suffix_, no_grad_dict[0]))
            op_path = backward._find_op_path_(block, out_vars, [],
                                              block_no_grad_set)

            # logging.info("block_no_grad_set: {}".format(block_no_grad_set))
            logging.info("------------------------\nop path:")
            for op in op_path:
                logging.info(program_utils.op_to_code(op))
            logging.info("------------------------")

            no_grad_vars = backward._find_no_grad_vars(block, op_path, out_vars,
                                                       block_no_grad_set)

            # logging.info("no_grad_vars: {}".format(no_grad_vars))

            block_no_grad_set.update(no_grad_vars)
            no_grad_dict[0].update(
                list(map(backward._append_grad_suffix_, block_no_grad_set)))

            # logging.info("block_no_grad_set: {}".format(block_no_grad_set))
            # logging.info("no grad dict: {}".format(no_grad_dict))

            grad_to_var = dict()

            # backward._append_backward_ops_(
            #     block,  # the block where forward ops are in
            #     op_path,
            #     target_block,
            #     no_grad_dict,
            #     grad_to_var)
            self._generate_backward_ops(
                block,  # the block where forward ops are in
                op_path,
                target_block,
                no_grad_dict,
                grad_to_var)

            logging.info("grad_to_var: {}".format(grad_to_var))

            grad_info_map = dict()

            # Because append_backward may be called multiple times,
            # we need rename the internal gradient variables so that they have
            # different names.
            backward._rename_grad_(target_block, 0, grad_to_var, {})

            self._generate_backward_vars(target_block, 0, grad_to_var,
                                         grad_info_map)

            # Step 4. insert feed, fetch op for backward
            # feed, fetch_list, feed_var_name, fetch_var_name = \
            #     self._prepare_feed_fetch_var()
            # bwd_program = self._insert_feed_fetch_ops(bwd_program, feed, fetch_list, 
            #     feed_var_name, fetch_var_name)
            self._append_output_grad_vars(bwd_program)

            # Step 5. sync two program
            fwd_program._sync_with_cpp()
            bwd_program._sync_with_cpp()

            return bwd_program.desc

    def _append_output_grad_vars(self, bwd_program):
        global_block = bwd_program.global_block()

        output_grad_names = list(
            map(backward._append_grad_suffix_, self._output_names))

        for i, name in enumerate(output_grad_names):
            global_block.create_var(
                name=name,
                shape=self._output_descs[i].shape(),
                dtype=self._output_descs[i].dtype(),
                type=self._output_descs[i].type(),
                persistable=False)

    def _build_program_by_desc(self, program_desc):
        with framework._static_graph_guard():
            prog = framework.Program()
            prog.desc = program_desc
            prog.blocks = [
                framework.Block(prog, i)
                for i in six.moves.range(prog.desc.num_blocks())
            ]
            prog._sync_with_cpp()
        return prog

    def _generate_backward_ops(self, block, ops, target_block, no_grad_dict,
                               grad_to_var):

        # grad_op_descs holds created grad_op, and will be appended to target_block
        grad_op_descs = []
        program = block.program
        target_program = target_block.program

        # add grad_op_desc by reversed ops
        for op in reversed(ops):
            grad_sub_block_list = []
            # If the op has its own sub-block, deal with the sub-block first
            if op.has_attr("sub_block"):
                sub_block = program.block(op._block_attr_id("sub_block"))
                grad_sub_block = target_program._create_block()
                # TODO: no forward block, set to parent block temporarily
                grad_sub_block._set_forward_block_idx(sub_block.parent_idx)
                sub_block_path = backward._get_sub_block_path(
                    sub_block, op, no_grad_dict[sub_block.idx])
                self._generate_backward_ops(sub_block, sub_block_path,
                                            grad_sub_block, no_grad_dict,
                                            grad_to_var)

                program._rollback()
                grad_sub_block_list.append(grad_sub_block.desc)

            # Getting op's corresponding grad_op
            grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
                op.desc,
                cpt.to_text(no_grad_dict[block.idx]), grad_sub_block_list)

            # Set device for grad_op according to forward Op
            device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName(
            )
            op_device = op.desc.attr(device_attr_name)
            for op_desc in grad_op_desc:
                op_desc._set_attr(device_attr_name, op_device)

            grad_op_descs.extend(grad_op_desc)
            grad_to_var.update(op_grad_to_var)

        # sum parameter's gradients' var given multiple var gradient
        grad_op_descs = backward._addup_repetitive_outputs_(grad_op_descs,
                                                            block.idx)

        # if all outputs of the grad op are in no_grad_set, then just remove and fill zero
        # if all inputs of the grad op are in no_grad_set, just remove this op
        grad_op_descs = backward._remove_no_grad_branch_(
            grad_op_descs, no_grad_dict[block.idx])

        # remove some backward ops
        not_need_ops = backward._find_not_need_ops(grad_op_descs, ops, {})

        grad_op_descs = [
            op_desc for op_desc in grad_op_descs if op_desc not in not_need_ops
        ]

        # append op_desc in grad_op_descs to target_block
        op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
        backward_role = core.op_proto_and_checker_maker.OpRole.Backward
        for op_desc in grad_op_descs:
            new_op_desc = target_block.desc.append_op()
            new_op_desc.copy_from(op_desc)
            new_op_desc._set_attr(op_role_attr_name, backward_role)
            grad_to_var["__current_op_desc__"] = new_op_desc

    def _generate_backward_vars(self, block, start_op_idx, grad_to_var,
                                grad_info_map):
        ops_to_remove = []
        '''
        NOTE(paddle-dev): while_grad op may hold some inputs which are not found 
        in the parent/forward block, and they are also the outputs of while_grad 
        op. These kinds of inputs are the recursive outputs inside while_grad op. 
        They should be considered as "already created" when scanning the inner 
        ops of while_grad ops.  
        '''
        parent_op = backward._find_parent_op_(block)
        parent_op_vars = []
        if parent_op is not None:
            input_args = parent_op.input_arg_names()
            output_args = parent_op.output_arg_names()
            for in_arg in input_args:
                if in_arg in output_args:
                    parent_op_vars.append(in_arg)

        for op_idx in range(start_op_idx, block.desc.op_size()):
            op_desc = block.desc.op(op_idx)
            if op_desc.has_attr("sub_block"):
                sub_block = block.program.block(
                    op_desc._block_attr_id("sub_block"))
                self._generate_backward_vars(sub_block, 0, grad_to_var,
                                             grad_info_map)

            grad_var_ins = [
                var for var in op_desc.input_arg_names()
                if backward._is_grad_var_(var)
            ]
            grad_var_outs = [
                var for var in op_desc.output_arg_names()
                if backward._is_grad_var_(var)
            ]

            inputs = [
                var for var in op_desc.input_arg_names()
                if var != core.empty_var_name()
            ]
            outputs = [
                var for var in op_desc.output_arg_names()
                if var != core.empty_var_name()
            ]

            # If the outputs of grad op is empty, just remove it 
            if not outputs:
                ops_to_remove.append(op_idx)
                continue

            new_vars = set()
            # create new gradient variables
            for grad_var_name in op_desc.output_arg_names():
                if block.desc.has_var_recursive(cpt.to_bytes(
                        grad_var_name)) or grad_var_name == core.empty_var_name(
                        ):
                    continue
                block.desc.var(cpt.to_bytes(grad_var_name))
                new_vars.add(grad_var_name)
                if grad_var_name not in grad_to_var:
                    continue
                grad_info_map[grad_to_var[grad_var_name]] = (grad_var_name,
                                                             block)
            # infer_shape and infer_type
            # op_desc.infer_var_type(block.desc)
            # op_desc.infer_shape(block.desc)

            # for arg in op_desc.output_arg_names():
            #     if arg in new_vars:
            #         backward._infer_var_data_type_shape_(arg, block)

        for op_idx in reversed(ops_to_remove):
            block.desc._remove_op(op_idx, op_idx + 1)

    def _var_desc(self, block_idx, name):
        cur_blcok_idx = block_idx
        name = cpt.to_bytes(name)
        var_desc = self._program_desc.block(cur_blcok_idx).find_var(name)
        while var_desc is None:
            parent_blcok_idx = self._program_desc.block(cur_blcok_idx).parent
            logging.info("cur_blcok_idx: %d" % parent_blcok_idx)
            if cur_blcok_idx == -1:
                logging.info("var %s is not find." % name)
                break
            var_desc = self._program_desc.block(parent_blcok_idx).find_var(name)
            cur_blcok_idx = parent_blcok_idx
        return var_desc

    def _all_var_names(self):
        var_names = set()
        for i in six.moves.range(self._program_desc.num_blocks()):
            block = self._program_desc.block(i)
            for var in block.all_vars():
                var_names.add(var.name())
        return var_names

    def _load_static_model(self, model_dir, model_filename=None):
        # Step 1. dir and filename check
        load_dirname = os.path.normpath(model_dir)
        if not os.path.isdir(load_dirname):
            raise ValueError("There is no directory named '%s'" % load_dirname)

        if model_filename is not None:
            model_filename = os.path.basename(model_filename)
        else:
            model_filename = "__model__"
        model_filename = os.path.join(load_dirname, model_filename)

        # Step 2. parse program desc
        with open(model_filename, "rb") as f:
            program_desc_str = f.read()

        program_desc = core.ProgramDesc(program_desc_str)
        if not core._is_program_version_supported(program_desc._version()):
            raise ValueError("Unsupported program version: %d\n" %
                             program_desc._version())

        # Step 3. set all `is_test` attributes to False
        self._change_is_test_status(program_desc)

        # Step 4. record feed, fetch and remove useless scale-1 op
        ops_to_remove = []
        # replace_name_dict = {}
        root_block = program_desc.block(0)
        for i in six.moves.range(root_block.op_size()):
            op = root_block.op(i)
            if op.type() == 'feed':
                ops_to_remove.append(i)
                # feed_in_var_name = op.input('X')[0]
                # if self._feed_var_name is not None:
                #     assert self._feed_var_name == feed_in_var_name
                # self._feed_var_name = feed_in_var_name
            # remove useless scale-1 op
            if op.type() == 'scale' and op.output('Out')[0].startswith(
                    'save_infer_model/scale_'):
                ops_to_remove.append(i)
                out_var_name = cpt.to_bytes(op.output('Out')[0])
                root_block._remove_var(out_var_name)
                # if hold fetch, record dict info
                # replace_name_dict[out_var_name] = [cpt.to_bytes(x) for x in op.input('X')]
                # record fetch targets var desc
                self._output_names.append(cpt.to_bytes(op.input('X')[0]))
                self._output_descs.append(
                    root_block.find_var(cpt.to_bytes(op.input('X')[0])))
            if op.type() == 'fetch' and op.input('X')[0].startswith(
                    'save_infer_model/scale_'):
                ops_to_remove.append(i)
                # if hold fetch, record dict info
                # in_var_name = cpt.to_bytes(op.input('X')[0])
                # op.set_input('X', replace_name_dict[in_var_name])
                # fetch_out_var_name = op.output('Out')[0]
                # if self._fetch_var_name is not None:
                #     assert self._fetch_var_name == fetch_out_var_name
                # self._fetch_var_name = fetch_out_var_name

        for op_idx in reversed(ops_to_remove):
            root_block._remove_op(op_idx, op_idx + 1)

        return program_desc

    def _is_persistable(self, var_desc):
        if var_desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
                var_desc.type() == core.VarDesc.VarType.FETCH_LIST or \
                var_desc.type() == core.VarDesc.VarType.READER or \
                var_desc.type() == core.VarDesc.VarType.RAW:
            return False
        return var_desc.persistable()

    def _is_parameter(self, persis_var_desc):
        assert self._program_desc is not None
        for block_idx in six.moves.range(self._program_desc.num_blocks()):
            block = self._program_desc.block(block_idx)
            for op_idx in six.moves.range(block.op_size()):
                op = block.op(op_idx)
                # NOTE: parameter is not the output of any op (no optimizer)
                if persis_var_desc.name() in op.output_arg_names():
                    return False
        for block_idx in six.moves.range(self._program_desc.num_blocks()):
            block = self._program_desc.block(block_idx)
            for op_idx in six.moves.range(block.op_size()):
                op = block.op(op_idx)
                # NOTE: parameter is the input of a certain op
                if persis_var_desc.name() in op.input_arg_names():
                    return True
        return False

    def _change_is_test_status(self, program_desc):
        # change all `is_test` attributes to True
        for i in six.moves.range(program_desc.num_blocks()):
            block = program_desc.block(i)
            for j in six.moves.range(block.op_size()):
                op = block.op(j)
                if op.has_attr('is_test'):
                    op._set_attr('is_test', False)

    def _append_loaded_suffix(self, name):
        """
        Append grad suffix to the given variable name
        e.g. x ==> x@LOADED
        """
        return cpt.to_text(name) + core.loaded_var_suffix()

    def _append_loaded_suffix_to_param(self, param_desc):
        old_name = param_desc.name()
        new_name = self._append_loaded_suffix(param_desc.name())
        param_desc.set_name(new_name)
        for block_idx in six.moves.range(self._program_desc.num_blocks()):
            block = self._program_desc.block(block_idx)
            for op_idx in six.moves.range(block.op_size()):
                op = block.op(op_idx)
                op._rename_input(old_name, new_name)
                op._rename_output(old_name, new_name)

    def _load_persisitable_dict(self, model_dir, params_filename=None):
        load_dirname = os.path.normpath(model_dir)
        assert self._program_desc is not None
        # TODO: other blocks?
        persis_vars = list(
            filter(self._is_persistable, self._program_desc.block(0).all_vars(
            )))
        load_var_map = {}
        for each_var in persis_vars:
            orig_each_name = each_var.name()
            self._append_loaded_suffix_to_param(each_var)
            # create output varbase
            new_var = framework.ParamBase(
                shape=each_var.shape(),
                dtype=each_var.dtype(),
                name=each_var.name(),
                type=each_var.type(),
                persistable=True)
            if params_filename is None:
                if not self._is_parameter(each_var):
                    continue
                # logging.info("persis var name %s" % each_var.name())
                framework._dygraph_tracer().trace_op(
                    type='load',
                    inputs={},
                    outputs={'Out': new_var},
                    attrs={
                        'file_path': os.path.join(load_dirname, orig_each_name)
                    })
                new_var.stop_gradient = False
                self.add_parameter(name=new_var.name, parameter=new_var)
                self._param_names.append(new_var.name)
            else:
                load_var_map[each_var.name()] = new_var

        if params_filename is not None:
            load_var_list = []
            for name in sorted(load_var_map.keys()):
                load_var_list.append(load_var_map[name])

            framework._dygraph_tracer().trace_op(
                type='load_combine',
                inputs={},
                outputs={'Out': load_var_list},
                attrs={
                    'file_path': os.path.join(load_dirname, params_filename)
                })

            for each_var in persis_vars:
                if not self._is_parameter(each_var):
                    continue
                param = load_var_map[each_var.name()]
                param.stop_gradient = False
                self.add_parameter(name=param.name, parameter=param)
                self._param_names.append(param.name)

    def _fill_param_with_var(self, param, var):
        param.value().get_tensor().set(var.numpy(),
                                       framework._current_expected_place())

    # def _prepare_feed_fetch_var(self):
    #     # Step 1. prepare feed fetch var name
    #     feed_grad_var_name = backward._append_grad_suffix_(self._feed_var_name)
    #     fetch_grad_var_name = backward._append_grad_suffix_(self._fetch_var_name)

    #     logging.info("feed grad var: %s" % feed_grad_var_name)
    #     logging.info("fetch grad var: %s" % fetch_grad_var_name)

    #     # prepare outputs grad var name
    #     out_grad_name_list = []
    #     for out_desc in self._output_descs:
    #         out_grad_name_list.append(out_desc.name())
    #     out_grad_name_list = list(map(backward._append_grad_suffix_, out_grad_name_list))

    #     logging.info(out_grad_name_list)

    #     # prepare param grad var
    #     param_grad_name_list = []
    #     for param_name in self._parameters:
    #         param_grad_name_list.append(param_name)
    #     param_grad_name_list = list(map(backward._append_grad_suffix_, param_grad_name_list))

    #     logging.info(param_grad_name_list)

    #     return out_grad_name_list, param_grad_name_list, feed_grad_var_name, fetch_grad_var_name

    # def _insert_feed_fetch_ops(self, program, feed, fetch_list, feed_var_name,
    #                         fetch_var_name):
    #     tmp_program = program.clone()

    #     global_block = tmp_program.global_block()

    #     if feed_var_name in global_block.vars:
    #         feed_var = global_block.var(feed_var_name)
    #     else:
    #         feed_var = global_block.create_var(
    #             name=feed_var_name,
    #             type=core.VarDesc.VarType.FEED_MINIBATCH,
    #             persistable=True)

    #     if fetch_var_name in global_block.vars:
    #         fetch_var = global_block.var(fetch_var_name)
    #     else:
    #         fetch_var = global_block.create_var(
    #             name=fetch_var_name,
    #             type=core.VarDesc.VarType.FETCH_LIST,
    #             persistable=True)

    #     # prepend feed operators
    #     if not executor.has_feed_operators(global_block, feed, feed_var_name):
    #         for i, name in enumerate(feed):
    #             out = global_block.create_var(
    #                 name=name,
    #                 shape=self._output_descs[i].shape(),
    #                 dtype=self._output_descs[i].dtype(),
    #                 type=self._output_descs[i].type(),
    #                 persistable=False)
    #             global_block._prepend_op(
    #                 type='feed',
    #                 inputs={'X': [feed_var]},
    #                 outputs={'Out': [out]},
    #                 attrs={'col': i})

    #     # append fetch_operators
    #     if not executor.has_fetch_operators(global_block, fetch_list, fetch_var_name):
    #         for i, name in enumerate(fetch_list):
    #             # assert isinstance(var, Variable) or isinstance(
    #             #     var, six.string_types), (
    #             #         "Wrong type for fetch_list[%s]: %s" % (i, type(var)))
    #             var = global_block.var(name)
    #             global_block.append_op(
    #                 type='fetch',
    #                 inputs={'X': [var]},
    #                 outputs={'Out': [fetch_var]},
    #                 attrs={'col': i})

    #     return tmp_program

    ##### Debug Functions #####

    def _op_desc_to_string(self, desc):
        protostr = desc.serialize_to_string()
        proto = framework_pb2.OpDesc.FromString(six.binary_type(protostr))
        return framework._debug_string_(proto, True)

    def _var_desc_to_string(self, desc):
        protostr = desc.serialize_to_string()
        proto = framework_pb2.VarDesc.FromString(six.binary_type(protostr))
        return framework._debug_string_(proto, True)

    def _print_program_desc(self, desc):
        with framework._static_graph_guard():
            p = framework.Program()
            p.desc = desc
            p.blocks = [
                framework.Block(p, i)
                for i in six.moves.range(p.desc.num_blocks())
            ]
            p._sync_with_cpp()
            program_utils.program_to_code(p)

    def _print_all_op_info(self):
        for i in six.moves.range(self._program_desc.num_blocks()):
            block = self._program_desc.block(i)
            for j in six.moves.range(block.op_size()):
                op = block.op(j)
                print(self._op_desc_to_string(op))

    def _print_execute_op_info(self, op_type, inputs, outputs, attrs):
        logging.info("-------------------------")
        logging.info("Op Name: {}".format(op_type))
        logging.info("Inputs: {}".format(inputs))
        logging.info("Outputs: {}".format(outputs))
        logging.info("Attrs: {}".format(attrs))
        logging.info("-------------------------")

    def _print_outputs_info(self):
        logging.info("---------output names-----------")
        for out in self._output_descs:
            logging.info(self._var_desc_to_string(out))
        logging.info("--------------------")
