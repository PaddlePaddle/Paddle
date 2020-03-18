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
from .. import Executor
from .. import framework
from .. import io
from ..proto import framework_pb2
from ... import compat as cpt

__all__ = ["StaticModelRunner"]

no_kernel_op_set = {"read_from_array", "write_to_array"}

control_flow_op_set = {"conditional_block", "while"}

# Set Log level
logging.getLogger().setLevel(logging.ERROR)


class StaticModelRunner(layers.Layer):
    def __init__(self, model_dir, model_filename=None, params_filename=None):
        super(StaticModelRunner, self).__init__()
        # Step 1. load program desc from disk
        self._program_desc = self._load_static_model(model_dir, model_filename,
                                                     params_filename)
        # Step 2. load all parameters
        self._load_persisitable_dict(self._program_desc, model_dir,
                                     params_filename)
        # Step 3. load network and config
        self._output_names = set()
        self._root_block_op_info = self._parse_root_block_ops()
        # NOTE: the inputs and outputs of op with sub-block also in root block
        self._sub_blocks_op_info = self._parse_sub_blocks_ops()
        # debug info
        self._print_execute_info()

        # Other object variable
        self._cur_inputs = {}

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
                inputs[key] = var

        # Step 2. run ops
        self._cur_inputs.update(inputs)
        self._cur_inputs.update(self._parameters)
        # TODO：Some variables may not only be used as input one time. 
        # Here records the number of times all variables in the entire network
        # will be used as input. If the number of times is zero, the variable
        # can be cleaned
        # TODO: Each block maintains its own variable life cycle
        # root_block_var_life_count = collections.defaultdict(int)
        for op_type, inputs_dict, outputs_dict, attrs in self._root_block_op_info:
            # self._print_execute_op_info(op_type, inputs_dict, outputs_dict, attrs)
            if self._is_control_flow_op(op_type):
                logging.info("----------------\n- encounter control flow op: %s"
                             % op_type)
                if op_type == "while":
                    self._run_while_block(inputs_dict, outputs_dict, attrs)
                elif op_type == "conditional_block":
                    self._run_cond_block()
            else:
                op_outputs = self._run_op(0, op_type, inputs_dict, outputs_dict,
                                          attrs)

            # add outputs into current inputs
            for var in op_outputs.values():
                self._cur_inputs[var.name] = var

        # Step 3. contruct outputs
        # NOTE: Only need to fetch targets as outputs
        outputs = []
        for var_name in self._output_names:
            assert var_name in self._cur_inputs
            outputs.append(self._cur_inputs[var_name])
        if len(outputs) == 1:
            outputs = outputs[0]

        # Step 4. clear cur input dict
        self._cur_inputs.clear()

        return outputs

    def _run_while_block(self, sub_inputs_dict, sub_outputs_dict, sub_attrs):
        # get target block
        sub_block_idx = int(sub_attrs['sub_block'])
        cond_var_name = sub_inputs_dict['Condition'][0]
        assert cond_var_name in self._cur_inputs
        cond_var = self._cur_inputs[cond_var_name]
        while cond_var.numpy()[0] == True:
            for op_type, inputs_dict, outputs_dict, attrs in self._sub_blocks_op_info[
                    sub_block_idx - 1]:
                self._print_execute_op_info(op_type, inputs_dict, outputs_dict,
                                            attrs)
                if self._is_control_flow_op(op_type):
                    logging.info(
                        "----------------\n- encounter control flow op: %s" %
                        op_type)
                    if op_type == "while":
                        self._run_while_block(inputs_dict, outputs_dict, attrs)
                    elif op_type == "conditional_block":
                        self._run_cond_block()
                else:
                    op_outputs = self._run_op(sub_block_idx, op_type,
                                              inputs_dict, outputs_dict, attrs)

                # add outputs into current inputs
                for var in op_outputs.values():
                    self._cur_inputs[var.name] = var

            cond_var = self._cur_inputs[cond_var_name]
            logging.info("while cond value: %r" % cond_var.numpy()[0])

    def _run_cond_block(self):
        pass

    def _run_op(self, block_idx, op_type, inputs_dict, outputs_dict, attrs):
        # Step 1. build op's inputs
        skip_cur_op = False
        # del_var_list = []
        op_inputs = {}
        for var_key, var_names in inputs_dict.items():
            if len(var_names) == 0:
                continue
            # TODO: some variable have multiple arguments
            if len(var_names) > 1:
                logging.warning("Op %s's input %s has multiple args:" %
                                (op_type, var_key))
                for var_name in var_names:
                    logging.warning("- %s" % var_name)
            # NOTE: all args are valid input
            vars = []
            for var_name in var_names:
                if var_name in self._cur_inputs:
                    var = self._cur_inputs[var_name]
                    # TODO: check shape, there is -1 axis
                    # self._check_var_shape(self._var_desc(var_name), var)
                    vars.append(var)
                    # TODO: remove used input var from current input
                    # if var_life_count[var_name] > 0:
                    #     var_life_count[var_name] -= 1
                    # if var_life_count[var_name] == 0:
                    #     del_var_list.append(self._cur_inputs.pop(var_name))
                else:
                    # cannot find op's input, so skip this op
                    logging.info("- cannot find %s's input %s, skip it" %
                                 (op_type, var_names[0]))
                    skip_cur_op = True
                    break
            if len(vars) == 1:
                vars = vars[0]
            op_inputs[var_key] = vars

        # Step 2. build op's outputs
        op_outputs = {}
        # user may only input part of feed targets
        if skip_cur_op is True:
            return op_outputs

        for var_key, var_names in outputs_dict.items():
            # TODO: can output be null?
            # TODO: can output have multiple name?
            # TODO: need to deal with other block?
            # NOTE: dtype, dims, name, type, persistable=False
            assert len(var_names) == 1
            name = var_names[0]
            var_desc = self._var_desc(block_idx, name)
            # NOTE: If the output variable already exists, use it directly
            if name in self._cur_inputs:
                var = self._cur_inputs[name]
            else:
                var = core.VarBase(var_desc.dtype(),
                                   var_desc.shape(), name,
                                   var_desc.type(), False)
                var.stop_gradient = False
            op_outputs[var_key] = var

        # Step 3. run op
        # logging.info("-------------------")
        logging.info("- execute op %s" % op_type)
        # for k, v in self._cur_inputs.items():
        #     logging.info("- %s stop_gradient: %r" % (v.name, v.stop_gradient))
        # NOTE: some op can't be executed directly
        if self._is_no_kernel_op(op_type):
            logging.info("----------------\n- encounter no kernel op: %s" %
                         op_type)
            if op_type == "write_to_array":
                self._write_to_array(op_inputs, op_outputs, attrs)
            elif op_type == "read_from_array":
                self._read_from_array(op_inputs, op_outputs, attrs)
        else:
            framework._dygraph_tracer().trace_op(op_type, op_inputs, op_outputs,
                                                 attrs)

        # TODO: delete useless var
        # for var in del_var_list:
        #     del var

        return op_outputs

    def _write_to_array(self, inputs, outputs, attrs):
        assert len(outputs) == 1
        tensor_array = outputs['Out'].value().get_lod_tensor_array()
        index = inputs['I'].numpy()[0]
        tensor = inputs['X'].value().get_tensor()
        if index >= len(tensor_array):
            tensor_array._resize(index + 1)
        tensor_array[index] = tensor

    def _read_from_array(self, inputs, outputs, attrs):
        assert len(outputs) == 1
        tensor = outputs['Out'].value().get_tensor()
        tensor_array = inputs['X'].value().get_lod_tensor_array()
        index = inputs['I'].numpy()[0]
        if index >= len(tensor_array):
            raise ValueError(
                "index is out or range in _read_from_array, index: %d, array len: %d"
                % (index, len(tensor_array)))
        tensor.set(tensor_array[index].__array__(),
                   framework._current_expected_place())

    def _parse_sub_blocks_ops(self):
        result_list = []
        for i in six.moves.range(1, self._program_desc.num_blocks()):
            sub_block = self._program_desc.block(i)
            result_list.append(self._parse_block_ops(sub_block))
        return result_list

    def _parse_root_block_ops(self):
        root_block = self._program_desc.block(0)
        return self._parse_block_ops(root_block)

    def _parse_block_ops(self, block):
        result_list = []
        for i in six.moves.range(block.op_size()):
            op = block.op(i)
            if op.type() == 'feed' or op.type() == 'fetch':
                continue
            # remove useless scale-1 op
            if op.type() == 'scale' and op.output(op.output_names()[0])[
                    0].startswith('save_infer_model/scale_'):
                # record fetch targets variable name
                self._output_names.add(op.input('X')[0])
                continue
            inputs = {}
            for i in six.moves.range(len(op.input_names())):
                input_name = op.input_names()[i]
                inputs[input_name] = op.input(input_name)
                # TODO: Output as input count
                # tmp_var_names = op.input(input_name)
                # for tmp_var_name in tmp_var_names:
                #     self._var_life_count[tmp_var_name] += 1
            outputs = {}
            for i in six.moves.range(len(op.output_names())):
                output_name = op.output_names()[i]
                outputs[output_name] = op.output(output_name)
            attrs = {}
            attr_names = sorted(op.attr_names())
            for i in six.moves.range(len(attr_names)):
                name = attr_names[i]
                if name == "op_callstack":
                    continue
                attr_type = op.attr_type(name)
                if attr_type == core.AttrType.BLOCK:
                    attrs[name] = "{value}".format(
                        value=op._block_attr_id(name))
                elif attr_type == core.AttrType.BLOCKS:
                    attrs[name] = "blocks[{value}]".format(
                        value=op._block_attr_ids(name))
                else:
                    attrs[name] = op.attr(name)
            op_info_tuple = (op.type(), inputs, outputs, attrs)
            result_list.append(op_info_tuple)
        return result_list

    def _is_control_flow_op(self, op_type):
        global control_flow_op_set
        if op_type in control_flow_op_set:
            return True
        return False

    def _is_no_kernel_op(self, op_type):
        global no_kernel_op_set
        if op_type in no_kernel_op_set:
            return True
        return False

    def _check_var_shape(self, var_desc, var):
        logging.info("-----------------\n%s:" % var_desc.name())
        expected_shape = var_desc.shape()
        true_shape = var.shape
        logging.info("expected shape: {}".format(expected_shape))
        logging.info("true shape: {}".format(true_shape))
        for i, value in enumerate(expected_shape):
            if value != true_shape[i]:
                raise ValueError("The shape of input variable {} is invalid. \
                    expected shape is {}, but receiveed shape is {}.".format(
                    var_desc.name(), expected_shape, true_shape))

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

    def _next_execute_op_info(self):
        def __reader__():
            root_block = self._program_desc.block(0)
            for i in six.moves.range(root_block.op_size()):
                op = root_block.op(i)
                if op.type() == 'feed' or op.type() == 'fetch':
                    continue
                # remove useless scale-1 op
                if op.type() == 'scale' and op.output(op.output_names()[0])[
                        0].startswith('save_infer_model/scale_'):
                    # record fetch targets variable name
                    self._output_names.add(op.input('X')[0])
                    continue
                inputs = {}
                for i in six.moves.range(len(op.input_names())):
                    input_name = op.input_names()[i]
                    inputs[input_name] = op.input(input_name)
                outputs = {}
                for i in six.moves.range(len(op.output_names())):
                    output_name = op.output_names()[i]
                    outputs[output_name] = op.output(output_name)
                attrs = {}
                attr_names = sorted(op.attr_names())
                for i in six.moves.range(len(attr_names)):
                    name = attr_names[i]
                    if name == "op_callstack":
                        continue
                    attr_type = op.attr_type(name)
                    if attr_type == core.AttrType.BLOCK:
                        attrs[name] = "block[{value}]".format(
                            value=op._block_attr_id(name))
                    elif attr_type == core.AttrType.BLOCKS:
                        attrs[name] = "blocks[{value}]".format(
                            value=op._block_attr_ids(name))
                    else:
                        attrs[name] = op.attr(name)
                yield op.type(), inputs, outputs, attrs

        return __reader__()

    def _load_static_model(self,
                           model_dir,
                           model_filename=None,
                           params_filename=None):
        # Step 1. dir and filename check
        load_dirname = os.path.normpath(model_dir)
        if not os.path.isdir(load_dirname):
            raise ValueError("There is no directory named '%s'", dirname)

        if model_filename is not None:
            model_filename = os.path.basename(model_filename)
        else:
            model_filename = "__model__"
        model_filename = os.path.join(load_dirname, model_filename)

        if params_filename is not None:
            params_filename = os.path.basename(params_filename)

        # Step 2. parse program desc
        with open(model_filename, "rb") as f:
            program_desc_str = f.read()

        program_desc = core.ProgramDesc(program_desc_str)
        if not core._is_program_version_supported(program_desc._version()):
            raise ValueError("Unsupported program version: %d\n" %
                             program_desc._version())

        # Step 3. change all `is_test` attributes to False
        self._change_is_test_status(program_desc)

        return program_desc

    def _is_persistable(self, var_desc):
        if var_desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
                var_desc.type() == core.VarDesc.VarType.FETCH_LIST or \
                var_desc.type() == core.VarDesc.VarType.READER or \
                var_desc.type() == core.VarDesc.VarType.RAW:
            return False
        return var_desc.persistable()

    def _change_is_test_status(self, program_desc):
        # change all `is_test` attributes to True
        for i in six.moves.range(program_desc.num_blocks()):
            block = program_desc.block(i)
            for j in six.moves.range(block.op_size()):
                op = block.op(j)
                if op.has_attr('is_test'):
                    op._set_attr('is_test', False)

    def _load_persisitable_dict(self,
                                program_desc,
                                model_dir,
                                params_filename=None):
        load_dirname = os.path.normpath(model_dir)
        # TODO: 
        # 1. control flow need to be deal with
        # 2. persistable var may not be parameter
        # 3. params_filename is not none
        # 4. new parameter has redundant copy here
        persis_vars = list(
            filter(self._is_persistable, program_desc.block(0).all_vars()))
        for each_var in persis_vars:
            # logging.info("persis var name %s" % each_var.name())
            attrs = {'file_path': os.path.join(load_dirname, each_var.name())}
            out = core.ops.load({}, attrs)
            var = out['Out'][0]
            var.name = each_var.name()
            # logging.info(var.name)
            # logging.info(var)
            param = framework.ParamBase(
                name=var.name, shape=var.shape, dtype=var.dtype)
            # there is redundant copy here
            self._fill_param_with_var(param, var)
            param = self.add_parameter(name=param.name, parameter=param)
            param.stop_gradient = False
            # del useless var
            del var

    def _fill_param_with_var(self, param, var):
        param.value().get_tensor().set(var.numpy(),
                                       framework._current_expected_place())

    ##### Debug Functions #####

    def _op_desc_to_string(self, desc):
        protostr = desc.serialize_to_string()
        proto = framework_pb2.OpDesc.FromString(six.binary_type(protostr))
        return framework._debug_string_(proto, True)

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

    def _print_execute_info(self):
        logging.info("---------root block op info-----------")
        for t in self._root_block_op_info:
            logging.info("{}".format(t))
        logging.info("---------sub block op info-----------")
        for i, block in enumerate(self._sub_blocks_op_info):
            logging.info("block {}:".format(i))
            for t in block:
                logging.info("{}".format(t))
        # logging.info("---------var life count-----------")
        # for k, v in self._var_life_count.items():
        #     logging.info("{}: {}".format(k, v))
        logging.info("---------output names-----------")
        logging.info(self._output_names)
        logging.info("--------------------")
