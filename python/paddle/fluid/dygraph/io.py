# Copyright (c) 20w0 PaddlePaddle Authors. All Rights Reserved.
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

import os
import six
import pickle
import logging
import numpy as np
from collections import OrderedDict

from paddle import compat as cpt
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid import backward
from paddle.fluid.dygraph import layers
from paddle.fluid.layers import nn
from paddle.fluid.dygraph.base import switch_to_static_graph

VARIABLE_FILENAME = "__variables__"
EXTRA_VAR_INFO_FILENAME = "__variables.info__"


def _load_program_desc(model_file_path):
    # 1. parse program desc
    with open(model_file_path, "rb") as f:
        program_desc_str = f.read()

    program_desc = core.ProgramDesc(program_desc_str)
    if not core._is_program_version_supported(program_desc._version()):
        raise ValueError("Unsupported program version: %d\n" %
                         program_desc._version())

    return program_desc


def _is_persistable(var_desc):
    if var_desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
            var_desc.type() == core.VarDesc.VarType.FETCH_LIST or \
            var_desc.type() == core.VarDesc.VarType.READER or \
            var_desc.type() == core.VarDesc.VarType.RAW:
        return False
    return var_desc.persistable()


def _is_parameter(persis_var_desc, program_desc):
    # 1. firstly, param should be input of op
    input_ops = []  # op can be repeated
    for block_idx in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(block_idx)
        for op_idx in six.moves.range(block.op_size()):
            op = block.op(op_idx)
            # NOTE: parameter is the input of a certain op
            if persis_var_desc.name() in op.input_arg_names():
                input_ops.append(op)
    # 2. secondly, param should not be output of op or be same op's output
    for block_idx in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(block_idx)
        for op_idx in six.moves.range(block.op_size()):
            op = block.op(op_idx)
            if persis_var_desc.name() in op.output_arg_names():
                # such as batch_norm_op
                if op in input_ops:
                    continue
                else:
                    return False
    return True


def _get_persis_vars(program_desc):
    persis_vars = []
    for i in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(i)
        persis_vars.extend(list(filter(_is_persistable, block.all_vars())))
    return persis_vars


def _get_persis_var_names(program_desc):
    """
    Get all persistable variable names in ProgramDesc.
    """
    var_names = []
    persis_vars = _get_persis_vars(program_desc)
    for var in persis_vars:
        var_names.append(var.name())
    return var_names


def _get_all_var_names(program_desc):
    all_var_names = set()
    for i in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(i)
        for var in block.all_vars():
            all_var_names.add(var.name())
    return all_var_names


def _append_loaded_suffix(name):
    """
    Append grad suffix to the given variable name
    e.g. x ==> x@LOADED
    """
    suffix = core.loaded_var_suffix()
    name = cpt.to_text(name)
    if suffix not in name:
        name = name + suffix
    return name


def _remove_loaded_suffix(name):
    """
    Remove grad suffix to the given variable name
    e.g. x@LOADED ==> x
    """
    suffix = core.loaded_var_suffix()
    name = cpt.to_text(name)
    return name.replace(suffix, '')


def _append_loaded_suffix_to_var(program_desc):
    persis_vars = _get_persis_vars(program_desc)
    for var_desc in persis_vars:
        old_name = var_desc.name()
        new_name = _append_loaded_suffix(var_desc.name())
        var_desc.set_name(new_name)
        for block_idx in six.moves.range(program_desc.num_blocks()):
            block = program_desc.block(block_idx)
            for op_idx in six.moves.range(block.op_size()):
                op = block.op(op_idx)
                op._rename_input(old_name, new_name)
                op._rename_output(old_name, new_name)


@switch_to_static_graph
def _build_program_by_desc(program_desc):
    prog = framework.Program()
    prog.desc = program_desc
    prog.blocks = [
        framework.Block(prog, i)
        for i in six.moves.range(prog.desc.num_blocks())
    ]
    prog._sync_with_cpp()
    return prog


def _change_is_test_status(program_desc, is_test):
    # change all `is_test` attributes
    for i in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(i)
        for j in six.moves.range(block.op_size()):
            op = block.op(j)
            if op.has_attr('is_test'):
                op._set_attr('is_test', is_test)


class _ProgramHolder(object):
    """
    Holds the execution information of a Program.
    """

    def __init__(self, program_desc):
        super(_ProgramHolder, self).__init__()

        # input, output, persistable var info
        self._input_names = []
        self._persis_names = []
        self._output_descs = []

        # execution scope
        self._inner_scope = core.Scope()

        # forward program
        self._infer_program_desc = self._preprocess(program_desc)
        # forward + backward program
        self._train_program_desc = self._append_backward_desc(
            self._infer_program_desc)

    @property
    def infer_program(self):
        return self._infer_program_desc

    @property
    def train_program(self):
        return self._train_program_desc

    @property
    def input_names(self):
        return self._input_names

    @property
    def output_decs(self):
        return self._output_descs

    @property
    def persistable_names(self):
        return self._persis_names

    @property
    def scope(self):
        return self._inner_scope

    def _preprocess(self, program_desc):
        # 1. Crop original program
        # remove feed, fetch and scale-1 op, remove op_callstack attr
        ops_to_remove = []
        root_block = program_desc.block(0)
        for i in six.moves.range(root_block.op_size()):
            op = root_block.op(i)
            if op.type() == 'feed':
                ops_to_remove.append(i)
                feed_var_name = cpt.to_bytes(op.input('X')[0])
                root_block._remove_var(feed_var_name)
                self._input_names.append(cpt.to_bytes(op.output('Out')[0]))
            elif op.type() == 'scale' and op.output('Out')[0].startswith(
                    'save_infer_model/scale_'):
                ops_to_remove.append(i)
                out_var_name = cpt.to_bytes(op.output('Out')[0])
                root_block._remove_var(out_var_name)
                self._output_descs.append(
                    root_block.find_var(cpt.to_bytes(op.input('X')[0])))
            elif op.type() == 'fetch':
                ops_to_remove.append(i)
                fetch_var_name = cpt.to_bytes(op.output('Out')[0])
                root_block._remove_var(fetch_var_name)
                # NOTE: some old pre-train models have no extra scale_op
                if not op.input('X')[0].startswith('save_infer_model/scale_'):
                    self._output_descs.append(
                        root_block.find_var(cpt.to_bytes(op.input('X')[0])))
            else:
                if op.has_attr("op_callstack"):
                    op.remove_attr("op_callstack")

        for op_idx in reversed(ops_to_remove):
            root_block._remove_op(op_idx, op_idx + 1)

        # 2. Input processing, reverse feed vars
        self._input_names.reverse()

        # 3. Output processing, add scale for outputs
        tmp_program = _build_program_by_desc(program_desc)
        self._append_scale_to_output(tmp_program)

        # 4. Persistable vars processing
        # - append @LOADED suffix to persistable vars
        # NOTE: [why need to append suffix to persistable vars]
        # Dygraph and static graph mode use the same naming mechanism. 
        # If users want to load the model fine-tune, it is possible 
        # to add the existing Layer in the loaded model to enhance 
        # the network. For example, the original saved model has linear, 
        # and later after loading, a new linear is added. At this time, 
        # there will be a problem of duplicate names, so here is unified 
        # to add the LOADED suffix to the parameters of the model loaded
        #  during training.
        _append_loaded_suffix_to_var(program_desc)
        # - get persistable var
        self._persis_names = _get_persis_var_names(program_desc)

        return program_desc

    @switch_to_static_graph
    def _append_scale_to_output(self, program):
        # 1. append scale & save var
        scale_output_vars = []
        with framework.program_guard(program):
            for i, out in enumerate(self._output_descs):
                var = program.global_block().var(out.name())
                var = nn.scale(
                    var, 1., name="static_model_runner/scale_{}".format(i))
                scale_output_vars.append(var)
        # 2. update output names & descs
        for i, var in enumerate(scale_output_vars):
            self._output_descs[i] = var.desc

    @switch_to_static_graph
    def _append_backward_desc(self, infer_program_desc):
        program_desc_copy = core.ProgramDesc(infer_program_desc)

        # 1. set all `is_test` attributes to False
        _change_is_test_status(program_desc_copy, False)

        # 2. prepare program and related var
        # NOTE: To reuse backward interfaces, build Program firstly.
        # Originally, there is no need to build a program, but need to almost
        # rewrite a series of methods for append_backward for program_desc. 
        # Therefore, in order to reuse the method of backward.py, build the program here.
        fwd_op_num = program_desc_copy.block(0).op_size()
        program = _build_program_by_desc(program_desc_copy)

        # TODO: could the targets be in sub block?
        targets = []
        for out in self._output_descs:
            targets.append(program.global_block().var(out.name()))

        # 3. append backward
        backward.gradients(targets=targets, inputs=[])
        return program.desc


# [ TranslatedLayer executed in imperative mode ]
# 
# DESIGN IDEA: Add an special operator, execute static program inside operator.
#
# Op's Inputs:
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


# NOTE: [compatible] deal with model saved by save_inference_model,
# which need get var info from program desc
def _load_persis_vars_by_program(model_path,
                                 program_holder,
                                 params_filename=None):
    # make sure the path has been checked
    persis_vars = _get_persis_vars(program_holder.infer_program)
    load_var_dict = {}
    for each_var in persis_vars:
        orig_each_name = _remove_loaded_suffix(each_var.name())
        # create output varbase
        new_var = framework.ParamBase(
            shape=each_var.shape(),
            dtype=each_var.dtype(),
            name=each_var.name(),
            type=each_var.type(),
            persistable=True)
        if params_filename is None:
            if not _is_parameter(each_var, program_holder.infer_program):
                continue
            framework._dygraph_tracer().trace_op(
                type='load',
                inputs={},
                outputs={'Out': new_var},
                attrs={'file_path': os.path.join(model_path, orig_each_name)})
            new_var.stop_gradient = False
        load_var_dict[each_var.name()] = new_var

    if params_filename is not None:
        load_var_list = []
        for name in sorted(load_var_dict.keys()):
            load_var_list.append(load_var_dict[name])

        framework._dygraph_tracer().trace_op(
            type='load_combine',
            inputs={},
            outputs={'Out': load_var_list},
            attrs={'file_path': os.path.join(model_path, params_filename)})

        for each_var in persis_vars:
            if not _is_parameter(each_var, program_holder.infer_program):
                continue
            param = load_var_dict[each_var.name()]
            param.stop_gradient = False

    # NOTE: After loading the model, the stop_gradient information 
    # of the original variable is lost, but if a parameter does not
    # have a corresponding @GRAD variable in the backward program,
    # it can be said that it is also stop_gradient
    all_var_names = _get_all_var_names(program_holder.train_program)
    for var_name in load_var_dict:
        grad_var_name = var_name + core.grad_var_suffix()
        if grad_var_name not in all_var_names:
            load_var_dict[var_name].stop_gradient = True

    return load_var_dict


def _load_persis_vars(var_file_path, var_info_path):
    # 1. load extra var info
    with open(var_info_path, 'rb') as f:
        extra_var_info = pickle.load(f) if six.PY2 else pickle.load(
            f, encoding='latin1')

    # 2. construct var dict
    load_var_dict = dict()
    load_var_list = []
    # NOTE: some var may not be Parameter
    for name in sorted(extra_var_info):
        # append suffix, see [why need to append suffix to persistable vars]
        new_name = _append_loaded_suffix(name)
        # create output varbase
        if extra_var_info[name].get('trainable', None) is not None:
            # use default shape and dtype
            new_var = framework.ParamBase(
                shape=[1],  # only to pass check
                dtype=core.VarDesc.VarType.FP32,
                name=new_name,
                persistable=True)
        else:
            new_var = framework._varbase_creator(
                name=new_name, persistable=True)
        new_var.stop_gradient = extra_var_info[name]['stop_gradient']
        load_var_dict[new_name] = new_var
        load_var_list.append(new_var)

    # 3. load all vars
    framework._dygraph_tracer().trace_op(
        type='load_combine',
        inputs={},
        outputs={'Out': load_var_list},
        attrs={'file_path': var_file_path})

    return load_var_dict


def _construct_program_holders(model_path, model_filename=None):
    # make sure the path has been checked
    program_holder_dict = dict()

    if model_filename is not None:
        # [compatible] if assign model_filename, only can load one program as Layer.forward
        model_filename = os.path.basename(model_filename)
        model_file_path = os.path.join(model_path, model_filename)
        program_holder_dict['forward'] = _ProgramHolder(
            _load_program_desc(model_file_path))
    else:
        for _, _, file_names in os.walk(model_path):
            for name in file_names:
                if 'model' in name:
                    model_file_path = os.path.join(model_path, name)
                    method_name = name.strip('_')
                    if method_name == 'model':
                        method_name = 'forward'
                    else:
                        method_name.replace('model', '')
                    program_holder_dict[method_name] = _ProgramHolder(
                        _load_program_desc(model_file_path))

    return program_holder_dict


def _construct_params_and_buffers(model_path, programs, params_filename=None):
    var_info_path = os.path.join(model_path, EXTRA_VAR_INFO_FILENAME)
    if os.path.exists(var_info_path):
        if params_filename is not None:
            var_file_path = os.path.join(model_path, params_filename)
        else:
            var_file_path = os.path.join(model_path, VARIABLE_FILENAME)
        var_dict = _load_persis_vars(var_file_path, var_info_path)
    else:
        var_dict = _load_persis_vars_by_program(model_path, programs['forward'],
                                                params_filename)
    return var_dict


class TranslatedLayer(layers.Layer):
    """
    jit.save and jit.load related Class.

    Avoid exposing TranslatedLayer class interface to users


    Executed forward part of StaticModelRunner Layer.
        Generally execute directly using the Layer object.

        Args:
            args(tuple(np.ndarray|Variable)): the inputs of StaticModelRunner.
                The order of input variables needs to be the same as the order 
                of feed variables when using `save_inference_model` to save model.
        
        Returns:
            Variable|list[Variable]: The forward outputs of StaticModelRunner Layer.
                If there is only one output, return Variable;
                if there are multiple outputs, return list[Variable].
    """

    def __init__(self, programs, persistable_vars):
        super(TranslatedLayer, self).__init__()

        if not isinstance(programs, dict):
            raise TypeError(
                "TranslatedLayer need to use _ProgramHolder's dict for initialization."
            )
        if not isinstance(persistable_vars, dict):
            raise TypeError(
                "TranslatedLayer need to use persisatbale variable dict for initialization."
            )

        self._program_holder_dict = programs

        for name, var in persistable_vars.items():
            if isinstance(var, framework.ParamBase):
                self.add_parameter(name, var)
            else:
                # TODO(chenweihang): support add buffers
                raise TypeError(
                    "Adding persistent variable which  to layer is not supported now"
                )

        self._is_test = True

    @staticmethod
    @framework.dygraph_only
    def _construct(model_path, configs=None):
        # 0. dir and filename check
        model_path = os.path.normpath(model_path)
        if not os.path.isdir(model_path):
            raise ValueError("There is no directory named '%s'" % model_path)
        model_filename = None
        params_filename = None
        if configs is not None:
            model_filename = configs.model_filename
            params_filename = configs.params_filename

        # 1. load program desc & construct _ProgramHolder
        programs = _construct_program_holders(model_path, model_filename)

        # 2. load layer parameters & parameter attirbutes
        persis_vars = _construct_params_and_buffers(model_path, programs,
                                                    params_filename)

        # 3. construct TranslatedLayer object
        translated_layer = TranslatedLayer(programs, persis_vars)

        # 4. create TranslatedLayer's execution method
        for method_name, program_holder in programs.items():
            setattr(TranslatedLayer, method_name,
                    TranslatedLayer._execution_method_creator(method_name,
                                                              program_holder))

        # 5. set TranslatedLayer's default mode to eval
        translated_layer.eval()

        return translated_layer

    @staticmethod
    def _execution_method_creator(method_name, program_holder):
        def __impl__(self, *input):
            # 1. prepare inputs, outputs, attrs
            input_vars = []
            for i, value in enumerate(input):
                if not isinstance(value, (np.ndarray, core.VarBase)):
                    raise TypeError(
                        "The type of input in TranslatedLayer must be numpy array or Variable(VarBase), but received %s."
                        % type(value))
                # NOTE: In order to unify the API, firstly convert the input to VarBase
                if isinstance(value, np.ndarray):
                    var = core.VarBase(
                        value=value,
                        name=program_holder.input_names[i],
                        persistable=False,
                        place=framework._current_expected_place(),
                        zero_copy=True)
                else:
                    var = value
                    # TODO: here may have important name set by user
                    var.name = program_holder.input_names[i]
                input_vars.append(var)

            persis_vars = []
            for var_name in program_holder.persistable_names:
                if var_name in self._parameters:
                    persis_vars.append(self._parameters[var_name])
                else:
                    # TODO(chenweihang): buffer support
                    raise ValueError(
                        "The persistable variable %s is not exists in current TranslatedLayer."
                        % var_name)

            output_vars = []
            for var_desc in program_holder.output_decs:
                var = core.VarBase(var_desc.dtype(),
                                   var_desc.shape(),
                                   var_desc.name(), var_desc.type(), False)
                output_vars.append(var)

            # hold forward variables
            tmp_scope_vec = core.VarBase(core.VarDesc.VarType.FP32, [],
                                         "program_out_scope",
                                         core.VarDesc.VarType.STEP_SCOPES, True)
            tmp_scope_vec.value().set_scope(program_holder.scope)

            # 2. run prorgam by op
            trace_program = program_holder.infer_program if self._is_test else program_holder.train_program
            end_op_index = program_holder.infer_program.block(0).op_size()
            framework._dygraph_tracer().trace_op(
                type='run_program',
                inputs={'X': input_vars,
                        'Params': persis_vars},
                outputs={'Out': output_vars,
                         'OutScope': tmp_scope_vec},
                attrs={
                    'global_block': trace_program.block(0),
                    'start_op_index': 0,
                    'end_op_index': end_op_index,
                    'is_test': self._is_test
                })

            # NOTE: [ why need set param's gradient type here ]
            # if user set sparse gradient mode, the param's gradient
            # will be SelectedRows, not LoDTensor. But tracer will just
            # set param grad VarBase by forward VarBase(LoDTensor)
            # If we don't change grad_var type here, RunProgramOp need
            # transform SelectedRows to LoDTensor forcely, it may not
            # be user wanted result.
            for persis_var in persis_vars:
                grad_var_name = var.name + core.grad_var_suffix()
                grad_var = trace_program.block(0).find_var(
                    cpt.to_bytes(grad_var_name))
                # NOTE: cannot find var desc maybe no problem, such as in batch_norm
                if grad_var is None:
                    continue
                persis_var._set_grad_type(grad_var.type())

            # Step 3. prepare output, keep same form with inputs
            outs = output_vars
            if len(output_vars) == 1:
                outs = output_vars[0]
            return outs

        __impl__.__name__ = method_name
        return __impl__

    def train(self):
        self._is_test = False

    def eval(self):
        self._is_test = True
