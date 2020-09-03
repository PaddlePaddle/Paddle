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

import logging
import numpy as np
import os
import six

from . import layers
from .. import core
from .. import framework
from .. import backward

from ..layers import nn
from .base import switch_to_static_graph
from ... import compat as cpt

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


class StaticModelRunner(layers.Layer):
    """
    A Dynamic graph Layer for loading inference program and related parameters,
    and then performing fine-tune training or inference.

    The loaded program and parameters are saved by `fluid.io.save_inference_model`.

    .. note::
        **1. Dynamic graph mode do not support LoDTensor. 
             All original static graph model's feed targets or parametars 
             that depend on LoD are temporarily unavailable.**
        **2. All saved inference model's feed targets need be given.**
        **3. The ``stop_gradient`` information is lost and can not be recovered.**
        **4. The parameter's ``trainable`` information is lost and can not be recovered.**
        **5. Double gradient model is not supported now.**
        **6. Now only supports loading models saved by `fluid.io.save_inference_model`.**

    Args:
        model_dir(str): The directory path where the model is saved.
        model_filename(str, optional): The file name of saved inference program. 
                                       If set to None, a default filename is
                                       :code:`__model__`.
                                       The default value is None.
        params_filename(str, optional): The file name of saved all related parameters.
                                        If set to None, parameters are saved
                                        in separate files. 
                                        The default value is None.

    Returns:
        Layer: A Layer object can run loaded program.

    Examples:
      .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        BATCH_SIZE = 32
        BATCH_NUM = 20
        SAVE_DIRNAME = "fc.inference.model"

        def random_batch_reader():
            def _get_random_images_and_labels(image_shape, label_shape):
                image = np.random.random(size=image_shape).astype('float32')
                label = np.random.random(size=label_shape).astype('int64')
                return image, label

            def __reader__():
                for _ in range(BATCH_NUM):
                    batch_image, batch_label = _get_random_images_and_labels(
                        [BATCH_SIZE, 784], [BATCH_SIZE, 1])
                    yield batch_image, batch_label

            return __reader__

        def train_and_save_static_model(place):
            img = fluid.data(name='img', shape=[None, 784], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')

            pred = fluid.layers.fc(input=img, size=10, act='softmax')

            loss = fluid.layers.cross_entropy(input=pred, label=label)
            avg_loss = fluid.layers.mean(loss)

            optimizer = fluid.optimizer.SGD(learning_rate=0.001)
            optimizer.minimize(avg_loss)

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            loader = fluid.io.DataLoader.from_generator(
                feed_list=[img, label], capacity=5, iterable=True)
            loader.set_batch_generator(random_batch_reader(), places=place)

            for data in loader():
                exe.run(
                    fluid.default_main_program(),
                    feed=data, 
                    fetch_list=[avg_loss])

            # save model by fluid.io.save_inference_model
            fluid.io.save_inference_model(
                SAVE_DIRNAME, ["img"], [pred], exe)


        # Step 1. train and save inference model in static graph mode
        place = fluid.CPUPlace()
        train_and_save_static_model(place)

        # Step 2. load inference model in dygraph and fine-tune
        with fluid.dygraph.guard(place):
            fc = fluid.dygraph.static_runner.StaticModelRunner(SAVE_DIRNAME)

            sgd = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=fc.parameters())

            train_loader = fluid.io.DataLoader.from_generator(capacity=5)
            train_loader.set_batch_generator(
                random_batch_reader(), places=place)

            for data in train_loader():
                img = data[0]
                label = data[1]
                label.stop_gradient = True

                cost = fc(inputs=img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()
                sgd.minimize(avg_loss)
    """

    def __init__(self, model_dir, model_filename=None, params_filename=None):
        super(StaticModelRunner, self).__init__()

        # Step 0. key variable definitions
        # loaded inference program desc
        self._infer_program_desc = None
        # recovered train program desc
        self._train_program_desc = None
        # StaticModelRunner executed program desc,
        # switch infer or train by train() and eval()
        self._trace_program_desc = None
        self._inner_scope = core.Scope()
        # the layer outputs var desc
        self._output_descs = []
        # input, output, params name list
        self._input_names = []
        self._output_names = []
        self._param_names = []
        # train or eval flag
        self._is_test = False

        # Step 1. load program desc from disk
        # the saved model hold feed, fetch & scale op, no need, can be remove
        self._infer_program_desc = self._load_static_model(model_dir,
                                                           model_filename)

        # Step 2. load all parameters
        self._load_persisitable_dict(model_dir, params_filename)

        # Step 3. generate backwar program desc
        self._train_program_desc = self._append_backward_desc()

        # Step 4. recheck parameters stop gradients
        self._recheck_stop_gradients()

        # Step 5. set default mode to train
        self.train()

    def train(self):
        self._is_test = False
        self._trace_program_desc = self._train_program_desc

    def eval(self):
        self._is_test = True
        self._trace_program_desc = self._infer_program_desc

    def forward(self, *args):
        """
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
        # Step 1. prepare inputs, outputs, attrs
        input_vars = []
        for i, value in enumerate(args):
            if not isinstance(value, (np.ndarray, core.VarBase)):
                raise TypeError(
                    "The type of inputs.value in StaticModelRunner.forward must be numpy array or Variable(VarBase), but received %s."
                    % type(value))
            # NOTE: In order to unify the API, firstly convert the input to VarBase
            if isinstance(value, np.ndarray):
                var = core.VarBase(
                    value=value,
                    name=self._input_names[i],
                    persistable=False,
                    place=framework._current_expected_place(),
                    zero_copy=True)
            else:
                var = value
                # TODO: here may have important name set by user
                var.name = self._input_names[i]
            input_vars.append(var)

        params = []
        for param in self._parameters.values():
            params.append(param)

        output_vars = []
        for var_desc in self._output_descs:
            var = core.VarBase(var_desc.dtype(),
                               var_desc.shape(),
                               var_desc.name(), var_desc.type(), False)
            output_vars.append(var)

        # hold forward variables
        tmp_scope_vec = core.VarBase(core.VarDesc.VarType.FP32, [],
                                     "program_out_scope",
                                     core.VarDesc.VarType.STEP_SCOPES, True)
        tmp_scope_vec.value().set_scope(self._inner_scope)

        # Step 2. run prorgam by op
        framework._dygraph_tracer().trace_op(
            type='run_program',
            inputs={'X': input_vars,
                    'Params': params},
            outputs={'Out': output_vars,
                     'OutScope': tmp_scope_vec},
            attrs={
                'global_block': self._trace_program_desc.block(0),
                'start_op_index': 0,
                'end_op_index': self._infer_program_desc.block(0).op_size(),
                'is_test': self._is_test
            })

        # NOTE: [ why need set param's gradient type here ]
        # if user set sparse gradient mode, the param's gradient
        # will be SelectedRows, not LoDTensor. But tracer will just
        # set param grad VarBase by forward VarBase(LoDTensor)
        # If we don't change grad_var type here, RunProgramOp need
        # transform SelectedRows to LoDTensor forcely, it may not
        # be user wanted result.
        for param in params:
            grad_name = param.name + core.grad_var_suffix()
            grad_var = self._trace_program_desc.block(0).find_var(
                cpt.to_bytes(grad_name))
            # NOTE: cannot find var desc maybe no problem, such as in batch_norm
            if grad_var is None:
                continue
            param._set_grad_type(grad_var.type())

        # Step 3. prepare output, keep same form with inputs
        outs = output_vars
        if len(output_vars) == 1:
            outs = output_vars[0]
        return outs

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

        # Step 3. 
        # - remove feed, fetch and useless scale-1 op
        # - remove op_callstack attr
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
                self._output_names.append(cpt.to_bytes(op.input('X')[0]))
                self._output_descs.append(
                    root_block.find_var(cpt.to_bytes(op.input('X')[0])))
            elif op.type() == 'fetch':
                ops_to_remove.append(i)
                fetch_var_name = cpt.to_bytes(op.output('Out')[0])
                root_block._remove_var(fetch_var_name)
                # NOTE: some old pre-train models have no extra scale_op
                if not op.input('X')[0].startswith('save_infer_model/scale_'):
                    self._output_names.append(cpt.to_bytes(op.input('X')[0]))
                    self._output_descs.append(
                        root_block.find_var(cpt.to_bytes(op.input('X')[0])))
            else:
                if op.has_attr("op_callstack"):
                    op.remove_attr("op_callstack")

        for op_idx in reversed(ops_to_remove):
            root_block._remove_op(op_idx, op_idx + 1)

        # NOTE: reverse feed vars
        self._input_names.reverse()

        # Step 4. add scale for outputs
        tmp_program = self._build_program_by_desc(program_desc)
        self._append_scale_to_output(tmp_program)

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
            self._output_names[i] = var.name
            self._output_descs[i] = var.desc

    @switch_to_static_graph
    def _append_backward_desc(self):
        assert self._infer_program_desc is not None, "The StaticModelRunner not initialized properly."
        program_desc_copy = core.ProgramDesc(self._infer_program_desc)

        # Step 1. set all `is_test` attributes to False
        self._change_is_test_status(program_desc_copy, False)

        # Step 2. prepare program and related var
        # NOTE: To reuse backward interfaces, build Program firstly.
        # Originally, there is no need to build a program, but need to almost
        # rewrite a series of methods for append_backward for program_desc. 
        # Therefore, in order to reuse the method of backward.py, build the program here.
        fwd_op_num = program_desc_copy.block(0).op_size()
        program = self._build_program_by_desc(program_desc_copy)

        # TODO: could the targets be in sub block?
        targets = []
        for out in self._output_descs:
            targets.append(program.global_block().var(out.name()))

        # Step 3. append backward
        backward.gradients(targets=targets, inputs=[])
        return program.desc

    def _load_persisitable_dict(self, model_dir, params_filename=None):
        load_dirname = os.path.normpath(model_dir)
        assert self._infer_program_desc is not None, "The StaticModelRunner not initialized properly."

        persis_vars = self._get_persis_vars(self._infer_program_desc)
        load_var_map = {}
        for each_var in persis_vars:
            orig_each_name = each_var.name()
            # append suffix
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

    def _recheck_stop_gradients(self):
        assert self._train_program_desc is not None, "The StaticModelRunner not initialized properly."
        # NOTE: After loading the model, the stop_gradient information 
        # of the original variable is lost, but if a parameter does not
        # have a corresponding @GRAD variable in the backward program,
        # it can be said that it is also stop_gradient
        all_var_names = self._get_all_var_names(self._train_program_desc)
        for param_name in self._parameters:
            param_grad_name = param_name + core.grad_var_suffix()
            if param_grad_name not in all_var_names:
                self._parameters[param_name].stop_gradient = True

    def _get_all_var_names(self, program_desc):
        all_var_names = set()
        for i in six.moves.range(program_desc.num_blocks()):
            block = program_desc.block(i)
            for var in block.all_vars():
                all_var_names.add(var.name())
        return all_var_names

    def _get_persis_vars(self, program_desc):
        persis_vars = []
        for i in six.moves.range(program_desc.num_blocks()):
            block = program_desc.block(i)
            persis_vars.extend(
                list(filter(self._is_persistable, block.all_vars())))
        return persis_vars

    @switch_to_static_graph
    def _build_program_by_desc(self, program_desc):
        prog = framework.Program()
        prog.desc = program_desc
        prog.blocks = [
            framework.Block(prog, i)
            for i in six.moves.range(prog.desc.num_blocks())
        ]
        prog._sync_with_cpp()
        return prog

    def _is_persistable(self, var_desc):
        if var_desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
                var_desc.type() == core.VarDesc.VarType.FETCH_LIST or \
                var_desc.type() == core.VarDesc.VarType.READER or \
                var_desc.type() == core.VarDesc.VarType.RAW:
            return False
        return var_desc.persistable()

    def _is_parameter(self, persis_var_desc):
        assert self._infer_program_desc is not None, "The StaticModelRunner not initialized properly."
        # 1. firstly, param should be input of op
        input_ops = []  # op can be repeated
        for block_idx in six.moves.range(self._infer_program_desc.num_blocks()):
            block = self._infer_program_desc.block(block_idx)
            for op_idx in six.moves.range(block.op_size()):
                op = block.op(op_idx)
                # NOTE: parameter is the input of a certain op
                if persis_var_desc.name() in op.input_arg_names():
                    input_ops.append(op)
        # 2. secondly, param should not be output of op or be same op's output
        for block_idx in six.moves.range(self._infer_program_desc.num_blocks()):
            block = self._infer_program_desc.block(block_idx)
            for op_idx in six.moves.range(block.op_size()):
                op = block.op(op_idx)
                if persis_var_desc.name() in op.output_arg_names():
                    # such as batch_norm_op
                    if op in input_ops:
                        continue
                    else:
                        return False
        return True

    def _change_is_test_status(self, program_desc, is_test):
        # change all `is_test` attributes
        for i in six.moves.range(program_desc.num_blocks()):
            block = program_desc.block(i)
            for j in six.moves.range(block.op_size()):
                op = block.op(j)
                if op.has_attr('is_test'):
                    op._set_attr('is_test', is_test)

    def _append_loaded_suffix(self, name):
        """
        Append grad suffix to the given variable name
        e.g. x ==> x@LOADED
        """
        suffix = core.loaded_var_suffix()
        name = cpt.to_text(name)
        if suffix not in name:
            name = name + suffix
        return name

    def _append_loaded_suffix_to_param(self, param_desc):
        old_name = param_desc.name()
        new_name = self._append_loaded_suffix(param_desc.name())
        param_desc.set_name(new_name)
        for block_idx in six.moves.range(self._infer_program_desc.num_blocks()):
            block = self._infer_program_desc.block(block_idx)
            for op_idx in six.moves.range(block.op_size()):
                op = block.op(op_idx)
                op._rename_input(old_name, new_name)
                op._rename_output(old_name, new_name)
