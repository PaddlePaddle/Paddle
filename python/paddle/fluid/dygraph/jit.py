# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ['TracedLayer', 'declarative', 'dygraph_to_static_func']

import logging
import numpy as np
import six
import paddle.fluid as fluid
from paddle.fluid import framework
from paddle import compat as cpt
from paddle.fluid import backward
from paddle.fluid import core
from paddle.fluid.compiler import CompiledProgram
from paddle.fluid.dygraph.base import program_desc_tracing_guard, switch_to_static_graph
from dygraph_to_static.program_translator import ProgramTranslator
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.executor import Executor, scope_guard
from paddle.fluid.framework import Program, Block, Variable, _dygraph_tracer, dygraph_only, _dygraph_guard, _current_expected_place, in_dygraph_mode
from paddle.fluid.wrapped_decorator import wrap_decorator

logger = logging.getLogger("fluid")


def create_program_from_desc(program_desc):
    program = Program()
    program.desc = program_desc
    program.blocks = [Block(program, 0)]
    program._sync_with_cpp()
    return program


def _extract_vars(inputs, result_list):
    if isinstance(inputs, Variable):
        result_list.append(inputs)

    if isinstance(inputs, (list, tuple)):
        for var in inputs:
            _extract_vars(var, result_list)


def extract_vars(inputs):
    result_list = []
    _extract_vars(inputs, result_list)
    return result_list


def _dygraph_to_static_func_(dygraph_func):
    """
    Converts imperative dygraph APIs into declarative function APIs. Decorator
    @dygraph_to_static_func only converts imperative dygraph APIs into
    declarative net-building APIs, which means it doesn't return immediate
    digital result as imperative mode. Users should handle Program and Executor
    by themselves.

    Note:
    This decorator is NOT our recommended way to transform imperative function
    to declarative function. We will remove this decorator after we finalize
    cleaning up code.

    Args:
        dygraph_func (callable): callable imperative function.

    Returns:
        Callable: converting imperative dygraph APIs into declarative
        net-building APIs.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          from paddle.fluid.dygraph.jit import dygraph_to_static_func

          @dygraph_to_static_func
          def func(x):
              if fluid.layers.mean(x) < 0:
                  x_v = x - 1
              else:
                  x_v = x + 1

               return x_v

          x = fluid.layers.fill_constant(shape=[3, 3], value=0, dtype='float64')

          x_v = func(x)
          exe = fluid.Executor(fluid.CPUPlace())
          out = exe.run(fetch_list=[x_v])
          print(out[0])
          # [[1. 1. 1.]
          #  [1. 1. 1.]
          #  [1. 1. 1.]]

    """

    # TODO: remove this decorator after we finalize training API
    def __impl__(*args, **kwargs):
        program_translator = ProgramTranslator()
        if in_dygraph_mode() or not program_translator.enable_declarative:
            logger.info(
                "The decorator 'dygraph_to_static_func' doesn't work in "
                "dygraph mode or set ProgramTranslator.enable to False. "
                "We will just return dygraph output.")
            return dygraph_func(*args, **kwargs)
        static_func = program_translator.get_func(dygraph_func)
        return static_func(*args, **kwargs)

    return __impl__


dygraph_to_static_func = wrap_decorator(_dygraph_to_static_func_)


def _declarative_(dygraph_func):
    """
    Converts imperative dygraph APIs into declarative function APIs. Decorator
    @declarative handles the Program and Executor of static mode and returns
    the result as a dygraph VarBase.

    Args:
        dygraph_func (callable): callable imperative function.

    Returns:
        VarBase: containing the numerical result.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          from paddle.fluid.dygraph.jit import declarative


          @declarative
          def func(x):
              x = fluid.dygraph.to_variable(x)
              if fluid.layers.mean(x) < 0:
                  x_v = x - 1
              else:
                  x_v = x + 1
              return x_v

          x = np.ones([1, 2])
          x_v = func(x)
          print(x_v.numpy()) # [[2. 2.]]

    """

    def __impl__(*args, **kwargs):
        program_translator = ProgramTranslator()
        # if in_dygraph_mode() or not program_translator.enable_declarative:
        #     logger.info(
        #         "The decorator 'declarative' doesn't work in dygraph "
        #         "mode or set ProgramTranslator.enable to False. We will "
        #         "just return dygraph output.")
        #     return dygraph_func(*args, **kwargs)
        program_translator = ProgramTranslator()
        return program_translator.get_output(dygraph_func, *args, **kwargs)

    return __impl__


declarative = wrap_decorator(_declarative_)


@dygraph_only
def _trace(layer,
           inputs,
           feed_prefix='feed_',
           fetch_prefix='fetch_',
           tmp_prefix='t_'):
    assert isinstance(layer, Layer)

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    tracer = _dygraph_tracer()._get_program_desc_tracer()

    var_list = extract_vars(inputs)

    with program_desc_tracing_guard(True):
        original_outputs = layer(*inputs)
        if not isinstance(original_outputs, (list, tuple)):
            outputs = [original_outputs]
        else:
            outputs = original_outputs
        out_vars = [var for var in outputs]

        program_desc, feed_names, fetch_names, parameters = tracer.create_program_desc(
            var_list, feed_prefix, out_vars, fetch_prefix, tmp_prefix)
        tracer.reset()

    with _dygraph_guard(None):
        program = create_program_from_desc(program_desc)

    return original_outputs, program, feed_names, fetch_names, parameters


class TracedLayer(object):
    """
    TracedLayer is used to convert a forward dygraph model to a static
    graph model. This is mainly used to save the dygraph model for online
    inference using C++. Besides, users can also do inference in Python
    using the converted static graph model, which usually has better
    performance than the original dygraph model.

    TracedLayer would run the static graph model using :code:`Executor`
    and :code:`CompiledProgram` . The static graph model would share
    parameters with the dygraph model.

    All TracedLayer objects should not be created by constructor and should
    be created by static method :code:`TracedLayer.trace(layer, inputs)` .

    The TracedLayer can only be used to convert the data-independent dygraph
    model into the static graph model, which means the dygraph model should
    be independent with the tensor data and shape.
    """

    def __init__(self, program, parameters, feed_names, fetch_names):
        self._program = program
        self._feed_names = feed_names
        self._fetch_names = fetch_names
        self._params = parameters

        self._place = _current_expected_place()

        self._scope = core.Scope()
        for p in parameters:
            src_tensor = p.value().get_tensor()
            dst_tensor = self._scope.var(p.name).get_tensor()
            dst_tensor._share_data_with(src_tensor)

        self._exe = Executor(self._place)
        self._compiled_program = None
        self._build_strategy = None
        self._exec_strategy = None

    @property
    def program(self):
        return self._program

    def _switch(self, is_test=True):
        for block_id in range(self._program.num_blocks):
            block = self._program.block(block_id)
            for op in block.ops:
                if op.has_attr("is_test"):
                    op._set_attr("is_test", is_test)

    @staticmethod
    @dygraph_only
    def trace(layer, inputs):
        """
        This method is the only allowed method to create TracedLayer object.
        It would call the :code:`layer(*inputs)` method to run the dygraph
        model and convert it into a static graph model.

        Args:
            layer (dygraph.Layer): the layer object to be traced.
            inputs (list(Variable)): the input variables of the layer object.

        Returns:
            tuple: A tuple of 2 items, whose the first item is the output of
            :code:`layer(*inputs)` , and the second item is the created
            TracedLayer object.

        Examples:
            .. code-block:: python:

                import paddle.fluid as fluid
                from paddle.fluid.dygraph import Linear, to_variable, TracedLayer
                import numpy as np

                class ExampleLayer(fluid.dygraph.Layer):
                    def __init__(self):
                        super(ExampleLayer, self).__init__()
                        self._fc = Linear(3, 10)

                    def forward(self, input):
                        return self._fc(input)

                with fluid.dygraph.guard():
                    layer = ExampleLayer()
                    in_np = np.random.random([2, 3]).astype('float32')
                    in_var = to_variable(in_np)
                    out_dygraph, static_layer = TracedLayer.trace(layer, inputs=[in_var])

                    # run the static graph model using Executor inside
                    out_static_graph = static_layer([in_var])

                    print(len(out_static_graph)) # 1
                    print(out_static_graph[0].shape) # (2, 10)

                    # save the static graph model for inference
                    static_layer.save_inference_model(dirname='./saved_infer_model')
        """
        outs, prog, feed, fetch, parameters = _trace(layer, inputs)
        traced = TracedLayer(prog, parameters, feed, fetch)
        return outs, traced

    def set_strategy(self, build_strategy=None, exec_strategy=None):
        """
        Set the strategies when running static graph model.

        Args:
            build_strategy (BuildStrategy, optional): build strategy of
                :code:`CompiledProgram` inside TracedLayer. Default None.
            exec_strategy (ExecutionStrategy, optional): execution strategy of
                :code:`CompiledProgram` inside TracedLayer. Default None.

        Returns:
            None

        Examples:
            .. code-block:: python:

                import paddle.fluid as fluid
                from paddle.fluid.dygraph import Linear, to_variable, TracedLayer
                import numpy as np

                class ExampleLayer(fluid.dygraph.Layer):
                    def __init__(self):
                        super(ExampleLayer, self).__init__()
                        self._fc = Linear(3, 10)

                    def forward(self, input):
                        return self._fc(input)

                with fluid.dygraph.guard():
                    layer = ExampleLayer()
                    in_np = np.random.random([2, 3]).astype('float32')
                    in_var = to_variable(in_np)

                    out_dygraph, static_layer = TracedLayer.trace(layer, inputs=[in_var])

                    build_strategy = fluid.BuildStrategy()
                    build_strategy.enable_inplace = True

                    exec_strategy = fluid.ExecutionStrategy()
                    exec_strategy.num_threads = 2

                    static_layer.set_strategy(build_strategy=build_strategy, exec_strategy=exec_strategy)
                    out_static_graph = static_layer([in_var])
        """
        assert self._compiled_program is None, "Cannot set strategy after run"
        self._build_strategy = build_strategy
        self._exec_strategy = exec_strategy

    @switch_to_static_graph
    def _compile(self):
        self._compiled_program = CompiledProgram(
            self._program).with_data_parallel(
                build_strategy=self._build_strategy,
                exec_strategy=self._exec_strategy,
                places=self._place)

    def _build_feed(self, inputs):
        assert isinstance(inputs, (list, tuple)), \
            "Inputs should be a list or tuple of variables"
        assert len(inputs) == len(self._feed_names)
        feed_dict = {}
        if in_dygraph_mode():
            for x, name in zip(inputs, self._feed_names):
                feed_dict[name] = x.value().get_tensor()
        else:
            for x, name in zip(inputs, self._feed_names):
                feed_dict[name] = x

        return feed_dict

    @switch_to_static_graph
    def _run(self, feed):
        return self._exe.run(self._compiled_program,
                             feed=feed,
                             fetch_list=self._fetch_names)

    def __call__(self, inputs):
        with scope_guard(self._scope):
            if self._compiled_program is None:
                self._compile()

            return self._run(self._build_feed(inputs))

    @switch_to_static_graph
    def save_inference_model(self, dirname, feed=None, fetch=None):
        """
        Save the TracedLayer to a model for inference. The saved
        inference model can be loaded by C++ inference APIs.

        Args:
            dirname (str): the directory to save the inference model.
            feed (list[int], optional): the input variable indices of the saved
                inference model. If None, all input variables of the
                TracedLayer object would be the inputs of the saved inference
                model. Default None.
            fetch (list[int], optional): the output variable indices of the
                saved inference model. If None, all output variables of the
                TracedLayer object would be the outputs of the saved inference
                model. Default None.

        Returns:
            None

        Examples:
            .. code-block:: python:

                import paddle.fluid as fluid
                from paddle.fluid.dygraph import Linear, to_variable, TracedLayer
                import numpy as np

                class ExampleLayer(fluid.dygraph.Layer):
                    def __init__(self):
                        super(ExampleLayer, self).__init__()
                        self._fc = Linear(3, 10)

                    def forward(self, input):
                        return self._fc(input)

                save_dirname = './saved_infer_model'
                in_np = np.random.random([2, 3]).astype('float32')

                with fluid.dygraph.guard():
                    layer = ExampleLayer()
                    in_var = to_variable(in_np)
                    out_dygraph, static_layer = TracedLayer.trace(layer, inputs=[in_var])
                    static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                program, feed_vars, fetch_vars = fluid.io.load_inference_model(save_dirname,
                                                    exe)

                fetch, = exe.run(program, feed={feed_vars[0]: in_np}, fetch_list=fetch_vars)
                print(fetch.shape) # (2, 10)
        """
        from paddle.fluid.io import save_inference_model

        def get_feed_fetch(all_vars, partial_vars):
            if partial_vars is None:
                return all_vars

            return [all_vars[idx] for idx in partial_vars]

        with scope_guard(self._scope):
            feeded_var_names = get_feed_fetch(self._feed_names, feed)
            target_var_names = get_feed_fetch(self._fetch_names, fetch)
            target_vars = []
            for name in target_var_names:
                target_var = self._program.global_block().vars.get(name, None)
                assert target_var is not None, "{} cannot be found".format(name)
                target_vars.append(target_var)

            save_inference_model(
                dirname=dirname,
                feeded_var_names=feeded_var_names,
                target_vars=target_vars,
                executor=self._exe,
                main_program=self._program.clone())


class ProgramContext(object):
    def __init__(self, main_program, input_names, output_names):
        self.input_names = input_names
        self.output_name = output_names
        self.program = main_program

    def _parse_persistable_params(self):
        program_desc = self.program.desc
        persistable_vars = self._get_persis_vars(program_desc)

    def _get_persis_vars(self, program_desc):
        persis_vars = []
        for i in six.moves.range(program_desc.num_blocks()):
            block = program_desc.block(i)
            persis_vars.extend(
                list(filter(self._is_persistable, block.all_vars())))
        return persis_vars

    def _is_persistable(self, var_desc):
        if var_desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
                var_desc.type() == core.VarDesc.VarType.FETCH_LIST or \
                var_desc.type() == core.VarDesc.VarType.READER or \
                var_desc.type() == core.VarDesc.VarType.RAW:
            return False
        return var_desc.persistable()


class PartialProgram(Layer):
    def __init__(self, traced_layer):
        super(PartialProgram, self).__init__()
        self._traced_layer = traced_layer

        self._input_names = []
        self._output_names = []
        self._params = []
        self._output_descs = []
        self._is_test = False

        self._inner_scope = self._traced_layer._scope
        self._inner_scope = core.Scope()
        self._check_input_output()

        self._program_desc = self._append_backward_desc()

    def _check_input_output(self):
        assert isinstance(self._traced_layer, TracedLayer)
        self._input_names = self._traced_layer._feed_names
        self._output_names = self._traced_layer._fetch_names
        self._params = self._traced_layer._params
        all_vars_map = self.program.global_block().vars
        self._output_descs = [
            all_vars_map[var_name].desc for var_name in all_vars_map
            if var_name in self._output_names
        ]

    @switch_to_static_graph
    def _append_backward_desc(self):
        program_desc_copy = core.ProgramDesc(self.program.desc)

        # Step 1. prepare program and related var
        # NOTE: To reuse backward interfaces, build Program firstly.
        # Originally, there is no need to build a program, but need to almost
        # rewrite a series of methods for append_backward for program_desc.
        # Therefore, in order to reuse the method of backward.py, build the program here.
        program = self._build_program_by_desc(program_desc_copy)

        # TODO: could the targets be in sub block?
        targets = []
        for out in self._output_descs:
            targets.append(program.global_block().var(out.name()))

        # Step 2. append backward
        backward.gradients(targets=targets, inputs=[])
        return program.desc

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

    def forward(self, inputs):
        # Step 1. prepare inputs, outputs, attrs
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        input_vars = []
        for i, value in enumerate(inputs):
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

        # params = []
        # for param in self._parameters.values():
        #     params.append(param)

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
                    'Params': self._params},
            outputs={'Out': output_vars,
                     'OutScope': tmp_scope_vec},
            attrs={
                'global_block': self._program_desc.block(0),
                'start_op_index': 0,
                'end_op_index': self.program.desc.block(0).op_size(),
                'is_test': self._is_test
            })

        # NOTE: [ why need set param's gradient type here ]
        # if user set sparse gradient mode, the param's gradient
        # will be SelectedRows, not LoDTensor. But tracer will just
        # set param grad VarBase by forward VarBase(LoDTensor)
        # If we don't change grad_var type here, RunProgramOp need
        # transform SelectedRows to LoDTensor forcely, it may not
        # be user wanted result.
        for param in self._params:
            grad_name = param.name + core.grad_var_suffix()
            grad_var = self._program_desc.block(0).find_var(
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

    @property
    def program(self):
        return self._traced_layer._program
