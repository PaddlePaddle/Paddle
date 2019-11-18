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

__all__ = ['TracedLayer']

from .base import program_desc_tracing_guard, switch_to_static_graph
from .layers import Layer
from paddle.fluid import core
from paddle.fluid.framework import Program, Block, Variable, _dygraph_tracer, dygraph_only, _dygraph_guard, _current_expected_place, in_dygraph_mode
from paddle.fluid.executor import Executor, scope_guard
from paddle.fluid.compiler import CompiledProgram
import paddle.fluid.io as fluid_io


def create_program_from_desc(program_desc):
    program = Program()
    program.desc = program_desc
    program.blocks = [Block(program, 0)]
    program._sync_with_cpp()
    return program


def _extract_vars(inputs, result_list):
    if isinstance(inputs, Variable):
        result_list.append(inputs._ivar)

    if isinstance(inputs, (list, tuple)):
        for var in inputs:
            _extract_vars(var, result_list)


def extract_vars(inputs):
    result_list = []
    _extract_vars(inputs, result_list)
    return result_list


@dygraph_only
def _trace(layer, inputs, feed_names=None, fetch_names=None):
    """
    Trace dygraph network into a :code:`Program`. The returned :code:`Program`
    can be run in static graph mode. This method would simply record all
    operators in the network with :code:`inputs` . Users should guarantee that
    the traced dygraph network is independent with input data, input shapes,
    and would not be changed between different batches. Otherwise, the traced
    result may be different.

    Args:
        layer(Layer): the layer to be traced.
        inputs(list): the input arguments of :code:`layer.forward()` method.
        feed_names(list(str), optional): the input variable names in the 
                traced :code:`Program` corresponding to :code:`inputs` . If it
                is None, the variable name of :code:`inputs` would be used. 
                It is suggested that users should set :code:`feed_names` 
                manually. Otherwise, the input variable names would be 
                different between different batches. Default None.
        fetch_names(list(str), optional): the output variable names in the 
                traced :code:`Program` corresponding to the output variables
                of :code:`layer.forward()` method. If it is None, the variable
                name of the outputs of :code:`layer.forward()` would be used.
                It is suggested that users should set :code:`fetch_names`
                manually. Otherwise, the output variable names would be
                different between different batches. Default None.
                
    Returns:
        A tuple of 4 items, whose first item is the outputs of 
        :code:`layer.forward()` method, and second item is the traced 
        :code:`Program`, and the third item is names of feed variables,
        and the fourth item is names of fetch variables.

    Examples:

        .. code-block:: python:

            import paddle.fluid as fluid
            from paddle.fluid.dygraph import FC, to_variable
            import paddle.fluid.dygraph.jit as jit
            import numpy as np

            class ExampleLayer(fluid.dygraph.Layer):
                def __init__(self, name_scope):
                    super(ExampleLayer, self).__init__(name_scope)
                    self._fc = FC(self.full_name(), 10)

                def forward(self, input):
                    return self._fc(input)

            with fluid.dygraph.guard():
                layer = ExampleLayer("example_layer")
                in_np = np.random.random([2, 3]).astype('float32')
                in_var = to_variable(in_np)
                out, program, _, _ = jit._trace(layer, inputs=[in_var],
                                         feed_names=['input'],
                                         fetch_names=['fc_out'])

    """
    assert isinstance(layer, Layer)

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    if feed_names is None:
        feed_names = []

    if fetch_names is None:
        fetch_names = []

    tracer = _dygraph_tracer()._get_program_desc_tracer()

    var_list = extract_vars(inputs)
    if callable(feed_names):
        feed_names = feed_names(len(var_list))

    tracer.set_feed_vars(var_list, feed_names)

    with program_desc_tracing_guard(True):
        original_outputs = layer(*inputs)
        if not isinstance(original_outputs, (list, tuple)):
            outputs = [original_outputs]
        else:
            outputs = original_outputs
        out_vars = [var._ivar for var in outputs]

        if callable(fetch_names):
            fetch_names = fetch_names(len(out_vars))

        tracer.set_fetch_vars(out_vars, fetch_names)
        tracer.set_name_prefix('t_')

        program_desc = tracer.create_program_desc()
        tracer.reset()

    with _dygraph_guard(None):
        program = create_program_from_desc(program_desc)

    return original_outputs, program, feed_names, fetch_names


class TracedLayer(object):
    """
    TracedLayer is a callable object which is converted from dygraph model. 
    Inside TracedLayer, the dygraph model is converted into a static graph
    model, and it would run the static graph model using 
    :code:`Executor` and :code:`CompiledProgram` . The static graph model 
    would share parameters with the dygraph model. 
    
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

        self._place = _current_expected_place()

        self._scope = core.Scope()
        for p in parameters:
            src_tensor = p._ivar.value().get_tensor()
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
            layer (paddle.fluid.dygraph.Layer): the layer object to be traced.
            inputs (list(Variable)): the input variables of the layer object. 

        Returns:
            A tuple of 2 items, whose the first item is the output of 
            :code:`layer(*inputs)` , and the second item is the created
            TracedLayer object. 
            
        Examples:

            .. code-block:: python:

                import paddle.fluid as fluid
                from paddle.fluid.dygraph import FC, to_variable, TracedLayer
                import paddle.fluid.dygraph.jit as jit
                import numpy as np

                class ExampleLayer(fluid.dygraph.Layer):
                    def __init__(self, name_scope):
                        super(ExampleLayer, self).__init__(name_scope)
                        self._fc = FC(self.full_name(), 10)

                    def forward(self, input):
                        return self._fc(input)

                with fluid.dygraph.guard():
                    layer = ExampleLayer("example_layer")
                    in_np = np.random.random([2, 3]).astype('float32')
                    in_var = to_variable(in_np)
                    out_dygraph, static_layer = TracedLayer.trace(layer, inputs=[in_var])
                    out_static_graph = static_layer([in_var]) 
        """
        feed_func = lambda n: ['feed_{}'.format(i) for i in range(n)]
        fetch_func = lambda n: ['fetch_{}'.format(i) for i in range(n)]
        outs, prog, feed, fetch = _trace(layer, inputs, feed_func, fetch_func)
        traced = TracedLayer(prog, layer.parameters(), feed, fetch)
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
                from paddle.fluid.dygraph import FC, to_variable, TracedLayer
                import paddle.fluid.dygraph.jit as jit
                import numpy as np

                class ExampleLayer(fluid.dygraph.Layer):
                    def __init__(self, name_scope):
                        super(ExampleLayer, self).__init__(name_scope)
                        self._fc = FC(self.full_name(), 10) 

                    def forward(self, input):
                        return self._fc(input)

                with fluid.dygraph.guard():
                    layer = ExampleLayer("example_layer")
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
                feed_dict[name] = x._ivar.value().get_tensor()
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
        Save the TracedLayer to an model for inference. The saved
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
            The fetch variables' name list
        
        Return Type: 
            list(str)

        Examples:

            .. code-block:: python:

                import paddle.fluid as fluid
                from paddle.fluid.dygraph import FC, to_variable, TracedLayer
                import paddle.fluid.dygraph.jit as jit
                import numpy as np

                class ExampleLayer(fluid.dygraph.Layer):
                    def __init__(self, name_scope):
                        super(ExampleLayer, self).__init__(name_scope)
                        self._fc = FC(self.full_name(), 10) 

                    def forward(self, input):
                        return self._fc(input)

                with fluid.dygraph.guard():
                    layer = ExampleLayer("example_layer")
                    in_np = np.random.random([2, 3]).astype('float32')
                    in_var = to_variable(in_np)
                    out_dygraph, static_layer = TracedLayer.trace(layer, inputs=[in_var])
                    static_layer.save_inference_model('./saved_infer_model')
        """

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

            return fluid_io.save_inference_model(
                dirname=dirname,
                feeded_var_names=feeded_var_names,
                target_vars=target_vars,
                executor=self._exe,
                main_program=self._program.clone())
