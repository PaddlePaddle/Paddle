# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import annotations

import os

import numpy as np

import paddle
from paddle.base import core, dygraph
from paddle.base.framework import (
    Variable,
)
from paddle.jit.api import (
    _get_function_names_from_layer,
    get_ast_static_function,
    to_static,
)
from paddle.jit.dy2static.program_translator import (
    StaticFunction,
)
from paddle.nn import Layer
from paddle.tensorrt.converter import PaddleToTensorRTConverter
from paddle.tensorrt.util import (
    forbid_op_lower_trt,
    run_pir_pass,
    warmup_shape_infer,
)


class Input:
    """
    A class used to configure and generate random input data for different shapes and data types.

    This class supports generating random input data for minimum, optimal, and maximum shapes, with configurable data types (e.g., 'int' or 'float') and value ranges.

    Args:
        min_input_shape : tuple
            The shape of the minimum input tensor.
        max_input_shape : tuple
            The shape of the maximum input tensor.
        optim_input_shape : tuple, optional
            The shape of the optimal input tensor (default is None).
        input_data_type : str, optional
            The data type for the input tensors, such as 'float32' or 'int64' or 'float32' or 'int32'  (default is float32).
        input_range : tuple, optional
            The range of values used to generate input data. For floats, the default range is (0.0, 1.0). For integers, the default range is (1, 10).
    Returns:
        None

    Examples:
        .. code-block:: python
        >>> # example :
        >>> from paddle.tensorrt.export import Input
        >>> input = Input(
        ...    min_input_shape=(1,100),
        ...    optim_input_shape=(4,100),
        ...    max_input_shape=(8,100),
        ... )

        >>> input.input_data_type='int64'
        >>> input.input_range=(1,10)
    """

    def __init__(
        self,
        min_input_shape: tuple,
        max_input_shape: tuple,
        optim_input_shape: tuple | None = None,
        input_data_type: str | None = 'float32',
        input_range: tuple | None = None,
    ) -> None:
        self.min_input_shape = min_input_shape
        self.max_input_shape = max_input_shape
        self.optim_input_shape = optim_input_shape
        self.input_data_type = input_data_type
        self.input_range = input_range

    def generate_input_data(self):
        """
        Generates random input data based on the specified shapes and data types and input_range

        Returns:
            tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray): A tuple containing the generated input data for the minimum, optimal, and maximum shapes.

        Examples:
            .. code-block:: python
            >>> # example :
            >>> from paddle.tensorrt.export import Input
            >>> input = Input(
            ...    min_input_shape=(1,100),
            ...    optim_input_shape=(4,100),
            ...    max_input_shape=(8,100),
            ... )
            >>> input.input_data_type='int64'
            >>> input.input_range=(1,10)
            >>> input_min_data, input_optim_data, input_max_data = input_config.generate_input_data()
        """
        if self.input_data_type is None:
            self.input_data_type = 'float32'

        if self.input_range is None:
            self.input_range = (
                (0.0, 1.0) if 'float' in self.input_data_type else (1, 10)
            )

        if 'int' in self.input_data_type:
            low, high = self.input_range
            self.input_min_data = np.random.randint(
                low, high, size=self.min_input_shape
            )
            self.input_optim_data = np.random.randint(
                low, high, size=self.optim_input_shape
            )
            self.input_max_data = np.random.randint(
                low, high, size=self.max_input_shape
            )
        else:
            low, high = self.input_range if self.input_range else (0, 1)
            self.input_min_data = np.random.uniform(
                low, high, size=self.min_input_shape
            ).astype(self.input_data_type)
            self.input_optim_data = np.random.uniform(
                low, high, size=self.optim_input_shape
            ).astype(self.input_data_type)
            self.input_max_data = np.random.uniform(
                low, high, size=self.max_input_shape
            ).astype(self.input_data_type)

        return self.input_min_data, self.input_optim_data, self.input_max_data


class TensorRTConfig:
    def __init__(
        self,
        inputs: list,
        min_subgraph_size: int | None = 3,
        save_model_dir: str | None = None,
        disable_ops: str | None = None,
    ) -> None:
        """
        A class for configuring TensorRT optimizations.

        Args:
            inputs : list
                A list of Input configurations
            min_subgraph_size : int, optional
                The minimum number of operations in a subgraph for TensorRT to optimize (default is 3).
            save_model_dir : str, optional
                The directory where the optimized model will be saved (default is None).
            disable_ops : str, optional
                A string representing the names of operations that should not be entering by TensorRT (default is None).

        Returns:
            None

        Examples:
            .. code-block:: python
            >>> # example :
            >>> from paddle.tensorrt.export import (
            ...    Input,
            ...    TensorRTConfig,
            ... )
            >>> input = Input(
            ...    min_input_shape=(1,100),
            ...    optim_input_shape=(4,100),
            ...    max_input_shape=(8,100),
            ... )
            >>> input.input_data_type='int64'
            >>> input.input_range=(1,10)

            >>> trt_config = TensorRTConfig(inputs=[input])
            >>> trt_config.disable_ops = "pd_op.dropout"
        """
        self.inputs = (inputs,)
        self.min_subgraph_size = min_subgraph_size
        self.save_model_dir = save_model_dir
        self.disable_ops = disable_ops
        paddle.framework.set_flags(
            {'FLAGS_trt_min_group_size': min_subgraph_size}
        )


# return an optimized program with pd_op.tensorrt_engine operations.
def convert_to_trt(program, trt_config, scope):
    if not isinstance(program, paddle.base.libpaddle.pir.Program):
        raise TypeError(
            f"program type must be paddle.base.libpaddle.pir.Program, but received {type(program)}"
        )

    output_var = []
    feed_name = []

    for op in program.global_block().ops:
        if op.name() == "pd_op.fetch":
            for operand in op.operands():
                source = operand.source()
                output_var.append(source)
        if op.name() == "pd_op.data" or op.name() == "pd_op.feed":
            param_name = op.attrs()["name"]
            feed_name.append(param_name)

    with paddle.pir_utils.IrGuard():
        for i, input_instance in enumerate(trt_config.inputs):
            # get fake inputs
            min_data, _, max_data = input_instance[i].generate_input_data()
            program_with_output = program.list_vars()[-1]

            # run warmup for collecting shape
            program = warmup_shape_infer(
                program,
                min_shape_feed={feed_name[0]: min_data},
                max_shape_feed={feed_name[0]: max_data},
                scope=scope,
            )

        # run pir pass (including trt_op_marker_pass)
        program_with_pir = run_pir_pass(program, partition_mode=False)

        # specify certain operators to be excluded from entering TensorRT
        if trt_config.disable_ops:
            forbid_op_lower_trt(program, trt_config.disable_ops)

        # run pir pass (including trt_sub_graph_extract_pass)
        program_with_pir = run_pir_pass(program, partition_mode=True)
        trt_output_var = []

        for op in program_with_pir.global_block().ops:
            if op.name() == "pd_op.fetch":
                for operand in op.operands():
                    source = operand.source()
                    trt_output_var.append(source)

        # Step4: run TRTConverter (would lower group_op into tensorrt_engine_op)
        converter = PaddleToTensorRTConverter(program_with_pir, scope)
        converter.convert_program_to_trt()

        # Save PIR program as JSON
        if trt_config.save_model_dir:
            input_values = []
            input_values.extend(
                result
                for op in program_with_pir.global_block().ops
                if op.name() == "pd_op.data" or op.name() == "pd_op.feed"
                for result in op.results()
            )
            place = paddle.CUDAPlace(0)
            exe = paddle.static.Executor(place)

            paddle.static.save_inference_model(
                trt_config.save_model_dir,
                input_values,
                trt_output_var,
                exe,
                program=program_with_pir,
            )
        return program_with_pir


# Obtain a program with tensorrt_op for dynamic-to-static scenarios.
def convert(function=None, input_spec=None, config=None, **kwargs):
    """
    Convert a dynamic graph API to a static graph and apply TensorRT optimizations if relevant parameters are configured.

    Args:
        function (callable): Callable dynamic graph function. If it used as a
            decorator, the decorated function will be parsed as this parameter.
        input_spec (list[InputSpec]|tuple[InputSpec]): list/tuple of InputSpec to
            specific the shape/dtype/name information of each input Tensor.
        config: TensorRTConfig configurations
        kwargs: Support keys including `property`, set `property` to True if the function
            is python property.

    Returns:
        tuple: A tuple containing two elements. The first element is the TensorRT optimized program., optionally optimized with TensorRT if configured. The second element is the scope containing the parameters.

    Examples:
        .. code-block:: python
        >>> # example
        >>> from paddle import nn
        >>> from paddle.static import InputSpec
        >>> import paddle
        >>> from paddle.tensorrt.export import (
        ...    Input,
        ...    TensorRTConfig,
        ...    convert,
        ... )
        >>> import paddle.nn.functional as F

        >>> class CumsumModel(nn.Layer):
        ...    def __init__(self, input_dim):
        ...        super().__init__()
        ...        self.linear = nn.Linear(input_dim, input_dim)

        >>>    def forward(self, x):
        ...        linear_out = self.linear(x)
        ...        relu_out = F.relu(linear_out)
        ...        axis = paddle.full([1], 2, dtype='int64')
        ...        out = paddle.cumsum(relu_out, axis=axis)
        ...        return out

        >>> def test_run():
        ...     with paddle.pir_utils.IrGuard():
        ...         input_config = Input(
        ...             min_input_shape=(9, 10, 11),
        ...             optim_input_shape=(9, 10, 11),
        ...             max_input_shape=(9, 10, 11),
        ...         )
        ...         trt_config = TensorRTConfig(inputs=[input_config])
        ...         for i, input_instrance in enumerate(trt_config.inputs):
        ...             min_data, _, max_data = input_instrance[i].generate_input_data()
        ...             paddle.disable_static()
        ...             x = paddle.to_tensor(min_data)
        ...             net = CumsumModel(input_dim=min_data.shape[-1])
        ...             out=net(x)
        ...            input_spec = [InputSpec(shape=min_data.shape, dtype='float32')]
        ...             program_with_trt ,scope= convert(
        ...                 net,
        ...                 input_spec=input_spec,
        ...                 config=trt_config,
        ...                 full_graph=True,
        ...             )
        ...             output_var = program_with_trt.list_vars()[-1]
        ...             with paddle.pir_utils.IrGuard():
        ...                with paddle.static.scope_guard(scope):
        ...                  place=paddle.CUDAPlace(0)
        ...                  executor=paddle.static.Executor(place)
        ...                  output=executor.run(program_with_trt, feed={"x": min_data}, fetch_list=[output_var],scope=scope)

        >>> test_run()

    """
    # Converts dynamic graph APIs into static graph
    static_net = paddle.jit.to_static(
        function,
        input_spec=input_spec,
        **kwargs,
    )
    is_prim_infer = core._is_fwd_prim_enabled() and core._is_bwd_prim_enabled()
    # If the input layer be wrapped by DataParallel,
    # the args and kwargs of forward method will can't be parsed by
    # function_spec, so here we save DataParallel._layers instead
    # DataParallel it self
    #  using inner_layer, do not change input layer
    if isinstance(static_net, paddle.DataParallel):
        inner_layer = static_net._layers
    else:
        inner_layer = static_net

    # avoid change user given input_spec
    inner_input_spec = None
    if input_spec is not None:
        if isinstance(static_net, Layer):
            for member_name in _get_function_names_from_layer(inner_layer):
                static_func = getattr(inner_layer, member_name, None)
                if (
                    isinstance(static_func, StaticFunction)
                    and 'forward' != member_name
                ):
                    raise ValueError(
                        f"If there are static functions other than 'forward' that need to be saved, the input 'input_spec' should be None, but received the type of 'input_spec' is {type(input_spec)}."
                    )
        if not isinstance(input_spec, (list, tuple)):
            raise TypeError(
                f"The input input_spec should be 'list', but received input_spec's type is {type(input_spec)}."
            )
        inner_input_spec = []
        for var in paddle.utils.flatten(input_spec):
            if isinstance(var, paddle.static.InputSpec):
                inner_input_spec.append(var)
            elif isinstance(
                var, (core.eager.Tensor, Variable, paddle.pir.Value)
            ):
                inner_input_spec.append(
                    paddle.static.InputSpec.from_tensor(var)
                )
            else:
                #  Support non-Tensor type in `input_spec`
                inner_input_spec.append(var)

    # whether outermost layer has pre/post hook, if does, we need also save
    # these operators in program.
    with_hook = False
    scope = core.Scope()
    extra_var_info = {}
    if isinstance(static_net, Layer):
        functions = list(set(_get_function_names_from_layer(static_net)))
        functions = sorted(functions)
        if static_net._forward_pre_hooks or static_net._forward_post_hooks:
            with_hook = True
    else:
        # layer is function
        functions = [static_net]

    property_vals = []  # (value, key)
    concrete_program = None
    for attr_func in functions:
        if isinstance(static_net, Layer):
            static_func = get_ast_static_function(
                getattr(inner_layer, attr_func, None)
            )
            if isinstance(static_func, StaticFunction):
                if static_func.is_property:
                    # property method to be exported
                    immediate_val = static_func()
                    property_vals.append(
                        (
                            immediate_val,
                            static_net.__class__.__name__ + '.' + attr_func,
                        )
                    )
                    continue
                concrete_program = (
                    static_func.concrete_program_specify_input_spec(
                        inner_input_spec,
                        with_hook=with_hook,
                        is_prim_infer=is_prim_infer,
                    )
                )
            elif 'forward' == attr_func:
                # if input_spec is incomplete, declarative will throw error
                # inner_input_spec is list[InputSpec], it should be packed with same structure
                # as original input_spec here
                if inner_input_spec:
                    inner_input_spec = paddle.utils.pack_sequence_as(
                        input_spec, inner_input_spec
                    )
                static_forward = to_static(
                    inner_layer.forward,
                    input_spec=inner_input_spec,
                    full_graph=True,
                )
                concrete_program = (
                    static_forward.concrete_program_specify_input_spec(
                        with_hook=with_hook, is_prim_infer=is_prim_infer
                    )
                )
                inner_input_spec = None
            else:
                continue
        else:
            # When layer is a function
            if isinstance(attr_func, StaticFunction):
                static_func = get_ast_static_function(attr_func)
                if static_func.is_property:
                    immediate_val = static_func()
                    property_vals.append((immediate_val, static_func))
                    continue

                concrete_program = (
                    static_func.concrete_program_specify_input_spec(
                        inner_input_spec, is_prim_infer=is_prim_infer
                    )
                )
            else:
                static_func = get_ast_static_function(attr_func)
                if inner_input_spec:
                    inner_input_spec = paddle.utils.pack_sequence_as(
                        input_spec, inner_input_spec
                    )
                static_function = to_static(
                    static_func,
                    input_spec=inner_input_spec,
                    full_graph=True,
                )
                concrete_program = static_function.concrete_program

        # when save multi `StaticFunction`, all `StaticFunction` share params.
        dygraph_state_dict = None
        if isinstance(inner_layer, Layer):
            dygraph_state_dict = inner_layer.to_static_state_dict()
        elif isinstance(attr_func, StaticFunction):
            if static_func._class_instance:
                dygraph_state_dict = (
                    static_func._class_instance.to_static_state_dict()
                )
        if dygraph_state_dict:
            #  we maintain the mapping of variable name to
            # structured name, the buffer variable (non-persistable)
            # saved to inference program may not need by dygraph Layer,
            # we only record the state_dict variable's structured name
            state_names_dict = {}
            state_var_dict = {}
            for strcutured_name, var in dygraph_state_dict.items():
                state_names_dict[var.name] = strcutured_name
                state_var_dict[var.name] = var
        #  share parameters from Layer to scope & record var info
        with dygraph.guard():
            for tensor, value in zip(*concrete_program.parameters):
                if not value.persistable:
                    continue
                param_or_buffer_tensor = scope.var(value.name).get_tensor()

                src_tensor = state_var_dict[tensor.name].value().get_tensor()
                param_or_buffer_tensor._share_data_with(src_tensor)
    with paddle.pir_utils.IrGuard():
        main_program = concrete_program.main_program
        program_with_trt = convert_to_trt(main_program, config, scope)
        return program_with_trt, scope


# Obtain a program with tensorrt_op by directly loading the model.
def convert_loaded_model(model_dir, config):
    """
    Loading a PaddlePaddle Model and Exporting the TensorRT-Optimized Program.

    Args:
       model_dir(str):The directory path where the PaddlePaddle model is located.
       config(TensorRTConfig):The configuration of TensorRTConfig

    Returns:
        program:The TensorRT optimized program.

    Examples:
        .. code-block:: python
            >>> import paddle
            >>> import numpy as np
            >>> import tempfile
            >>> import paddle.inference as paddle_infer
            >>> from paddle.tensorrt.export import (
            ...      Input,
            ...      TensorRTConfig,
            ...      export,
            ...      convert_loaded_model,
            ... )
            >>> import os
            >>> from paddle import nn
            >>> import paddle.nn.functional as F

            >>> class CumsumModel(nn.Layer):
            ...    def __init__(self, input_dim):
            ...        super().__init__()
            ...        self.linear = nn.Linear(input_dim, input_dim)

            ...    def forward(self, x):
            ...        linear_out = self.linear(x)
            ...        relu_out = F.relu(linear_out)
            ...        axis = paddle.full([1], 2, dtype='int64')
            ...        out = paddle.cumsum(relu_out, axis=axis)
            ...        return out

            >>> temp_dir = tempfile.TemporaryDirectory()
            >>> save_path = os.path.join(temp_dir.name, 'tensor_axis_cumsum')

            >>> with paddle.pir_utils.IrGuard():
            ...    paddle.enable_static()
            ...    np_x = np.random.randn(9, 10, 11).astype('float32')
            ...    main_prog = paddle.static.Program()
            ...    startup_prog = paddle.static.Program()
            ...    with paddle.static.program_guard(main_prog, startup_prog):
            ...        x = paddle.static.data(
            ...            shape=np_x.shape, name='x', dtype=np_x.dtype
            ...        )
            ...        model = CumsumModel(input_dim=np_x.shape[-1])
            ...        out = model(x)
            ...        loss = paddle.mean(out)
            ...        sgd = paddle.optimizer.SGD(learning_rate=0.0)
            ...        sgd.minimize(paddle.mean(out))

            ...        exe = paddle.static.Executor(paddle.CUDAPlace(0))
            ...        exe.run(startup_prog)
            ...        static_out = exe.run(feed={'x': np_x}, fetch_list=[out])

            ...        # run infer
            ...        paddle.static.save_inference_model(
            ...            save_path, [x], [out], exe
            ...        )

            ...        config = paddle_infer.Config(
            ...            save_path + '.json', save_path + '.pdiparams'
            ...        )
            ...        config.enable_new_ir()
            ...        config.enable_new_executor()
            ...        config.use_optimized_model(True)

            ... # Set input
            ...    input_config = Input(
            ...        min_input_shape=(9, 10, 11),
            ...        optim_input_shape=(9, 10, 11),
            ...        max_input_shape=(9, 10, 11),
            ...    )
            ...    # Create a TensorRTConfig with inputs as a required field.
            ...    trt_config = TensorRTConfig(inputs=[input_config])

            ...    trt_save_path = os.path.join(temp_dir.name, 'trt')
            ...    trt_config.save_model_dir = trt_save_path

            ...    program_with_trt = convert_loaded_model(save_path, trt_config)

            ...    # Create a config for inference.
            ...    config = paddle_infer.Config(
            ...        trt_config.save_model_dir + '.json',
            ...        trt_config.save_model_dir + '.pdiparams',
            ...    )

            ...    if paddle.is_compiled_with_cuda():
            ...        config.enable_use_gpu(100, 0)
            ...    else:
            ...        config.disable_gpu()
            ...    predictor = paddle_infer.create_predictor(config)
            ...    input_names = predictor.get_input_names()

            ...    paddle.disable_static()
            ...    for i, input_instrance in enumerate(trt_config.inputs):
            ...        min_data, _, max_data = input_instrance[i].generate_input_data()
            ...        model_inputs = paddle.to_tensor(min_data)
            ...        output_converted = predictor.run([model_inputs])

    """
    if os.path.abspath(config.save_model_dir) == os.path.abspath(model_dir):
        raise ValueError(
            "The `config.save_model_dir` and `model_dir` cannot be the same. Please specify a different directory for saving the model."
        )

    scope = paddle.static.global_scope()
    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)

    model_filename = model_dir + '.json'
    if os.path.exists(model_dir + '.json'):
        model_filename = model_dir + '.json'
    elif os.path.exists(model_dir + '.pdmodel'):
        model_filename = model_dir + '.pdmodel'
    else:
        raise ValueError(
            f"No valid model file found in the directory '{model_dir}'. Expected either 'json' or 'pdmodel'. Please ensure that the directory contains one of these files."
        )

    params_filename = model_dir + '.pdiparams'

    with paddle.pir_utils.IrGuard():
        [program, feed_target_names, fetch_targets] = (
            paddle.static.io.load_inference_model(
                model_dir,
                executor=exe,
                model_filename=model_filename,
                params_filename=params_filename,
            )
        )

    return convert_to_trt(program, config, scope)
