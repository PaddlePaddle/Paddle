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

from __future__ import annotations

import os

import numpy as np

import paddle
from paddle.base import core, framework, unique_name
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.framework import in_dynamic_mode
from paddle.nn.layer import layers
from paddle.pir.core import datatype_to_vartype

__all__ = []

PIR_INFER_MODEL_SUFFIX = ".json"

from .translated_layer import (
    BUFFER_NAME_PREFIX,
    INFER_PARAMS_SUFFIX,
    PARAMETER_NAME_PREFIX,
)


def _load_pir_program(model_file_path):
    program = paddle.static.Program()
    trainable = paddle.base.core.deserialize_pir_program(
        model_file_path, program, 1
    )

    return program, trainable


@switch_to_static_graph
def _generate_unique_var_name(prefix):
    return unique_name.generate_with_ignorable_key(prefix)


@switch_to_static_graph
def _generate_unique_var_name(prefix):
    return unique_name.generate(prefix)


from paddle.static.pir_io import get_pir_parameters


def _get_pir_parameters_var_names(program):
    persistable_vars = []
    persistable_names = []
    rename_new_old_dict = {}
    param, opt = get_pir_parameters(program)
    vars = param + opt
    for var in vars:
        persistable_vars.append(var)
        origin_name = var.name
        var.name = _generate_unique_var_name(var.name)
        rename_new_old_dict[var.name] = origin_name
        persistable_names.append(var.name)

    return (
        persistable_vars,
        persistable_names,
        rename_new_old_dict,
    )


class _PirProgramHolder:
    def __init__(self, program, trainable):
        super().__init__()

        # input, output, persistable,
        self._input_vars = []
        self._output_vars = []
        self._parameter_vars = []
        self._parameter_names = []

        self.support_train = trainable
        # append suffix var name dict
        self._suffix_varname_dict = None
        self._infer_program = program

        self._preprocess()

    def _preprocess(self):
        (
            self._parameter_vars,
            self._parameter_names,
            self._suffix_varname_dict,
        ) = _get_pir_parameters_var_names(self._infer_program)
        block = self._infer_program.global_block()
        for op in block.ops:
            if op.name() == 'pd_op.data':
                self._input_vars.append(op.result(0))
            elif op.name() == 'pd_op.feed':
                var_name = op.attr()["name"]
                org_value = op.result(0)
                with block:
                    value = paddle._pir_ops.data(
                        name=var_name,
                        shape=org_value.shape,
                        dtype=org_value.dtype,
                        place=paddle.base.core.Place(),
                    )
                    org_value.replace_all_uses_with(value)
                    value.get_defining_op().move_before(op)
                block.remove_op(op)
            if op.name() == 'pd_op.fetch':
                self._output_vars.append(op.operand_source(0))
                with block:
                    paddle._pir_ops.set_persistable_value(
                        op.operand_source(0),
                        "output_" + str(len(self._output_vars) - 1),
                    )
                block.remove_op(op)

    @property
    def infer_program(self):
        return self._infer_program

    @property
    def input_vars(self):
        return self._input_vars

    @property
    def output_vars(self):
        return self._output_vars

    @property
    def persistable_names(self):
        return self._parameter_names

    @property
    def persistable_vars(self):
        return self._parameter_vars


# [ PirTranslatedLayer : Run program in dygraph mode ]
#
# DESIGN IDEA: using an special operator `PirRunProgram`, execute program inside operator.
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
#   The variable/parameter of the dynamic graph is not in the scope, so before the op
#   executes the program internally, create persistent variables with the
#   same name as feed, parameters, and fetch in the scope, and share the
#   DenseTensor of the op input.
#
# 2. Forward and Backward Separation:
#   Because the dynamic graph op performs the forward and backward separately,
#   in the forward op RunProgram, we only execute the forward part of whole program,
#   and in the backward op RunProgramGrad, we execute the backward part of program.
#   We can not separate the program into forward and backward part, which will
#   make some control flow execution logic wrong.


# NOTE: [compatible] deal with model saved by save_inference_model,
# which need get var info from program desc


def _load_pir_parameter_vars(model_path, program_holder, params_filename):
    # construct var dict
    load_var_dict = {}
    load_var_list = []
    other_var_dict = {}
    load_densetensor_list = []
    persistable_var = program_holder.persistable_vars
    persistable_var_name = program_holder.persistable_names
    origin_persistable_var_name = [
        program_holder._suffix_varname_dict[var_name]
        for var_name in persistable_var_name
    ]
    for name, var in sorted(zip(origin_persistable_var_name, persistable_var)):
        if var.persistable:
            # use default shape and dtype
            new_var = framework.EagerParamBase(
                shape=var.shape,  # only to pass check, this shape is not meaningful
                dtype=core.VarDesc.VarType.FP32,
                name=var.name,
                persistable=True,
            )

            new_var.stop_gradient = var.stop_gradient
            load_var_dict[name] = new_var
            load_var_list.append(new_var)
            load_densetensor_list.append(new_var.get_tensor())

        else:
            new_var = core.eager.Tensor(
                dtype=datatype_to_vartype[var.dtype],
                dims=var.shape,
                name=var.name,
                type=core.VarDesc.VarType.DENSE_TENSOR,
                place=framework._current_expected_place(),
                persistable=False,
            )
            other_var_dict[name] = new_var

    # load all vars
    assert params_filename is not None, "params_filename should not be None."
    var_file_path = os.path.join(model_path, params_filename)
    if os.path.exists(var_file_path):
        core.load_combine_func(
            var_file_path,
            list(load_var_dict.keys()),
            load_densetensor_list,
            False,
            framework._current_expected_place(),
        )
    else:
        raise ValueError(
            f"The file {var_file_path} does not exist. Please check the model path."
        )

    load_var_dict.update(other_var_dict)
    return load_var_dict


def _construct_program_holders(model_path, model_filename=None):
    # make sure the path has been checked
    program_holder_dict = {}

    if model_filename is not None:
        # [compatible] if assign model_filename, only can load one program as Layer.forward
        model_filename = os.path.basename(model_filename)
        model_file_path = os.path.join(model_path, model_filename)
        model_name = model_filename[: -len(PIR_INFER_MODEL_SUFFIX)]
        # Load every file that meets the requirements in the directory model_path.
        for filename in os.listdir(model_path):
            if model_filename == filename:
                func_name = 'forward'
                model_file_path = os.path.join(model_path, model_filename)
            elif filename.endswith(
                PIR_INFER_MODEL_SUFFIX
            ) and filename.startswith(model_name):
                parsing_names = filename[
                    len(model_name) : -len(PIR_INFER_MODEL_SUFFIX) + 1
                ].split('.')
                if len(parsing_names) == 3 and len(parsing_names[1]) > 0:
                    func_name = parsing_names[1]
                    model_file_path = os.path.join(model_path, filename)
                else:
                    continue
            else:
                continue
            program, trainable = _load_pir_program(model_file_path)
            program_holder_dict[func_name] = _PirProgramHolder(
                program, trainable
            )
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
                    program, trainable = _load_pir_program(model_file_path)
                    program_holder_dict[func_name] = _PirProgramHolder(
                        program, trainable
                    )

    return program_holder_dict


def _construct_params_and_buffers(model_path, programs, params_filename=None):
    params_path = os.path.join(model_path, str(params_filename))

    if params_filename is not None and not os.path.exists(params_path):
        # When saving XX, there is only '*.pdmodel'
        return {}
    else:
        var_dict = _load_pir_parameter_vars(
            model_path, programs['forward'], params_filename
        )
        model_name = params_filename[: -len(INFER_PARAMS_SUFFIX)]
        # Load every file that meets the requirements in the directory model_path.
        for file_name in os.listdir(model_path):
            if file_name.startswith(model_name) and file_name.endswith(
                INFER_PARAMS_SUFFIX
            ):
                parsing_names = file_name[
                    len(model_name) : -len(INFER_PARAMS_SUFFIX) + 1
                ].split('.')
                if len(parsing_names) == 3 and len(parsing_names[1]) > 0:
                    func_name = parsing_names[1]
                else:
                    continue
            else:
                continue

            var_dict.update(
                _load_pir_parameter_vars(
                    model_path, programs[func_name], file_name
                )
            )

        return var_dict


def _run_dygraph(instance, input, program_holder):
    # 1. prepare inputs, outputs, attrs
    input_tensors = []
    input_tensor_names = []

    for i, value in enumerate(input):
        if not isinstance(value, (np.ndarray, core.eager.Tensor)):
            raise TypeError(
                f"The type of input in PirTranslatedLayer must be numpy array or Variable(Tensor), but received {type(value)}."
            )
        # NOTE: In order to unify the API, firstly convert the input to Tensor
        if isinstance(value, np.ndarray):
            tensor = core.eager.Tensor(
                value=value,
                name=program_holder.input_vars[i].name,
                persistable=False,
                place=framework._current_expected_place(),
                zero_copy=True,
            )
        else:
            tensor = value
            # NOTE: we changed var name here,
            # but it may be an important name set by user
            tensor.name = program_holder.input_vars[i].name
        input_tensor_names.append(tensor.name)
        input_tensors.append(tensor)

    persistable_tensors = []
    origin_persistable_var_name = [
        program_holder._suffix_varname_dict[var_name]
        for var_name in program_holder.persistable_names
    ]
    for var_name in origin_persistable_var_name:
        dy_var_name = instance._persistable_var_name_dict[var_name]
        if dy_var_name in instance._parameters:
            persistable_tensors.append(instance._parameters[dy_var_name])
        elif dy_var_name in instance._buffers:
            persistable_tensors.append(instance._buffers[dy_var_name])
        else:
            raise ValueError(
                f"The persistable variable {var_name} does not exist in current PirTranslatedLayer."
            )

    from paddle.jit.dy2static.pir_partial_program import PartialProgramLayer

    inputs = program_holder.input_vars
    outputs = program_holder.output_vars
    parameters = (persistable_tensors, program_holder.persistable_vars)

    layer = PartialProgramLayer(
        program_holder.infer_program,
        inputs,
        outputs,
        parameters,
    )
    instance.layer = layer
    if instance._is_test:
        layer.training = False
    else:
        if not program_holder.support_train:
            raise ValueError(
                "The model is not trainable, please check model_file of jit.save."
            )
        else:
            layer.training = True

    return instance.layer(input_tensors)


def _run_static_graph(inputs, program_holder, src_program):
    '''
    This function is used when the pirTranslatedLayer is
    applied for dy_to_static conversion.
    '''
    dst_program = paddle.static.default_main_program()
    value_map = paddle.pir.IrMapping()
    # Establish a mapping relationship between existing parameters
    # and corresponding parameters in the program to be copied
    len_dst_op = len(dst_program.global_block().ops)
    for dst_op in dst_program.global_block().ops:
        if dst_op.name() == "builtin.parameter":
            for src_op in src_program.global_block().ops[:len_dst_op]:
                if (
                    src_op.name() == dst_op.name()
                    and src_op.result(0).name == dst_op.result(0).name
                ):
                    for i in range(src_op.num_results()):
                        value_map.add(src_op.result(i), dst_op.result(i))
    # Establish a mapping relationship between truly inputs
    # and corresponding inputs in the program to be copied
    src_inputs = program_holder.input_vars
    if len(src_inputs) != len(inputs):
        raise ValueError(
            f"The number of input is invalid, expected {len(src_inputs)}, but received {len(inputs)}."
        )
    for src_input, input_ in zip(src_inputs, inputs):
        value_map.add(src_input, input_)

    # find the insert point for copy
    current_insert_point = paddle.pir.get_current_insertion_point()
    current_block = current_insert_point.block()
    src_program.copy_to_block(value_map, current_block)

    output = [value_map.look_up(v) for v in program_holder.output_vars]
    return output[0] if len(output) == 1 else output


def _collect_current_and_parent_var(program, block_idx):
    '''
    Get variables in current block and its parent block.

    Args:
        program(Program): The program containing the current block.
        block_idx(int): index of current block.

    Returns:
        List: list of variables.
    '''
    vars = []
    if block_idx < 0:
        return vars
    for var in program.block(block_idx).vars:
        vars.append(var)
    parent_idx = program.block(block_idx).parent_idx
    if parent_idx > -1:
        vars += _collect_current_and_parent_var(program, parent_idx)
    return vars


class PirTranslatedLayer(layers.Layer):
    """
    PirTranslatedLayer is a ``paddle.nn.Layer`` for holding the model
    loaded by :ref:`api_paddle_jit_load` . It can be used like a
    general Layer object in eval or train mode.

    .. note:
        The PirTranslatedLayer objects should not be created by constructor, it only can be loaded and constructed by :ref:`api_paddle_jit_load` .

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('`paddle.jit.to_static` can not run in xdoctest')
            >>> import numpy as np
            >>> import paddle
            >>> import paddle.nn as nn
            >>> import paddle.optimizer as opt

            >>> BATCH_SIZE = 16
            >>> BATCH_NUM = 4
            >>> EPOCH_NUM = 4

            >>> IMAGE_SIZE = 784
            >>> CLASS_NUM = 10

            >>> # define a random dataset
            >>> class RandomDataset(paddle.io.Dataset): # type: ignore[type-arg]
            ...     def __init__(self, num_samples):
            ...         self.num_samples = num_samples
            ...
            ...     def __getitem__(self, idx):
            ...         image = np.random.random([IMAGE_SIZE]).astype('float32')
            ...         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
            ...         return image, label
            ...
            ...     def __len__(self):
            ...         return self.num_samples
            ...
            >>> class LinearNet(nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
            ...
            ...     @paddle.jit.to_static
            ...     def forward(self, x):
            ...         return self._linear(x)
            ...
            >>> def train(layer, loader, loss_fn, opt):
            ...     for epoch_id in range(EPOCH_NUM):
            ...         for batch_id, (image, label) in enumerate(loader()):
            ...             out = layer(image)
            ...             loss = loss_fn(out, label)
            ...             loss.backward()
            ...             opt.step()
            ...             opt.clear_grad()
            ...             print("Epoch {} batch {}: loss = {}".format(
            ...                 epoch_id, batch_id, np.mean(loss.numpy())))
            ...
            >>> # 1. train & save model.
            >>> # create network
            >>> layer = LinearNet()
            >>> loss_fn = nn.CrossEntropyLoss()
            >>> adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

            >>> # create data loader
            >>> dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            >>> loader = paddle.io.DataLoader(dataset,
            ...     batch_size=BATCH_SIZE,
            ...     shuffle=True,
            ...     drop_last=True,
            ...     num_workers=2
            ... )
            >>> # train
            >>> train(layer, loader, loss_fn, adam)

            >>> # save
            >>> model_path = "linear.example.model"
            >>> paddle.jit.save(layer, model_path)

            >>> # 2. load model as PirTranslatedLayer
            >>> # load
            >>> translated_layer = paddle.jit.load(model_path)

            >>> # inference
            >>> translated_layer.eval()
            >>> x = paddle.randn([1, IMAGE_SIZE], 'float32')
            >>> pred = translated_layer(x)

            >>> # fine-tune
            >>> translated_layer.train()
            >>> adam = opt.Adam(learning_rate=0.001, parameters=translated_layer.parameters())
            >>> train(translated_layer, loader, loss_fn, adam)

    """

    def __init__(
        self,
        programs: dict[str, paddle.static.Program],
        persistable_vars: dict[str, paddle.Tensor],
    ):
        super().__init__()

        if not isinstance(programs, dict):
            raise TypeError(
                "PirTranslatedLayer need to use _ProgramHolder's dict for initialization."
            )
        if not isinstance(persistable_vars, dict):
            raise TypeError(
                "PirTranslatedLayer need to use persistable variable dict for initialization."
            )

        self._program_holder_dict = programs

        # NOTE(chenweihang): [ why not use var name directly? ]
        # When add parameter or buffer to Layer by follow apis,
        # the variable name can't contain `.`, because which may cause
        # AttributeError when access the newly added parameter or buffer
        # in the form of `self.**.**``, but the EagerParamBase or BarBase
        # name contains `.` originally, such as `linear_0.w_0`, so here
        # need to generate new var name for each var
        self._persistable_var_name_dict = {}
        # the PirTranslatedLayer object held var names count started from 0
        with unique_name.guard():
            for name, var in persistable_vars.items():
                if isinstance(var, framework.EagerParamBase):
                    dy_name = _generate_unique_var_name(PARAMETER_NAME_PREFIX)
                    self._persistable_var_name_dict[name] = dy_name
                    self.add_parameter(dy_name, var)
                elif isinstance(var, core.eager.Tensor):
                    dy_name = _generate_unique_var_name(BUFFER_NAME_PREFIX)
                    self._persistable_var_name_dict[name] = dy_name
                    self.register_buffer(dy_name, var)
                else:
                    raise TypeError(
                        "Adding persistent variable which  to layer is not supported now"
                    )

        self._is_test = True
        self._input_args_names = None

    @staticmethod
    @framework.dygraph_only
    def _construct(model_path, configs=None):
        # 0. dir and filename check
        model_path = os.path.normpath(model_path)
        if not os.path.isdir(model_path):
            raise ValueError(f"There is no directory named '{model_path}'")
        model_filename = None
        params_filename = None
        if configs is not None:
            model_filename = configs.model_filename
            params_filename = configs.params_filename

        # 1. load program desc & construct _ProgramHolder
        programs = _construct_program_holders(model_path, model_filename)

        # 2. load layer parameters & buffers
        persistable_vars = _construct_params_and_buffers(
            model_path, programs, params_filename
        )

        # 3. construct PirTranslatedLayer object
        translated_layer = PirTranslatedLayer(programs, persistable_vars)

        # 4. create PirTranslatedLayer's execution method
        for method_name, program_holder in programs.items():
            if translated_layer._input_args_names is None:
                translated_layer._input_args_names = [
                    ins.name for ins in program_holder.input_vars
                ]
            setattr(
                PirTranslatedLayer,
                method_name,
                PirTranslatedLayer._execution_method_creator(
                    method_name, program_holder
                ),
            )

        # 5. set PirTranslatedLayer's default mode to eval
        translated_layer.eval()

        return translated_layer

    @staticmethod
    def _execution_method_creator(method_name, program_holder):
        def __i_m_p_l__(self, *input):
            program_holder = self._program_holder_dict[__i_m_p_l__.__name__]
            # When using jit.save, it runs in static graph mode.
            # Run in dynamic graph mode when the model is inferring.
            if in_dynamic_mode():
                return _run_dygraph(self, input, program_holder)
            else:
                return _run_static_graph(
                    input, program_holder, program_holder.infer_program
                )

        __i_m_p_l__.__name__ = method_name
        return __i_m_p_l__

    def train(self):
        self._is_test = False
        self.training = True

    def eval(self):
        self._is_test = True
        self.training = False

    def program(self, method_name='forward'):
        """
        Gets translated program of specified method.

        Args:
            - method_name (string): method name corresponding to the program
                to be obtained. Default: 'forward'.

        Returns:
            Program

        Examples:
            .. code-block:: python

                >>> # doctest: +SKIP('`paddle.jit.to_static` can not run in xdoctest')
                >>> import numpy as np
                >>> import paddle
                >>> from paddle import nn
                >>> import paddle.optimizer as opt

                >>> BATCH_SIZE = 16
                >>> BATCH_NUM = 4
                >>> EPOCH_NUM = 4

                >>> IMAGE_SIZE = 784
                >>> CLASS_NUM = 10

                >>> # define a random dataset
                >>> class RandomDataset(paddle.io.Dataset): # type: ignore[type-arg]
                ...     def __init__(self, num_samples):
                ...         self.num_samples = num_samples
                ...
                ...     def __getitem__(self, idx):
                ...         image = np.random.random([IMAGE_SIZE]).astype('float32')
                ...         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                ...         return image, label
                ...
                ...     def __len__(self):
                ...         return self.num_samples
                ...
                >>> class LinearNet(nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
                ...
                ...     @paddle.jit.to_static
                ...     def forward(self, x):
                ...         return self._linear(x)
                ...
                >>> def train(layer, loader, loss_fn, opt):
                ...     for epoch_id in range(EPOCH_NUM):
                ...         for batch_id, (image, label) in enumerate(loader()):
                ...             out = layer(image)
                ...             loss = loss_fn(out, label)
                ...             loss.backward()
                ...             opt.step()
                ...             opt.clear_grad()
                ...             print("Epoch {} batch {}: loss = {}".format(
                ...                 epoch_id, batch_id, np.mean(loss.numpy())))
                ...
                >>> # create network
                >>> layer = LinearNet()
                >>> loss_fn = nn.CrossEntropyLoss()
                >>> adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())
                >>> # create data loader
                >>> dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
                >>> loader = paddle.io.DataLoader(dataset,
                ...     batch_size=BATCH_SIZE,
                ...     shuffle=True,
                ...     drop_last=True,
                ...     num_workers=2
                ... )
                >>> # train
                >>> train(layer, loader, loss_fn, adam)

                >>> # save
                >>> model_path = "linear.example.model"
                >>> paddle.jit.save(layer, model_path)

                >>> # load
                >>> translated_layer = paddle.jit.load(model_path)

                >>> # get program
                >>> program = translated_layer.program()
        """
        # 1. get program holder
        program_holder = self._get_program_holder(method_name)

        # 2. get inference program desc
        program = program_holder.infer_program

        return program

    def _get_program_holder(self, method_name='forward'):
        program_holder = self._program_holder_dict.get(method_name, None)
        if program_holder is None:
            raise ValueError(
                f"The method `{method_name}` does not exist in loaded PirTranslatedLayer."
            )
        return program_holder

    def _input_spec(self, method_name='forward'):
        # 1. get program holder
        program_holder = self._get_program_holder(method_name)

        # 2. build input spec by input desc
        input_spec = []
        for var in program_holder.input_vars:
            spec = paddle.static.InputSpec(
                shape=var.shape,
                dtype=var.dtype,
                name=var.name,
            )
            input_spec.append(spec)

        return input_spec

    def _output_spec(self, method_name='forward'):
        # 1. get program holder
        program_holder = self._get_program_holder(method_name)

        # 2. build output spec by output desc
        output_spec = []
        for var in program_holder.output_vars:
            # NOTE(chenweihang): InputSpec describes a tensor, not just input.
            # Maybe the name is not good enough. Here we use InputSpec to
            # construct the description of Output tensor
            spec = paddle.static.InputSpec(
                shape=var.shape,
                dtype=var.dtype,
                name=var.name,
            )
            output_spec.append(spec)

        return output_spec
