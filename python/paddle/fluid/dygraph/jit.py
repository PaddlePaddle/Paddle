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

import os
import pickle
import warnings

import six
from paddle.fluid import core
from paddle.fluid.compiler import BuildStrategy, CompiledProgram, ExecutionStrategy
from paddle.fluid.data_feeder import check_type
from paddle.fluid.dygraph.base import program_desc_tracing_guard, switch_to_static_graph
from paddle.fluid.dygraph.dygraph_to_static.error import ERROR_DATA
from paddle.fluid.dygraph.dygraph_to_static.program_translator import FunctionSpec, ProgramTranslator
from paddle.fluid.dygraph.io import EXTRA_VAR_INFO_FILENAME, VARIABLE_FILENAME, TranslatedLayer
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.executor import Executor, scope_guard
from paddle.fluid.framework import Block, ParamBase, Program, Variable
from paddle.fluid.framework import _current_expected_place, _dygraph_guard, _dygraph_tracer
from paddle.fluid.framework import dygraph_only, in_dygraph_mode
from paddle.fluid.wrapped_decorator import wrap_decorator

__all__ = ['TracedLayer', 'declarative', 'dygraph_to_static_func']


def create_program_from_desc(program_desc):
    program = Program()
    program.desc = program_desc
    program.blocks = [Block(program, 0)]
    program._sync_with_cpp()
    return program


def _extract_vars(inputs, result_list):
    if isinstance(inputs, Variable):
        result_list.append(inputs)
    elif isinstance(inputs, (list, tuple)):
        for var in inputs:
            _extract_vars(var, result_list)
    else:
        raise TypeError(
            "The type of 'each element of inputs' in fluid.dygraph.jit.TracedLayer.trace must be fluid.Variable, but received {}.".
            format(type(inputs)))


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
            warnings.warn(
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
        if not program_translator.enable_declarative:
            warnings.warn(
                "The decorator 'declarative' doesn't work when setting ProgramTranslator.enable=False. "
                "We will just return dygraph output.")
            return dygraph_func(*args, **kwargs)
        try:
            return program_translator.get_output(dygraph_func, *args, **kwargs)
        except Exception as e:
            error_data = getattr(e, ERROR_DATA, None)
            if error_data:
                new_exception = error_data.create_exception()
                if six.PY3:
                    # NOTE(liym27):
                    # 1. Why `raise new_exception from None`?
                    #   In Python 3, by default, an new exception is raised with trace information of the caught exception.
                    #   This only raises new_exception and hides unwanted implementation details from tracebacks of the
                    #   caught exception.
                    # 2. Use exec to bypass syntax error checking in Python 2.

                    six.exec_("raise new_exception from None")
                else:
                    raise new_exception
            else:
                raise

    return __impl__


declarative = wrap_decorator(_declarative_)


class SaveLoadConfig(object):
    """
    The additional configuration options may be used in function 
    :ref:`api_imperative_jit_save` that save :ref:`api_imperative_TranslatedLayer` 
    or used in function :ref:`api_imperative_jit_load` that 
    load :ref:`api_imperative_TranslatedLayer` .
    
    Examples:
        1. Using ``SaveLoadConfig`` when saving model

        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid
            from paddle.fluid.dygraph import Linear
            from paddle.fluid.dygraph import declarative

            class SimpleNet(fluid.dygraph.Layer):
                def __init__(self, in_size, out_size):
                    super(SimpleNet, self).__init__()
                    self._linear = Linear(in_size, out_size)

                @declarative
                def forward(self, x):
                    y = self._linear(x)
                    z = self._linear(y)
                    return z

            # enable dygraph mode
            fluid.enable_dygraph() 

            # train model
            net = SimpleNet(8, 8)
            adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
            x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
            for i in range(10):
                out = net(x)
                loss = fluid.layers.mean(out)
                loss.backward()
                adam.minimize(loss)
                net.clear_gradients()

            # use SaveLoadconfig when saving model
            model_path = "simplenet.example.model"
            configs = fluid.dygraph.jit.SaveLoadConfig()
            configs.model_filename = "__simplenet__"
            fluid.dygraph.jit.save(
                layer=net,
                model_path=model_path,
                input_spec=[x],
                configs=configs)

        2. Using ``SaveLoadConfig`` when loading model

        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            # enable dygraph mode
            fluid.enable_dygraph() 

            # use SaveLoadconfig when loading model
            model_path = "simplenet.example.model"
            configs = fluid.dygraph.jit.SaveLoadConfig()
            configs.model_filename = "__simplenet__"
            infer_net = fluid.dygraph.jit.load(model_path, configs=configs)
            # inference
            x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
            pred = infer_net(x)
    """

    def __init__(self):
        self._output_spec = None
        self._model_filename = None
        self._params_filename = None
        self._separate_params = False

        # NOTE: Users rarely use following configs, so these configs are not open to users,
        # reducing user learning costs, but we retain the configuration capabilities

        # If True, programs are modified to only support direct inference deployment. 
        # Otherwise,more information will be stored for flexible optimization and re-training. 
        # Currently, only True is supported
        self._export_for_deployment = True

        # If True, It will save inference program only, and do not save params of Program
        self._program_only = False

    @property
    def output_spec(self):
        """
        Selects the output targets of the saved model ( :ref:`api_imperative_TranslatedLayer` ).
        By default, all return variables of original Layer's forward function
        are kept as the output of the saved TranslatedLayer.

        The ``output_spec`` type should be list[Variable]. If the provided ``output_spec``
        list is not all output variables, the saved model will be pruned according to the
        given ``output_spec`` list.

        .. note::
            The ``output_spec`` is only used when saving model.

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle.fluid as fluid
                from paddle.fluid.dygraph import Linear
                from paddle.fluid.dygraph import declarative

                class SimpleNet(fluid.dygraph.Layer):
                    def __init__(self, in_size, out_size):
                        super(SimpleNet, self).__init__()
                        self._linear = Linear(in_size, out_size)

                    @declarative
                    def forward(self, x):
                        y = self._linear(x)
                        z = self._linear(y)
                        loss = fluid.layers.mean(z)
                        return z, loss

                # enable dygraph mode
                fluid.enable_dygraph() 

                # train model
                net = SimpleNet(8, 8)
                adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
                x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
                for i in range(10):
                    out, loss = net(x)
                    loss.backward()
                    adam.minimize(loss)
                    net.clear_gradients()

                # use SaveLoadconfig.output_spec
                model_path = "simplenet.example.model.output_spec"
                configs = fluid.dygraph.jit.SaveLoadConfig()
                # only keep the predicted output in saved model, diccard loss
                configs.output_spec = [out]

                fluid.dygraph.jit.save(
                    layer=net,
                    model_path=model_path,
                    input_spec=[x],
                    configs=configs)

                infer_net = fluid.dygraph.jit.load(model_path, configs=configs)
                x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
                # only have the predicted output
                pred = infer_net(x)
        """
        return self._output_spec

    @output_spec.setter
    def output_spec(self, spec):
        if not isinstance(spec, list):
            raise TypeError(
                "The SaveLoadConfig.output_spec should be 'list', but received input type is %s."
                % type(input))
            for var in spec:
                if not isinstance(var, core.VarBase):
                    raise TypeError(
                        "The element in SaveLoadConfig.output_spec list should be 'Variable', but received element's type is %s."
                        % type(var))
        self._output_spec = spec

    @property
    def model_filename(self):
        """
        The name of file to save the translated program of target Layer.
        Default filename is :code:`__model__` .

        Exampels:
            .. code-block:: python

                import numpy as np
                import paddle.fluid as fluid
                from paddle.fluid.dygraph import Linear
                from paddle.fluid.dygraph import declarative

                class SimpleNet(fluid.dygraph.Layer):
                    def __init__(self, in_size, out_size):
                        super(SimpleNet, self).__init__()
                        self._linear = Linear(in_size, out_size)

                    @declarative
                    def forward(self, x):
                        y = self._linear(x)
                        z = self._linear(y)
                        return z

                # enable dygraph mode
                fluid.enable_dygraph() 

                # train model
                net = SimpleNet(8, 8)
                adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
                x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
                for i in range(10):
                    out = net(x)
                    loss = fluid.layers.mean(out)
                    loss.backward()
                    adam.minimize(loss)
                    net.clear_gradients()

                model_path = "simplenet.example.model.model_filename"
                configs = fluid.dygraph.jit.SaveLoadConfig()
                configs.model_filename = "__simplenet__"

                # saving with configs.model_filename
                fluid.dygraph.jit.save(
                    layer=net,
                    model_path=model_path,
                    input_spec=[x],
                    configs=configs)
                # [result] the saved model directory contains:
                # __simplenet__  __variables__  __variables.info__

                # loading with configs.model_filename
                infer_net = fluid.dygraph.jit.load(model_path, configs=configs)
                x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
                pred = infer_net(x)
        """
        return self._model_filename

    @model_filename.setter
    def model_filename(self, filename):
        if not isinstance(filename, six.string_types):
            raise TypeError(
                "The SaveLoadConfig.model_filename should be str, but received input's type is %s."
                % type(filename))
        if len(filename) == 0:
            raise ValueError(
                "The SaveLoadConfig.model_filename is empty string.")
        self._model_filename = filename

    @property
    def params_filename(self):
        """
        The name of file to save all persistable variables in target Layer. 
        Default file name is :code:`__variables__` .
        
        Exampels:
            .. code-block:: python

                import numpy as np
                import paddle.fluid as fluid
                from paddle.fluid.dygraph import Linear
                from paddle.fluid.dygraph import declarative

                class SimpleNet(fluid.dygraph.Layer):
                    def __init__(self, in_size, out_size):
                        super(SimpleNet, self).__init__()
                        self._linear = Linear(in_size, out_size)

                    @declarative
                    def forward(self, x):
                        y = self._linear(x)
                        z = self._linear(y)
                        return z

                # enable dygraph mode
                fluid.enable_dygraph() 

                # train model
                net = SimpleNet(8, 8)
                adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
                x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
                for i in range(10):
                    out = net(x)
                    loss = fluid.layers.mean(out)
                    loss.backward()
                    adam.minimize(loss)
                    net.clear_gradients()

                model_path = "simplenet.example.model.params_filename"
                configs = fluid.dygraph.jit.SaveLoadConfig()
                configs.params_filename = "__params__"

                # saving with configs.params_filename
                fluid.dygraph.jit.save(
                    layer=net,
                    model_path=model_path,
                    input_spec=[x],
                    configs=configs)
                # [result] the saved model directory contains:
                # __model__  __params__  __variables.info__

                # loading with configs.params_filename
                infer_net = fluid.dygraph.jit.load(model_path, configs=configs)
                x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
                pred = infer_net(x)
        """
        return self._params_filename

    @params_filename.setter
    def params_filename(self, filename):
        if not isinstance(filename, six.string_types):
            raise TypeError(
                "The SaveLoadConfig.params_filename should be str, but received input's type is %s."
                % type(filename))
        if len(filename) == 0:
            raise ValueError(
                "The SaveLoadConfig.params_filename is empty string.")
        self._params_filename = filename

    # NOTE: [why not use params_filename=None control params saved separately]
    # The new save interface does not recommend parameters to be saved separately. 
    # Here, the concept should be separated as clearly as possible. 
    # Setting params_filename=None only means that the saved file name is set 
    # and without any other meaning. New separate_params control for file saved
    # separately can makes the concept clearer.
    @property
    def separate_params(self):
        """
        Configure whether to save the Layer parameters as separete files.
        (In order to be compatible with the behavior of :ref:`api_fluid_io_save_inference_model` )

        If True, each parameter will be saved to a file separately, the file name is the parameter name,
        and the SaveLoadConfig.params_filename configuration will not take effect. Default False.

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle.fluid as fluid
                from paddle.fluid.dygraph import Linear
                from paddle.fluid.dygraph import declarative

                class SimpleNet(fluid.dygraph.Layer):
                    def __init__(self, in_size, out_size):
                        super(SimpleNet, self).__init__()
                        self._linear = Linear(in_size, out_size)

                    @declarative
                    def forward(self, x):
                        y = self._linear(x)
                        z = self._linear(y)
                        return z

                # enable dygraph mode
                fluid.enable_dygraph() 

                # train model
                net = SimpleNet(8, 8)
                adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
                x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
                for i in range(10):
                    out = net(x)
                    loss = fluid.layers.mean(out)
                    loss.backward()
                    adam.minimize(loss)
                    net.clear_gradients()

                model_path = "simplenet.example.model.separate_params"
                configs = fluid.dygraph.jit.SaveLoadConfig()
                configs.separate_params = True

                # saving with configs.separate_params
                fluid.dygraph.jit.save(
                    layer=net,
                    model_path=model_path,
                    input_spec=[x],
                    configs=configs)
                # [result] the saved model directory contains:
                # linear_0.b_0  linear_0.w_0  __model__  __variables.info__

                # loading with configs.params_filename
                infer_net = fluid.dygraph.jit.load(model_path, configs=configs)
                x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
                pred = infer_net(x)
        """
        return self._separate_params

    @separate_params.setter
    def separate_params(self, value):
        if not isinstance(value, bool):
            raise TypeError(
                "The SaveLoadConfig.separate_params should be bool value, but received input's type is %s."
                % type(value))
        self._separate_params = value


@switch_to_static_graph
def save(layer, model_path, input_spec=None, configs=None):
    """
    Saves input declarative Layer as :ref:`api_imperative_TranslatedLayer` 
    format model, which can be used for inference or fine-tuning after loading.

    It will save the translated program and all related persistable 
    variables of input declarative Layer to given ``model_path``.
    
    The default saved translated program file name is ``__model__``,
    and the default saved persistable variables file name is ``__variables__``,
    and it also saved some additional variable description information to file 
    ``__varibales.info__``, these additional information is used in fine-tuning.

    The saved model can be loaded by follow APIs:
      - :ref:`api_imperative_jit_load`
      - :ref:`api_fluid_io_load_inference_model` (need pass ``params_filename='__variables__'``)
      - Other C++ inference APIs

    Args:
        layer (Layer): the Layer to be saved. The Layer should be decorated by `@declarative`.
        model_path (str): the directory to save the model.
        input_spec (list[Varibale], optional): Describes the input of the saved model. 
            It is the example inputs that will be passed to saved TranslatedLayer's forward
            function. If None, all input variables of the original Layer's forward function
            would be the inputs of the saved model. Default None.
        configs (SaveLoadConfig, optional): :ref:`api_imperative_jit_saveLoadConfig` object
            that specifies additional configuration options. Default None.
    Returns:
        None

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid
            from paddle.fluid.dygraph import Linear
            from paddle.fluid.dygraph import declarative

            BATCH_SIZE = 32
            BATCH_NUM = 20

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

            class LinearNet(fluid.dygraph.Layer):
                def __init__(self, in_size, out_size):
                    super(LinearNet, self).__init__()
                    self._linear = Linear(in_size, out_size)

                @declarative
                def forward(self, x):
                    return self._linear(x)

            # enable dygraph mode
            fluid.enable_dygraph() 

            # create network
            net = LinearNet(784, 1)
            adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
            # create data loader
            train_loader = fluid.io.DataLoader.from_generator(capacity=5)
            train_loader.set_batch_generator(random_batch_reader())
            # train
            for data in train_loader():
                img, label = data
                label.stop_gradient = True

                cost = net(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()
                adam.minimize(avg_loss)
                net.clear_gradients()

            # save model
            model_path = "linear.example.model"
            fluid.dygraph.jit.save(
                layer=net,
                model_path=model_path,
                input_spec=[img])
    """

    def get_inout_spec(all_vars, target_vars, return_name=False):
        result_list = []
        valid_var_dict = {}
        valid_vars = [var for var in all_vars if isinstance(var, Variable)]
        for var in valid_vars:
            valid_var_dict[var.name] = var
        if target_vars:
            for i, var in enumerate(target_vars):
                # check target var whether exists
                if var.name not in valid_var_dict:
                    raise RuntimeError(
                        "The variable to feed/fetch are not exist.")
                result_list.append(valid_var_dict[var.name])
        else:
            result_list = valid_vars
        if return_name:
            result_list = [var.name for var in result_list]

        return result_list

    # 1. input check
    prog_translator = ProgramTranslator()
    if not prog_translator.enable:
        raise RuntimeError(
            "The paddle.jit.save doesn't work when setting ProgramTranslator.enable=False."
        )
    if not isinstance(layer, Layer):
        raise TypeError(
            "The input layer of paddle.jit.save should be 'Layer', but received layer type is %s."
            % type(layer))

    if configs is None:
        configs = SaveLoadConfig()

    if input_spec is not None:
        if not isinstance(input_spec, list):
            raise TypeError(
                "The input input_spec should be 'list', but received input_spec's type is %s."
                % type(input_spec))
        for var in input_spec:
            if not isinstance(var, core.VarBase):
                raise TypeError(
                    "The element in input_spec list should be 'Variable', but received element's type is %s."
                    % type(var))

    # 2. get program of declarative Layer.forward
    prog_cache = prog_translator.get_program_cache()
    # make dummy args & kwargs, to get excepted FunctionSpec
    layer_func = FunctionSpec(type(layer).forward, [layer], {})
    concrete_program, _ = prog_cache.get_program(layer_func)

    # NOTE: we maintain the mapping of variable name to
    # structured name, the buffer variable (non-persistable)
    # saved to inference program may not need by dygraph Layer, 
    # we only record the state_dict variable's structured name
    state_names_dict = dict()
    for structured_name, var in layer.state_dict().items():
        state_names_dict[var.name] = structured_name

    # 3. share parameters from Layer to scope & record var info
    scope = core.Scope()
    extra_var_info = dict()
    for param_or_buffer in concrete_program.parameters:
        # share to scope
        param_or_buffer_tensor = scope.var(param_or_buffer.name).get_tensor()
        src_tensor = param_or_buffer.value().get_tensor()
        param_or_buffer_tensor._share_data_with(src_tensor)
        # record var info
        extra_info_dict = dict()
        if param_or_buffer.name in state_names_dict:
            extra_info_dict['structured_name'] = state_names_dict[
                param_or_buffer.name]
        extra_info_dict['stop_gradient'] = param_or_buffer.stop_gradient
        if isinstance(param_or_buffer, ParamBase):
            extra_info_dict['trainable'] = param_or_buffer.trainable
        extra_var_info[param_or_buffer.name] = extra_info_dict

    # 4. build input & output spec
    input_var_names = get_inout_spec(concrete_program.inputs, input_spec, True)
    output_vars = get_inout_spec(concrete_program.outputs, configs.output_spec)

    # 5. save inference model
    from paddle.fluid.io import save_inference_model

    # VARIABLE_FILENAME keep nameing style consistent with '__model__'
    if configs.params_filename is None:
        configs.params_filename = VARIABLE_FILENAME

    with scope_guard(scope):
        save_inference_model(
            dirname=model_path,
            feeded_var_names=input_var_names,
            target_vars=output_vars,
            executor=Executor(_current_expected_place()),
            main_program=concrete_program.main_program.clone(),
            model_filename=configs.model_filename,
            params_filename=None
            if configs.separate_params else configs.params_filename,
            export_for_deployment=configs._export_for_deployment,
            program_only=configs._program_only)

        # NOTE: [ Save extra variable info ]
        # save_inference_model will lose some important variable information, including:
        #   - Variable name and correspondence (when saved variables as one file)
        #   - Variable.stop_gradient information
        #   - Which persistent variable are parameter and which are not
        #   - Parameter.trainable information
        #
        # The lost information cannot be recovered when it is loaded again, 
        # so if we want to perform fine-tune after loading, we may need to 
        # configure redundant information to proceed.
        #
        # Due to compatibility issues, we cannot change the original storage structure, 
        # but we can save these information in `jit.save` without changing the original 
        # storage to improve user experience. So we save extra information into
        # file `__variables.info__`
        extra_var_info_path = os.path.join(model_path, EXTRA_VAR_INFO_FILENAME)
        with open(extra_var_info_path, 'wb') as f:
            pickle.dump(extra_var_info, f, protocol=2)


@dygraph_only
def load(model_path, configs=None):
    """
    :api_attr: imperative

    Load model saved by :ref:`api_imperative_jit_save` or :ref:`api_fluid_io_save_inference_model`
    as :ref:`api_imperative_TranslatedLayer`, then performing inference or fine-tune training.

    .. note::
        For some historical reasons, if you load model saved by :ref:`api_fluid_io_save_inference_model`,
        there will be the following limitations when using it in fine-tuning:
        1. Imperative mode do not support LoDTensor. All original model's feed targets or parametars that depend on LoD are temporarily unavailable.
        2. All saved model's feed targets need to be passed into TranslatedLayer's forwrad function.
        3. The variable's ``stop_gradient`` information is lost and can not be recovered.
        4. The parameter's ``trainable`` information is lost and can not be recovered.

    Args:
        model_path (str): The directory path where the model is saved.
        configs (SaveLoadConfig, optional): :ref:`api_imperative_jit_saveLoadConfig` object that specifies 
            additional configuration options. Default None.

    Returns:
        TranslatedLayer: A Layer object can run saved translated model.

    Examples:
        1. Load model saved by :ref:`api_imperative_jit_save` then performing inference and fine-tune training.

        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid
            from paddle.fluid.dygraph import Linear
            from paddle.fluid.dygraph import declarative

            BATCH_SIZE = 32
            BATCH_NUM = 20

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

            class LinearNet(fluid.dygraph.Layer):
                def __init__(self, in_size, out_size):
                    super(LinearNet, self).__init__()
                    self._linear = Linear(in_size, out_size)

                @declarative
                def forward(self, x):
                    return self._linear(x)

            # enable dygraph mode
            fluid.enable_dygraph() 

            # 1. train & save model.
            # create network
            net = LinearNet(784, 1)
            adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
            # create data loader
            train_loader = fluid.io.DataLoader.from_generator(capacity=5)
            train_loader.set_batch_generator(random_batch_reader())
            # train
            for data in train_loader():
                img, label = data
                label.stop_gradient = True

                cost = net(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()
                adam.minimize(avg_loss)
                net.clear_gradients()

            model_path = "linear.example.model"
            fluid.dygraph.jit.save(
                layer=net,
                model_path=model_path,
                input_spec=[img])

            # 2. load model & inference
            # load model
            infer_net = fluid.dygraph.jit.load(model_path)
            # inference
            x = fluid.dygraph.to_variable(np.random.random((1, 784)).astype('float32'))
            pred = infer_net(x)

            # 3. load model & fine-tune
            # load model
            train_net = fluid.dygraph.jit.load(model_path)
            train_net.train()
            adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=train_net.parameters())
            # create data loader
            train_loader = fluid.io.DataLoader.from_generator(capacity=5)
            train_loader.set_batch_generator(random_batch_reader())
            # fine-tune
            for data in train_loader():
                img, label = data
                label.stop_gradient = True

                cost = train_net(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()
                adam.minimize(avg_loss)
                train_net.clear_gradients()

        2. Load model saved by :ref:`api_fluid_io_save_inference_model` then performing and fine-tune training.

        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            BATCH_SIZE = 32
            BATCH_NUM = 20

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

            img = fluid.data(name='img', shape=[None, 784], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            pred = fluid.layers.fc(input=img, size=10, act='softmax')
            loss = fluid.layers.cross_entropy(input=pred, label=label)
            avg_loss = fluid.layers.mean(loss)

            optimizer = fluid.optimizer.SGD(learning_rate=0.001)
            optimizer.minimize(avg_loss)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            loader = fluid.io.DataLoader.from_generator(
                feed_list=[img, label], capacity=5, iterable=True)
            loader.set_batch_generator(random_batch_reader(), places=place)

            # 1. train and save inference model
            for data in loader():
                exe.run(
                    fluid.default_main_program(),
                    feed=data, 
                    fetch_list=[avg_loss])

            model_path = "fc.example.model"
            fluid.io.save_inference_model(
                model_path, ["img"], [pred], exe)

            # enable dygraph mode
            fluid.enable_dygraph() 

            # 2. load model & inference
            fc = fluid.dygraph.jit.load(model_path)
            x = fluid.dygraph.to_variable(np.random.random((1, 784)).astype('float32'))
            pred = fc(x)

            # 3. load model & fine-tune
            fc = fluid.dygraph.jit.load(model_path)
            fc.train()
            sgd = fluid.optimizer.SGD(learning_rate=0.001,
                                        parameter_list=fc.parameters())

            train_loader = fluid.io.DataLoader.from_generator(capacity=5)
            train_loader.set_batch_generator(
                random_batch_reader(), places=place)

            for data in train_loader():
                img, label = data
                label.stop_gradient = True

                cost = fc(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()
                sgd.minimize(avg_loss)
    """
    return TranslatedLayer._construct(model_path, configs)


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
    :api_attr: imperative
    
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
            inputs (list(Tensor)|tuple(Tensor)|Tensor): the input tensors of
                the layer object.

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
        assert isinstance(
            layer, Layer
        ), "The type of 'layer' in fluid.dygraph.jit.TracedLayer.trace must be fluid.dygraph.Layer, but received {}.".format(
            type(layer))
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
        assert isinstance(
            build_strategy, (type(None), BuildStrategy)
        ), "The type of 'build_strategy' in fluid.dygraph.jit.TracedLayer.set_strategy must be fluid.BuildStrategy, but received {}.".format(
            type(build_strategy))
        assert isinstance(
            exec_strategy, (type(None), ExecutionStrategy)
        ), "The type of 'exec_strategy' in fluid.dygraph.jit.TracedLayer.set_strategy must be fluid.ExecutionStrategy, but received {}.".format(
            type(exec_strategy))
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
        check_type(dirname, "dirname", str,
                   "fluid.dygraph.jit.TracedLayer.save_inference_model")
        check_type(feed, "feed", (type(None), list),
                   "fluid.dygraph.jit.TracedLayer.save_inference_model")
        if isinstance(feed, list):
            for f in feed:
                check_type(f, "each element of feed", int,
                           "fluid.dygraph.jit.TracedLayer.save_inference_model")
        check_type(fetch, "fetch", (type(None), list),
                   "fluid.dygraph.jit.TracedLayer.save_inference_model")
        if isinstance(fetch, list):
            for f in fetch:
                check_type(f, "each element of fetch", int,
                           "fluid.dygraph.jit.TracedLayer.save_inference_model")

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
