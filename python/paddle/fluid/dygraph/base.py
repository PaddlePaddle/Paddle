# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from ..wrapped_decorator import signature_safe_contextmanager, wrap_decorator
import contextlib
import sys
import numpy as np
from paddle.fluid import core
from paddle.fluid import framework
from .tracer import Tracer
import logging
import objgraph

__all__ = [
    'no_grad',
    'grad',
    'guard',
    'enable_dygraph',
    'disable_dygraph',
    'enabled',
    'to_variable',
]


def _switch_to_static_graph_(func):
    def __impl__(*args, **kwargs):
        with framework._dygraph_guard(None):
            return func(*args, **kwargs)

    return __impl__


switch_to_static_graph = wrap_decorator(_switch_to_static_graph_)


@signature_safe_contextmanager
def program_desc_tracing_guard(enable):
    tracer = framework._dygraph_tracer()
    if tracer:
        original_val = tracer._enable_program_desc_tracing
        tracer._enable_program_desc_tracing = enable
    yield
    if tracer:
        tracer._enable_program_desc_tracing = original_val


_functional_dygraph_context_manager = None


def enabled():
    """
    This function checks whether the program runs in dynamic graph mode or not.
    You can enter dynamic graph mode with :ref:`api_fluid_dygraph_guard` api,
    or enable and disable dynamic graph mode with :ref:`api_fluid_dygraph_enable`
    and :ref:`api_fluid_dygraph_disable` api .

    **Note**:
        ``fluid.dygraph.enabled`` is the alias of ``fluid.in_dygraph_mode``, and
        ``fluid.in_dygraph_mode`` is recommended to use.

    Returns:
        bool: Whether the program is running in dynamic graph mode.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            fluid.enable_dygraph()  # Now we are in dygragh mode
            print(fluid.dygraph.enabled())  # True
            fluid.disable_dygraph()
            print(fluid.dygraph.enabled())  # False
    """
    return framework.in_dygraph_mode()


def enable_dygraph(place=None):
    """
    This function enables dynamic graph mode.

    Parameters:
        place(fluid.CPUPlace or fluid.CUDAPlace, optional): Place to execute dygraph.
            If None, the running place will be determined according to the way of paddle compilation. Default: None

    return:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            fluid.enable_dygraph()  # Now we are in dygragh mode
            print(fluid.in_dygraph_mode())  # True
            fluid.disable_dygraph()
            print(fluid.in_dygraph_mode())  # False
    """
    global _functional_dygraph_context_manager
    _functional_dygraph_context_manager = guard(place=place)
    _functional_dygraph_context_manager.__enter__()


def disable_dygraph():
    """
    This function disables dynamic graph mode.

    return:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            fluid.enable_dygraph()  # Now we are in dygragh mode
            print(fluid.in_dygraph_mode())  # True
            fluid.disable_dygraph()
            print(fluid.in_dygraph_mode())  # False
    """
    global _functional_dygraph_context_manager
    if _functional_dygraph_context_manager is not None:
        _functional_dygraph_context_manager.__exit__(*sys.exc_info())
        _functional_dygraph_context_manager = None


@contextlib.contextmanager
def _switch_tracer_mode_guard_(is_train=True):
    tracer = framework._dygraph_tracer()
    if tracer:
        mode = tracer._train_mode
        tracer._train_mode = is_train
        yield
        tracer._train_mode = mode
    else:
        yield


def _no_grad_(func):
    """
    This Decorator will avoid the func being decorated creating backward network in dygraph mode

    Parameter:
        - **func** (python func): the func don't need grad

    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        @fluid.dygraph.no_grad
        def test_layer():
            with fluid.dygraph.guard():
                inp = np.ones([3, 1024], dtype='float32')
                t = fluid.dygraph.base.to_variable(inp)
                linear1 = fluid.Linear(1024, 4, bias_attr=False)
                linear2 = fluid.Linear(4, 4)
                ret = linear1(t)
                dy_ret = linear2(ret)

        test_layer()

    """

    def __impl__(*args, **kwargs):
        with _switch_tracer_mode_guard_(is_train=False):
            return func(*args, **kwargs)

    return __impl__


no_grad = wrap_decorator(_no_grad_)
# for fluidDoc
no_grad.__doc__ = _no_grad_.__doc__


@signature_safe_contextmanager
def guard(place=None):
    """
    This context will create a dygraph context for dygraph to run, using python ``with`` statement.

    Parameters:
        place(fluid.CPUPlace or fluid.CUDAPlace, optional): Place to execute dygraph. 
            If None, the running place will be determined according to the way of paddle compilation. Default: None

    return:
        None

    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        with fluid.dygraph.guard():
            inp = np.ones([3, 1024], dtype='float32')
            t = fluid.dygraph.base.to_variable(inp)
            linear1 = fluid.Linear(1024, 4, bias_attr=False)
            linear2 = fluid.Linear(4, 4)
            ret = linear1(t)
            dy_ret = linear2(ret)

    """
    train = framework.Program()
    startup = framework.Program()
    tracer = Tracer()
    VarBase = core.VarBase

    if place is None:
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
    tracer._expected_place = place

    with framework.program_guard(train, startup):
        with framework.unique_name.guard():
            with framework._dygraph_guard(tracer):
                with framework._dygraph_place_guard(place):
                    yield


def _print_debug_msg(parameter_list, limit=5, is_test=False):
    if not core._is_dygraph_debug_enabled():
        logging.warn(
            'Debug mode is not enabled. Please set FLAGS_dygraph_debug=1 to enable debug'
        )
        return
    unique_name_size = len(framework.unique_name.generator.ids)
    tracer_var_size = len(parameter_list)
    alive_cpp_var_size = len(core.VarBase._alive_vars())
    if not is_test:
        logging.warn(
            'unique_name num: {}, tracer vars num: {}, alive cpp vars num: {}'
            .format(unique_name_size, tracer_var_size, alive_cpp_var_size))
        objgraph.show_growth(limit=limit)
    else:
        return unique_name_size, tracer_var_size, alive_cpp_var_size


@framework.dygraph_only
def grad(outputs,
         inputs,
         grad_outputs=None,
         no_grad_set=None,
         create_graph=False,
         backward_strategy=None):
    ''' 
    .. note::
        **This API is ONLY available in Dygraph mode.**

    This API computes the sum of gradients of `outputs` with respect to each `inputs` .

    Parameters:
        outputs (Variable|list(Variable)|tuple(Variable)): any Variable or a 
            list/tuple of any Variables.
        inputs (Variable|list(Variable)|tuple(Variable)): any Variable or a 
            list/tuple of any Variables.
        grad_outputs (Variable|list(Variable|None)|tuple(Variable|None), optional): 
            initial gradient values of `outputs` . If `grad_outputs` is None, 
            the initial gradient values of `outputs` would be Tensors filled with 1; 
            if `grad_outputs` is not None, it must have the same length as `outputs` , 
            and in this case, the initial gradient value of the i-th `outputs` would
            be: (1) a Tensor filled with 1 when the i-th element of `grad_outputs` 
            is None; (2) the i-th element of `grad_outputs` when i-th element of
            `grad_outputs` is a Variable. Default None.
        no_grad_set (Variable|list(Variable)|tuple(Variable)|set(Variable), optional): 
            the Variables whose gradients are not needed to compute. Default None.
        create_graph (bool, optional): whether to create the gradient graphs of
            the computing process. When it is True, higher order derivatives are
            supported to compute; when it is False, the gradient graphs of the
            computing process would be discarded. Default False.
        backward_strategy (BackwardStrategy, optional): The backward strategy to
            compute gradients. See :ref:`api_fluid_dygraph_BackwardStrategy` for
            details. Default None.

    Returns:
        tuple: a tuple of Variable, whose length is the same as the Variable number 
        inside `inputs`, and the i-th returned Variable is the sum of gradients of 
        `outputs` with respect to the i-th `inputs`.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            def test_dygraph_grad(create_graph):
                with fluid.dygraph.guard(): 
                    x = fluid.layers.ones(shape=[1], dtype='float32') 
                    x.stop_gradient = False
                    y = x * x

                    # Since y = x * x, dx = 2 * x 
                    dx = fluid.dygraph.grad(outputs=[y], inputs=[x], 
                            create_graph=create_graph)[0]

                    z = y + dx

                    # If create_graph = False, the gradient of dx
                    # would not be backpropagated. Therefore,
                    # z = x * x + dx, and x.gradient() = 2 * x = 2.0
                    
                    # If create_graph = True, the gradient of dx
                    # would be backpropagated. Therefore, 
                    # z = x * x + dx = x * x + 2 * x, and
                    # x.gradient() = 2 * x + 2 = 4.0 

                    z.backward()
                    return x.gradient() 

            print(test_dygraph_grad(create_graph=False)) # [2.] 
            print(test_dygraph_grad(create_graph=True)) # [4.]
	'''

    def check_in_out(in_out_list, name):
        assert in_out_list is not None, "{} should not be None".format(name)

        if isinstance(in_out_list, (list, tuple)):
            assert len(in_out_list) > 0, "{} cannot be empty".format(name)
            for each_var in in_out_list:
                assert isinstance(
                    each_var,
                    core.VarBase), "Elements of {} must be Variable".format(
                        name)
            return in_out_list
        else:
            assert isinstance(
                in_out_list,
                core.VarBase), "{} must be Variable or list of Variable".format(
                    name)
            return [in_out_list]

    outputs = check_in_out(outputs, 'outputs')
    inputs = check_in_out(inputs, 'inputs')

    if grad_outputs is not None:
        if not isinstance(grad_outputs, (list, tuple)):
            grad_outputs = [grad_outputs]

        for each_var in grad_outputs:
            if each_var is not None:
                assert isinstance(
                    each_var, core.VarBase
                ), "grad_outputs must be None, a Variable or a list containing None or Variables"
    else:
        grad_outputs = []

    if len(grad_outputs) > 0:
        assert len(grad_outputs) == len(
            outputs), "The length of grad_outputs must be equal to outputs"

    if no_grad_set is None:
        no_grad_set = []
    elif isinstance(no_grad_set, core.VarBase):
        no_grad_set = [no_grad_set]
    elif isinstance(no_grad_set, (list, tuple, set)):
        no_grad_set = list(no_grad_set)
        for var in no_grad_set:
            assert isinstance(
                var, core.VarBase), "no_grad_set can only contains Variable"
    else:
        raise AssertionError(
            "no_grad_set must be None, Variable or list/tuple/set of Variables")

    if backward_strategy is None:
        backward_strategy = core.BackwardStrategy()

    assert isinstance(backward_strategy, core.BackwardStrategy), \
        "backward_strategy must be type paddle.fluid.dygraph.BackwardStrategy"

    assert isinstance(create_graph, bool), "create_graph must be True or False"

    place = core.Place()
    place.set_place(framework._current_expected_place())
    return core.dygraph_partial_grad(inputs, outputs, grad_outputs, no_grad_set,
                                     place, backward_strategy, create_graph)


@framework.dygraph_only
def to_variable(value, name=None, zero_copy=None):
    """
    The API will create a ``Variable`` object from numpy\.ndarray or Variable object.

    Parameters:
        value(ndarray|Variable): The numpy\.ndarray or Variable object that needs to be converted, it can be multi-dimension, and the data type is one of numpy\.{float16, float32, float64, int16, int32, int64, uint8, uint16}.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`
        zero_copy(bool, optional): Whether to share memory with the input numpy array. This parameter only works with CPUPlace and will be set to True when it is None. Default: None.

    Returns:
        Variable: If ``value`` is a numpy\.ndarray object, return ``Tensor`` created from the specified numpy\.ndarray object, which has same data type and shape with ``value``. If ``value`` is a Variable object, just return ``value``.


    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        with fluid.dygraph.guard(fluid.CPUPlace()):
            x = np.ones([2, 2], np.float32)
            y = fluid.dygraph.to_variable(x, zero_copy=False)
            x[0][0] = -1
            y[0][0].numpy()  # array([1.], dtype=float32)
            y = fluid.dygraph.to_variable(x)
            x[0][0] = 0
            y[0][0].numpy()  # array([0.], dtype=float32)

    """
    if isinstance(value, np.ndarray):
        assert framework.in_dygraph_mode(
        ), "to_variable could only be called in dygraph mode"
        if isinstance(framework._current_expected_place(),
                      framework.core.CPUPlace):
            if zero_copy is None:
                zero_copy = True
        else:
            assert not zero_copy, "zero_copy mode can only be used with CPUPlace"
            zero_copy = False
        py_var = core.VarBase(
            value=value,
            place=framework._current_expected_place(),
            persistable=False,
            zero_copy=zero_copy,
            name=name if name else '')
        return py_var
    elif isinstance(value, (core.VarBase, framework.Variable)):
        return value
    else:
        raise TypeError(
            "to_variable only accepts 'ndarray' and 'Variable' as value's input")
