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
import decorator
import contextlib
import functools
import sys
import numpy as np
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.multiprocess_utils import CleanupFuncRegistrar
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
    try:
        yield
    finally:
        if tracer:
            tracer._enable_program_desc_tracing = original_val


_functional_dygraph_context_manager = None


@signature_safe_contextmanager
def param_guard(parameters):
    # Note: parameters is a reference of self._parameters
    if not framework.in_dygraph_mode() and parameters:
        origin_parameters = parameters.copy()
        for name, var_base in parameters.items():
            if isinstance(var_base, core.VarBase):
                new_var = framework.Parameter(
                    var_base.block,
                    var_base.shape,
                    var_base.dtype,
                    var_base.type,
                    name=var_base.name)
                parameters[name] = new_var
        yield
        parameters.update(origin_parameters)
    else:
        yield


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
    :alias_main: paddle.enable_dygraph
	:alias: paddle.enable_dygraph,paddle.enable_imperative.enable_dygraph
	:old_api: paddle.fluid.dygraph.base.enable_dygraph

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
    if _functional_dygraph_context_manager is None:
        _functional_dygraph_context_manager = guard(place=place)
        _functional_dygraph_context_manager.__enter__()

        # call disable_dygraph when Python exit
        CleanupFuncRegistrar.register(disable_dygraph)


def disable_dygraph():
    """
    :alias_main: paddle.disable_dygraph
	:alias: paddle.disable_dygraph,paddle.disable_imperative.disable_dygraph
	:old_api: paddle.fluid.dygraph.base.disable_dygraph

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


@signature_safe_contextmanager
def _switch_tracer_mode_guard_(is_train=True):
    tracer = framework._dygraph_tracer()
    if tracer:
        mode = tracer._train_mode
        tracer._train_mode = is_train
        try:
            yield
        finally:
            tracer._train_mode = mode
    else:
        yield


def no_grad(func=None):
    """
    :api_attr: imperative

    Create a context which disables dygraph gradient calculation.
    In this mode, the result of every computation will have `stop_gradient=True`.

    Also functions as a decorator. (Make sure to instantiate without parenthesis.)

    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        # use as generator

        data = np.array([[2, 3], [4, 5]]).astype('float32')
        with fluid.dygraph.guard():
            l0 = fluid.Linear(2, 2)  # l0.weight.gradient() is None
            l1 = fluid.Linear(2, 2)
            with fluid.dygraph.no_grad():
                # l1.weight.stop_gradient is False
                tmp = l1.weight * 2  # tmp.stop_gradient is True
            x = fluid.dygraph.to_variable(data)
            y = l0(x) + tmp
            o = l1(y)
            o.backward()
            print(tmp.gradient() is None)  # True
            print(l0.weight.gradient() is None)  # False

        # use as decorator

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
    if func is None:
        return _switch_tracer_mode_guard_(is_train=False)
    else:

        @decorator.decorator
        def __impl__(func, *args, **kwargs):
            with _switch_tracer_mode_guard_(is_train=False):
                return func(*args, **kwargs)

        return __impl__(func)


@signature_safe_contextmanager
def guard(place=None):
    """
    :api_attr: imperative

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
         retain_graph=None,
         create_graph=False,
         only_inputs=True,
         allow_unused=False,
         no_grad_vars=None,
         backward_strategy=None):
    ''' 
    .. note::
        **This API is ONLY available in Dygraph mode.**

    This API computes the sum of gradients of `outputs` with respect to each `inputs` .

    Parameters:
        outputs (Variable|list(Variable)|tuple(Variable)): the output Variable or 
            Variable list/tuple of the graph to compute gradients.
        inputs (Variable|list(Variable)|tuple(Variable)): the input Variable or 
            Variable list/tuple of the graph to compute gradients. The returned
            values of this API are the gradients of `inputs` . 
        grad_outputs (Variable|list(Variable|None)|tuple(Variable|None), optional): 
            initial gradient values of `outputs` . If `grad_outputs` is None, 
            the initial gradient values of `outputs` would be Tensors filled with 1; 
            if `grad_outputs` is not None, it must have the same length as `outputs` , 
            and in this case, the initial gradient value of the i-th `outputs` would
            be: (1) a Tensor filled with 1 when the i-th element of `grad_outputs` 
            is None; (2) the i-th element of `grad_outputs` when the i-th element of
            `grad_outputs` is a Variable. Default None.
        retain_graph (bool, optional): whether to retain the forward graph which 
            is used to calculate the gradient. When it is True, the graph would 
            be retained, in which way users can calculate backward twice for the 
            same graph. When it is False, the graph would be freed. Default None,
            which means it is equal to `create_graph` . 
        create_graph (bool, optional): whether to create the gradient graphs of
            the computing process. When it is True, higher order derivatives are
            supported to compute; when it is False, the gradient graphs of the
            computing process would be discarded. Default False.
        only_inputs (bool, optional): whether to only compute the gradients of
            `inputs` . If it is False, the gradients of all remaining leaf 
            Variables in the graph would be also computed and accumulated. 
            If it is True, only the gradients of `inputs` would be computed.
            Default True. only_inputs=False is under development, and it is
            not supported yet.    
        allow_unused (bool, optional): whether to raise error or return None if some 
            Variables of `inputs` are unreachable in the graph. If some Variables of 
            `inputs` are unreachable in the graph (i.e., their gradients are None),  
            error would be raised if allow_unused=False, or None would be returned as
            their gradients if allow_unused=True. Default False.
        no_grad_vars (Variable|list(Variable)|tuple(Variable)|set(Variable), optional): 
            the Variables whose gradients are not needed to compute. Default None.
        backward_strategy (BackwardStrategy, optional): The backward strategy to
            compute gradients. See :ref:`api_fluid_dygraph_BackwardStrategy` for
            details. Default None.

    Returns:
        tuple: a tuple of Variables, whose length is the same as the Variable number 
        inside `inputs`, and the i-th returned Variable is the sum of gradients of 
        `outputs` with respect to the i-th `inputs`.

    Examples 1:
        .. code-block:: python

            import paddle.fluid as fluid

            def test_dygraph_grad(create_graph):
                with fluid.dygraph.guard(): 
                    x = fluid.layers.ones(shape=[1], dtype='float32') 
                    x.stop_gradient = False
                    y = x * x

                    # Since y = x * x, dx = 2 * x 
                    dx = fluid.dygraph.grad(
                            outputs=[y],
                            inputs=[x], 
                            create_graph=create_graph, 
                            retain_graph=True)[0]

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

    Examples 2:
        .. code-block:: python

            import paddle.fluid as fluid

            fluid.enable_dygraph()

            def test_dygraph_grad(grad_outputs=None):
                x = fluid.layers.fill_constant(shape=[1], value=2.0, dtype='float32')
                x.stop_gradient = False

                y1 = x * x
                y2 = x * 3 

                # If grad_outputs=None, dy1 = [1], dy2 = [1].
                # If grad_outputs=[g1, g2], then:
                #    - dy1 = [1] if g1 is None else g1
                #    - dy2 = [1] if g2 is None else g2

                # Since y1 = x * x, dx = 2 * x * dy1.
                # Since y2 = x * 3, dx = 3 * dy2.
                # Therefore, the final result would be:
                # dx = 2 * x * dy1 + 3 * dy2 = 4 * dy1 + 3 * dy2.

                dx = fluid.dygraph.grad(
                    outputs=[y1, y2], 
                    inputs=[x],
                    grad_outputs=grad_outputs)[0]

                return dx.numpy()

            THREE = fluid.layers.fill_constant(shape=[1], value=3.0, dtype='float32')
            FOUR = fluid.layers.fill_constant(shape=[1], value=4.0, dtype='float32')

            # dy1 = [1], dy2 = [1]
            print(test_dygraph_grad(None)) # [7.]

            # dy1 = [1], dy2 = [4]
            print(test_dygraph_grad([None, FOUR])) # [16.] 

            # dy1 = [4], dy2 = [1]
            print(test_dygraph_grad([FOUR, None])) # [19.]

            # dy1 = [3], dy2 = [4]
            print(test_dygraph_grad([THREE, FOUR])) # [24.]
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

    if no_grad_vars is None:
        no_grad_vars = []
    elif isinstance(no_grad_vars, core.VarBase):
        no_grad_vars = [no_grad_vars]
    elif isinstance(no_grad_vars, (list, tuple, set)):
        no_grad_vars = list(no_grad_vars)
        for var in no_grad_vars:
            assert isinstance(
                var, core.VarBase), "no_grad_vars can only contains Variable"
    else:
        raise AssertionError(
            "no_grad_vars must be None, Variable or list/tuple/set of Variables")

    if backward_strategy is None:
        backward_strategy = core.BackwardStrategy()

    assert isinstance(backward_strategy, core.BackwardStrategy), \
        "backward_strategy must be type paddle.fluid.dygraph.BackwardStrategy"

    assert isinstance(create_graph, bool), "create_graph must be True or False"

    if retain_graph is None:
        retain_graph = create_graph

    assert isinstance(retain_graph,
                      bool), "retain_graph must be None, True or False"

    assert isinstance(allow_unused, bool), "allow_unused must be True or False"

    assert isinstance(only_inputs, bool), "only_inputs must be True or False"
    assert only_inputs, "only_inputs=False is not supported yet"

    place = core.Place()
    place.set_place(framework._current_expected_place())
    return core.dygraph_partial_grad(
        inputs, outputs, grad_outputs, no_grad_vars, place, backward_strategy,
        create_graph, retain_graph, allow_unused, only_inputs)


@framework.dygraph_only
def to_variable(value, name=None, zero_copy=None):
    """
    :api_attr: imperative

    The API will create a ``Variable`` or ``ComplexVariable`` object from 
    numpy\.ndarray, Variable or ComplexVariable object.

    Parameters:
        value(ndarray|Variable|ComplexVariable): The numpy\.ndarray, Variable 
            or ComplexVariable object that needs to be converted, it can be 
            multi-dimension, and the data type is one of numpy\.{float16, 
            float32, float64, int16, int32, int64, uint8, uint16, complex64, 
            complex128}.
        name(str, optional): The default value is None. Normally there is no 
            need for user to set this property. For more information, please 
            refer to :ref:`api_guide_Name` .
        zero_copy(bool, optional): Whether to share memory with the input numpy 
            array. This parameter only works with CPUPlace and will be set to 
            True when it is None. Default: None.

    Returns:
        Variable or ComplexVariable: If ``value`` is a numpy\.ndarray object, return ``Tensor`` created from the specified numpy\.ndarray object, which has same data type and shape with ``value``. If ``value`` is a Variable or ComplexVariable object, just return ``value``.


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
            c = np.array([2+1j, 2])
            z = fluid.dygraph.to_variable(c)
            z.numpy() # array([2.+1.j, 2.+0.j])
            z.dtype # 'complex128'
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
        if np.iscomplexobj(value):
            if not name:
                name = framework.unique_name.generate('_generated_var')
            real_var = core.VarBase(
                value=value.real,
                place=framework._current_expected_place(),
                persistable=False,
                zero_copy=zero_copy,
                name=name + ".real")
            imag_var = core.VarBase(
                value=value.imag,
                place=framework._current_expected_place(),
                persistable=False,
                zero_copy=zero_copy,
                name=name + ".imag")
            return framework.ComplexVariable(real_var, imag_var)
        else:
            py_var = core.VarBase(
                value=value,
                place=framework._current_expected_place(),
                persistable=False,
                zero_copy=zero_copy,
                name=name if name else '')
            return py_var
    elif isinstance(value, (core.VarBase, framework.Variable,
                            framework.ComplexVariable)):
        return value
    else:
        raise TypeError(
            "The type of input value is invalid, expected type is 'ndarray', "
            "'Variable' or 'ComplexVariable', but received %s." % type(value))
