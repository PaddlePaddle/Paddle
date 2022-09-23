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

import inspect
import numpy as np
import warnings
import weakref
import sys

import paddle
from .. import framework
from ..framework import convert_np_dtype_to_dtype_, _in_legacy_dygraph
from .. import core
from .. import unique_name
from ..framework import Variable, Parameter, ParamBase, _getitem_impl_, _setitem_impl_, EagerParamBase, in_dygraph_mode
from .base import switch_to_static_graph
from .math_op_patch import monkey_patch_math_varbase
from .parallel import scale_loss
from paddle.fluid.data_feeder import convert_dtype, _PADDLE_DTYPE_2_NUMPY_DTYPE
import paddle.utils.deprecated as deprecated
import paddle.profiler as profiler
from paddle.profiler.utils import in_profiler_mode
from paddle import _C_ops, _legacy_C_ops

_grad_scalar = None


class TensorHookRemoveHelper(object):
    """
    A helper class that for removing Tensor gradient's hook.
    NOTE(wuweilong):the operation weakref.ref(tensor) will cause some unexpected errors in eager mode.
    """

    def __init__(self, tensor, hook_id):
        self._tensor = tensor if framework._in_eager_mode_ else weakref.ref(
            tensor)
        self._hook_id = hook_id

    def remove(self):
        """
        Remove reference Tensor's hook.

        Returns:
            bool: Return True if removed successfully
        """
        tensor = self._tensor if framework._in_eager_mode_ else self._tensor()
        if tensor is not None:
            res = tensor._remove_grad_hook(self._hook_id)
            if res is True:
                return True
            else:
                warnings.warn(
                    "The backward hook (ID: %d) of Tensor `%s` you want to remove does not exist or has been removed."
                    % (self._hook_id, tensor.name), RuntimeWarning)
        return False


_already_patch_repr = False


def monkey_patch_varbase():

    @switch_to_static_graph
    def _to_static_var(self, to_parameter=False, **kwargs):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**

        Transform a VarBase into static Variable with same attributes. It's a low level interface used
        in dy2static and shall not be called directly.

        Args:
            to_parameter (bool): It takes effect only if the input a VarBase. If set True,
                                 the VarBase will be converted into framework.Parameters. Otherwise, it will
                                 be converted into framework.Variable. Default False.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                from paddle.fluid.dygraph.base import to_variable
                import numpy as np

                data = np.ones([3, 1024], dtype='float32')
                with fluid.dygraph.guard():
                    var_base = to_variable(data)
                    static_var = var_base._to_static_var()

        """

        # Note: getattr(self, attr, None) will call x.grad=x.gradient(), but gradient() only available in dygraph.
        # It will fail. So, for propery that different between dynamic and static graph, should not getattr(self, attr, None).
        attr_not_need_keys = ['grad', 'T', 'place', '_place_str']
        param_keys = ['stop_gradient', 'trainable']
        if isinstance(self, (ParamBase, EagerParamBase)):
            attr_kwargs = self.__dict__.copy()
            for key in param_keys:
                attr_kwargs[key] = getattr(self, key)
        else:
            attr_names = []
            for name in dir(self):
                if name not in attr_not_need_keys:
                    if not inspect.ismethod(getattr(
                            self, name)) and not name.startswith('_'):
                        attr_names.append(name)
            attr_kwargs = {name: getattr(self, name) for name in attr_names}

        attr_keys = ['block', 'shape', 'dtype', 'type', 'name', 'persistable']
        for attr in attr_keys:
            attr_kwargs[attr] = getattr(self, attr, None)

        # If specify block, use it instead of self.block
        if 'block' in kwargs:
            attr_kwargs['block'] = kwargs['block']

        attr_kwargs.update(kwargs)

        if to_parameter or isinstance(self, (ParamBase, EagerParamBase)):
            del attr_kwargs['persistable']
            # NOTE(Aurelius84): All parameters should be placed into global block.
            attr_kwargs['block'] = attr_kwargs['block'].program.global_block()
            static_var = Parameter(**attr_kwargs)
        else:
            static_var = Variable(**attr_kwargs)
        return static_var

    # TODO(jiabin): move this to cplusplus end if we find some performance issue on it
    @framework.dygraph_only
    def set_value(self, value):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**

        Set a new value for this Variable.

        Args:
            value (Variable|np.ndarray): the new value.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                from paddle.fluid.dygraph.base import to_variable
                from paddle.fluid.dygraph import Linear
                import numpy as np

                data = np.ones([3, 1024], dtype='float32')
                with fluid.dygraph.guard():
                    linear = fluid.dygraph.Linear(1024, 4)
                    t = to_variable(data)
                    linear(t)  # call with default weight
                    custom_weight = np.random.randn(1024, 4).astype("float32")
                    linear.weight.set_value(custom_weight)  # change existing weight
                    out = linear(t)  # call with different weight

        """
        if framework._in_eager_mode_:
            base_tensor = core.eager.Tensor
        else:
            base_tensor = core.VarBase
        assert isinstance(value, (np.ndarray, base_tensor, dict, str)), \
            "Variable set_value function, arguments type only support Variable, numpy, VarBase, dict, string."

        if isinstance(value, (dict, str)):
            assert len(self) == len(
                value
            ), "Variable length not match, Variable [ {} ] need tensor with length {} but load set tensor with length {}".format(
                self.name, len(self), len(value))
            if isinstance(value, dict):
                self.value().set_vocab(value)
            else:
                self.value().set_string_list(value)
        else:
            assert self.shape == list(value.shape),  \
                "Variable Shape not match, Variable [ {} ] need tensor with shape {} but load set tensor with shape {}".format(
                    self.name, self.shape, value.shape)

            if isinstance(value, base_tensor):
                dtype = value.dtype
            else:
                dtype = convert_np_dtype_to_dtype_(value.dtype)

            assert self.dtype == dtype, \
                "Variable dtype not match, Variable [ {} ] need tensor with dtype {}  but load tensor with dtype {}".format(
                    self.name, self.dtype, dtype)

            # NOTE(wuweilong): self could be VarBase or Tensor, the subsequent behavior are defined in different files
            # if self is VarBase, method value() return Variable that bindded in imperative.cc, get_tensor() bindded in pybind.cc
            # if self is Tensor, method value() return self that defined in this file, get_tensor() defined in eager_method.cc
            # this Interface behavior will be unifed in the future.
            self.value().get_tensor().set(value,
                                          framework._current_expected_place())

    @framework.dygraph_only
    def backward(self, grad_tensor=None, retain_graph=False):
        """
        Run backward of current Graph which starts from current Tensor.

        The new gradient will accumulat on previous gradient.

        You can clear gradient by ``Tensor.clear_grad()`` .

        Args:
            grad_tensor(Tensor, optional): initial gradient values of the current Tensor. If `grad_tensor` is None,
            the initial gradient values of the current Tensor would be Tensor filled with 1.0;
            if `grad_tensor` is not None, it must have the same length as the current Tensor.
            Teh default value is None.

            retain_graph(bool, optional): If False, the graph used to compute grads will be freed. If you would
                like to add more ops to the built graph after calling this method( :code:`backward` ), set the parameter
                :code:`retain_graph` to True, then the grads will be retained. Thus, seting it to False is much more memory-efficient.
                Defaults to False.
        Returns:
            NoneType: None

        Examples:
            .. code-block:: python

                import paddle
                x = paddle.to_tensor(5., stop_gradient=False)
                for i in range(5):
                    y = paddle.pow(x, 4.0)
                    y.backward()
                    print("{}: {}".format(i, x.grad))
                # 0: [500.]
                # 1: [1000.]
                # 2: [1500.]
                # 3: [2000.]
                # 4: [2500.]

                x.clear_grad()
                print("{}".format(x.grad))
                # 0.

                grad_tensor=paddle.to_tensor(2.)
                for i in range(5):
                    y = paddle.pow(x, 4.0)
                    y.backward(grad_tensor)
                    print("{}: {}".format(i, x.grad))
                # 0: [1000.]
                # 1: [2000.]
                # 2: [3000.]
                # 3: [4000.]
                # 4: [5000.]

        """
        if framework._non_static_mode():
            if in_profiler_mode():
                record_event = profiler.RecordEvent(
                    "Gradient Backward", profiler.TracerEventType.Backward)
                record_event.begin()
            if grad_tensor is not None:
                if framework._in_eager_mode_:
                    assert isinstance(
                        grad_tensor, core.eager.Tensor
                    ), "The type of grad_tensor must be paddle.Tensor"
                else:
                    assert isinstance(
                        grad_tensor, paddle.Tensor
                    ), "The type of grad_tensor must be paddle.Tensor"
                assert grad_tensor.shape == self.shape, \
                    "Tensor shape not match, Tensor of grad_tensor [ {} ] with shape {} mismatch Tensor [ {} ] with shape {}".format(
                    grad_tensor.name, grad_tensor.shape, self.name, self.shape)

            if framework._in_eager_mode_:
                if grad_tensor is None:
                    grad_tensor = []
                else:
                    grad_tensor = [grad_tensor]
            if _grad_scalar:
                # When using amp with Fleet DistributedStrategy, we do loss scaling implicitly.
                self = _grad_scalar.scale(self)
            if paddle.is_compiled_with_xpu() or paddle.is_compiled_with_npu(
            ) or paddle.is_compiled_with_mlu():
                # TODO(liuyuhui): Currently only for xpu. Will be removed in the future.
                scaled_loss = scale_loss(self)
                if framework._in_eager_mode_:
                    core.eager.run_backward([scaled_loss], grad_tensor,
                                            retain_graph)
                else:
                    core.dygraph_run_backward([scaled_loss], [grad_tensor],
                                              retain_graph,
                                              framework._dygraph_tracer())
            else:
                if framework._in_eager_mode_:
                    core.eager.run_backward([self], grad_tensor, retain_graph)
                else:
                    core.dygraph_run_backward([self], [grad_tensor],
                                              retain_graph,
                                              framework._dygraph_tracer())
            if in_profiler_mode():
                record_event.end()
        else:
            raise ValueError(
                "Variable.backward() is only available in DyGraph mode")

    @framework.dygraph_only
    @deprecated(
        since="2.1.0",
        level=1,
        reason=
        "Please use tensor.grad, which returns the tensor value of the gradient."
    )
    def gradient(self):
        """
        .. warning::
          This API will be deprecated in the future, it is recommended to use
          :code:`x.grad` which returns the tensor value of the gradient.

        Get the Gradient of Current Tensor.

        Returns:
            ndarray: Numpy value of the gradient of current Tensor

        Examples:
            .. code-block:: python

                import paddle

                x = paddle.to_tensor(5., stop_gradient=False)
                y = paddle.pow(x, 4.0)
                y.backward()
                print("grad of x: {}".format(x.gradient()))
                # [500.]

        """
        if framework._in_eager_mode_:
            if self.grad is None:
                return None
            if self.grad.is_selected_rows():
                return (np.array(self.grad.numpy()), np.array(self.grad.rows()))
            return self.grad.numpy()
        else:
            if self._grad_ivar() is None:
                return None

            new_ivar = self._grad_ivar()._copy_to(core.CPUPlace(), True)
            if self._grad_ivar().type == core.VarDesc.VarType.SELECTED_ROWS:
                return (np.array(
                    new_ivar.value().get_selected_rows().get_tensor()),
                        np.array(new_ivar.value().get_selected_rows().rows()))
            else:
                return np.array(new_ivar.value().get_tensor())

    @framework.dygraph_only
    def register_hook(self, hook):
        """
        Registers a backward hook for current Tensor.

        The hook will be called every time the gradient Tensor of current Tensor is computed.

        The hook should not modify the input gradient Tensor, but it can optionally return
        a new gradient Tensor which will be used in place of current Tensor's gradient.

        The hook should have the following signature:

            hook(grad) -> Tensor or None

        Args:
            hook(function): A backward hook to be registered for Tensor.grad

        Returns:
            TensorHookRemoveHelper: A helper object that can be used to remove the registered hook by calling `remove()` method.

        Examples:
            .. code-block:: python

                import paddle

                # hook function return None
                def print_hook_fn(grad):
                    print(grad)

                # hook function return Tensor
                def double_hook_fn(grad):
                    grad = grad * 2
                    return grad

                x = paddle.to_tensor([0., 1., 2., 3.], stop_gradient=False)
                y = paddle.to_tensor([4., 5., 6., 7.], stop_gradient=False)
                z = paddle.to_tensor([1., 2., 3., 4.])

                # one Tensor can register multiple hooks
                h = x.register_hook(print_hook_fn)
                x.register_hook(double_hook_fn)

                w = x + y
                # register hook by lambda function
                w.register_hook(lambda grad: grad * 2)

                o = z.matmul(w)
                o.backward()
                # print_hook_fn print content in backward
                # Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
                #        [2., 4., 6., 8.])

                print("w.grad:", w.grad) # w.grad: [1. 2. 3. 4.]
                print("x.grad:", x.grad) # x.grad: [ 4.  8. 12. 16.]
                print("y.grad:", y.grad) # y.grad: [2. 4. 6. 8.]

                # remove hook
                h.remove()
        """
        if self.stop_gradient is True:
            raise RuntimeError(
                "Cannot register hook on a tensor that stop gradient.")

        hook_id = self._register_grad_hook(hook)
        helper = TensorHookRemoveHelper(self, hook_id)
        return helper

    @framework.dygraph_only
    def _to(self, device=None, dtype=None, blocking=None):

        if device is None and dtype is None and blocking is None:
            return self

        if device is not None:
            if isinstance(device, str):
                device = paddle.device._convert_to_place(device)
            elif isinstance(
                    device,
                (core.CPUPlace, core.CUDAPlace, core.CUDAPinnedPlace,
                 core.XPUPlace, core.CustomPlace)):
                pass
            else:
                raise ValueError(
                    "device value error, must be str, paddle.CPUPlace(), paddle.CUDAPlace(), paddle.CUDAPinnedPlace(), paddle.XPUPlace() or paddle.CustomPlace(), but the type of device is "
                    + type(device).__name__)

        if blocking is None:
            blocking = True
        else:
            assert isinstance(
                blocking,
                bool), "blocking value error, must be the True, False or None"

        def transform(t, device, dtype, blocking):
            if device is None:
                device = t.place
            if dtype is None:
                dtype = t.dtype
            if type(dtype) is str:
                dtype = framework.convert_np_dtype_to_dtype_(dtype)

            # 1. gpu place need to determine whether the memory is sufficient for allocation.
            if t.place.is_gpu_place():
                size_dtype = core.size_of_dtype(dtype)
                # Note(weilong wu): Paddle GPU minimum memory allocation unit is 256 bytes,
                # waiting_alloc_memory will compute the memory space occupied by 't'.
                # Coefficient 1.2 is used to avoid OOM that may occur in this critical state when the memory is just enough.
                waiting_alloc_memory = (
                    (t._numel() * size_dtype) / 256 + 1) * 256 * 1.2
                gpu_memory_available = core.gpu_memory_available()
                if gpu_memory_available < waiting_alloc_memory:
                    # Copy Tensor to cpu
                    t_used = t._copy_to(paddle.CPUPlace(), blocking)
                    # Release memory of t
                    t._clear()
                else:
                    # Tensor still in GPU
                    t_used = t
            else:
                t_used = t

            # 2. cast Tensor to dtype
            if dtype is not None and dtype != t_used.dtype:
                with paddle.fluid.framework._dygraph_place_guard(
                        place=t_used.place):
                    t_casted = t_used.cast(dtype=dtype)
            else:
                t_casted = t_used

            # 3. Copy casted Tensor(in CPU or GPU) to device
            if device is not None and not t_casted.place._equals(device):
                new_t = t_casted._copy_to(device, blocking)
            else:
                new_t = t_casted

            # 4. Share Tensor to origin Tensor
            dst_tensor = t.value().get_tensor()
            src_tensor = new_t.value().get_tensor()
            dst_tensor._share_data_with(src_tensor)

            return t

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return transform(self, device, dtype, blocking)

    @property
    def grad(self):
        """
        .. warning::
          This API will return the tensor value of the gradient. If you want
          to get the numpy value of the gradient, you can use :code:`x.grad.numpy()`.

        Get the Gradient of Current Tensor.

        Returns:
            Tensor: the gradient of current Tensor

        Examples:
            .. code-block:: python

                import paddle

                x = paddle.to_tensor(5., stop_gradient=False)
                y = paddle.pow(x, 4.0)
                y.backward()
                print("grad of x: {}".format(x.grad))
                # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False, [500.])

        """
        msg = 'tensor.grad will return the tensor value of the gradient.' \
            ' This is an incompatible upgrade for tensor.grad API. ' \
            ' It\'s return type changes from numpy.ndarray in version 2.0 to paddle.Tensor in version 2.1.0. ' \
            ' If you want to get the numpy value of the gradient, you can use :code:`x.grad.numpy()`'
        warning_msg = "\033[93m\nWarning:\n%s \033[0m" % (msg)
        # ensure ANSI escape sequences print correctly in cmd and powershell
        if sys.platform.lower() == 'win32':
            warning_msg = "\nWarning:\n%s " % (msg)
        warnings.warn(warning_msg)
        return self._grad_ivar()

    def clear_grad(self):
        """
        The alias of clear_gradient().
        """
        self.clear_gradient()

    def item(self, *args):
        """
        Convert element at specific position in Tensor into Python scalars. If the position is not specified, the Tensor must be a
        single-element Tensor.

        Args:
            *args(int): The input coordinates. If it's single int, the data in the corresponding order of flattened Tensor will be returned.
                Default: None, and it must be in the case where Tensor has only one element.

        Returns(Python scalar): A Python scalar, whose dtype is corresponds to the dtype of Tensor.

        Raises:
            ValueError: If the Tensor has more than one element, there must be coordinates.

        Examples:
            .. code-block:: python

                import paddle

                x = paddle.to_tensor(1)
                print(x.item())             #1
                print(type(x.item()))       #<class 'int'>

                x = paddle.to_tensor(1.0)
                print(x.item())             #1.0
                print(type(x.item()))       #<class 'float'>

                x = paddle.to_tensor(True)
                print(x.item())             #True
                print(type(x.item()))       #<class 'bool'>

                x = paddle.to_tensor(1+1j)
                print(x.item())             #(1+1j)
                print(type(x.item()))       #<class 'complex'>

                x = paddle.to_tensor([[1.1, 2.2, 3.3]])
                print(x.item(2))            #3.3
                print(x.item(0, 2))         #3.3

        """
        return self._getitem_from_offset(*args).item()

    @property
    def inplace_version(self):
        """
        The inplace version of current Tensor.
        The version number is incremented whenever the current Tensor is modified through an inplace operation.

        **Notes: This is a read-only property**

        Examples:
          .. code-block:: python

            import paddle
            var = paddle.ones(shape=[4, 2, 3], dtype="float32")
            print(var.inplace_version)  # 0

            var[1] = 2.2
            print(var.inplace_version)  # 1

        """
        return self._inplace_version()

    def __str__(self):
        """
        Convert a VarBase object to a readable string.

        Returns(str): A readable string.

        Examples:
            .. code-block:: python

                import paddle
                x = paddle.rand([2, 5])
                print(x)

                # Tensor(shape=[2, 5], dtype=float32, place=CPUPlace,
                #        [[0.30574632, 0.55739117, 0.30902600, 0.39413780, 0.44830436],
                #         [0.79010487, 0.53972793, 0.09495186, 0.44267157, 0.72112119]])
        """
        if framework._in_eager_mode_:
            from paddle.tensor.to_string import tensor_to_string
            return tensor_to_string(self)
        else:
            from paddle.tensor.to_string import to_string
            return to_string(self)

    def __deepcopy__(self, memo):
        """
        Deep copy Tensor, it will always performs Tensor copy.

        Examples:
            .. code-block:: python

                import paddle
                import copy
                x = paddle.to_tensor(2.)
                y = copy.deepcopy(x)

                print(x)
                # Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True,
                #        [2.])

                print(y)
                # Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True,
                #        [2.])

        """
        if not self.is_leaf:
            raise RuntimeError(
                "Only Leaf Tensor support the deepcopy at the moment, non-Leaf Tensors contains graph information that does't support deepcopy"
            )
        if framework._in_eager_mode_:
            new_varbase = core.eager.Tensor()
        else:
            new_varbase = core.VarBase()
        new_varbase.name = self.name + unique_name.generate("_deepcopy")
        memo[id(self)] = new_varbase
        new_varbase.copy_(self, True)
        return new_varbase

    @property
    def block(self):
        return framework.default_main_program().global_block()

    def __nonzero__(self):
        numel = np.prod(self.shape)
        assert numel == 1, "When Variable is used as the condition of if/while , Variable can only contain one element."
        if framework._in_eager_mode_:
            assert self._is_initialized(), "tensor not initialized"
            return bool(np.all(self.numpy() > 0))
        else:
            tensor = self.value().get_tensor()
            assert tensor._is_initialized(), "tensor not initialized"
            return bool(np.all(tensor.__array__() > 0))

    def __bool__(self):
        return self.__nonzero__()

    def __array__(self, dtype=None):
        """
        Returns a numpy array shows the value of current Tensor.

        Returns:
            ndarray: The numpy value of current Tensor.

        Returns type:
            ndarray: dtype is same as current Tensor

        Examples:
            .. code-block:: python

                import paddle
                import numpy as np
                x = paddle.randn([2, 2])
                x_array = np.array(x)

                print(type(x_array))      #<class 'numpy.ndarray'>
                print(x_array.shape)      #(2, 2)
        """
        array = self.numpy()
        if dtype:
            array = array.astype(dtype)
        return array

    def contain_tensor(item):
        if not isinstance(item, (tuple, list)):
            item = [item]

        for slice_item in item:
            if isinstance(slice_item, slice):
                if isinstance(slice_item.start, Variable)  \
                    or isinstance(slice_item.stop, Variable) \
                        or isinstance(slice_item.step, Variable):
                    return True
            else:
                if isinstance(
                        slice_item,
                    (Variable, np.ndarray)) and Variable.dtype != paddle.bool:
                    return True
        return False

    def __getitem__(self, item):

        def is_list_tuple(index, contain_type):

            def _is_list_tuple(item):
                if isinstance(item, (tuple, list)):
                    for s in item:
                        if not _is_list_tuple(s):
                            return False
                else:
                    if type(item) != contain_type:
                        return False
                return True

            if not isinstance(index, (tuple, list)):
                return False
            for s in index:
                if not _is_list_tuple(s):
                    return False
            return True

        if contain_tensor(item) or is_list_tuple(item, int):
            # 1. Call _getitem_impl_ when item contains tensor.
            # Why not call a c++ function ? Because item can't be parsed when it contains tensor.
            return _getitem_impl_(self, item)

        else:
            # 2. Call c++ func getitem_index_not_tensor to speedup.
            return self._getitem_index_not_tensor(item)

    def __setitem__(self, item, value):

        def contain_tensor_or_list(item):
            if not isinstance(item, tuple):
                item = [item]

            for slice_item in item:
                if isinstance(slice_item, list):
                    return True
                elif isinstance(slice_item, Variable):
                    return True

            return False

        def is_combine_index(item):
            var_type = None
            item_type = None
            if isinstance(item, (tuple, list)):
                for slice_item in item:
                    if item_type is None:
                        item_type = type(slice_item)
                    else:
                        if type(slice_item) != item_type:
                            return True

                    if isinstance(slice_item, Variable):
                        if var_type is None:
                            var_type = slice_item.dtype
                        else:
                            if var_type != slice_item.dtype:
                                return True
                return False

            return False

        if contain_tensor_or_list(item) and not is_combine_index(item):
            # To reuse code with static graph,
            # Call _setitem_impl_ when item contains tensor or list.
            return _setitem_impl_(self, item, value)

        else:
            if framework._in_eager_mode_:
                return self.__setitem_eager_tensor__(item, value)
            else:
                # Call c++ func __setitem_varbase__ to speedup.
                return self.__setitem_varbase__(item, value)

    @framework.dygraph_only
    def _grad_ivar(self):
        if self.grad is not None:
            if self.grad._is_initialized():
                return self.grad
        return None

    @framework.dygraph_only
    def _set_grad_ivar(self, value):
        if isinstance(self, EagerParamBase):
            self.grad = value
            self._unset_fake_empty()
        else:
            raise TypeError(
                "_set_grad_ivar is only supported for Parameter Tensor")

    @framework.dygraph_only
    def value(self):
        return self

    @framework.dygraph_only
    def _slice(self, begin_idx, end_idx):
        return core.eager.Tensor(self.get_tensor()._slice(begin_idx, end_idx))

    @framework.dygraph_only
    def _numel(self):
        return self.get_tensor()._numel()

    @framework.dygraph_only
    def _clear_data(self):
        self.get_tensor()._clear()

    @framework.dygraph_only
    def _uva(self, device_id=0):
        '''
        Returns self tensor with the UVA(unified virtual addressing).

        Args:
            device_id(int, optional): The destination GPU device id. Default: None, means current device.

        Examples:
            .. code-block:: python

              # required: gpu
              import paddle
              x = paddle.to_tensor([1, 2, 3], place=paddle.CPUPlace())
              x._uva()
              print(x)
        '''
        self._tensor_uva(device_id)

    @framework.dygraph_only
    def cpu(self):
        if self.place.is_cpu_place():
            return self
        else:
            res = self._copy_to(core.CPUPlace(), True)
            res.stop_gradient = self.stop_gradient
            res.persistable = self.persistable
            return res

    @framework.dygraph_only
    def cuda(self, device_id=None, blocking=True):
        if device_id is None:
            res_place = framework._current_expected_place()
            if not isinstance(res_place, core.CUDAPlace):
                res_place = core.CUDAPlace(0)
        elif isinstance(device_id, int):
            res_place = core.CUDAPlace(device_id)
        else:
            raise ValueError("device_id must be int|None")

        if self.place._equals(res_place):
            return self
        else:
            res = self._copy_to(res_place, True)
            res.stop_gradient = self.stop_gradient
            res.persistable = self.persistable
            return res

    @framework.dygraph_only
    def pin_memory(self):
        if self.place.is_cuda_pinned_place():
            return self
        else:
            res = self._copy_to(core.CUDAPinnedPlace(), True)
            res.stop_gradient = self.stop_gradient
            res.persistable = self.persistable
            return res

    @framework.dygraph_only
    def values(self):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**
        Get the values of current SparseTensor(COO or CSR).

        Returns:
            Tensor: A DenseTensor

        Examples:
            .. code-block:: python

                import paddle
                from paddle.fluid.framework import _test_eager_guard
                with _test_eager_guard():
                    indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
                    values = [1, 2, 3, 4, 5]
                    dense_shape = [3, 4]
                    sparse_x = paddle.incubate.sparse.sparse_coo_tensor(paddle.to_tensor(indices, dtype='int32'), paddle.to_tensor(values, dtype='float32'), shape=dense_shape)
                    print(sparse_x.values())
                    #[1, 2, 3, 4, 5]
        """
        return _C_ops.sparse_values(self)

    @framework.dygraph_only
    def to_dense(self):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**
        Convert the current SparseTensor(COO or CSR) to DenseTensor.

        Returns:
            Tensor: A DenseTensor

        Examples:
            .. code-block:: python

                import paddle
                from paddle.fluid.framework import _test_eager_guard
                with _test_eager_guard():
                    indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
                    values = [1, 2, 3, 4, 5]
                    dense_shape = [3, 4]
                    sparse_x = paddle.incubate.sparse.sparse_coo_tensor(paddle.to_tensor(indices, dtype='int64'), paddle.to_tensor(values, dtype='float32'), shape=dense_shape)
                    dense_x = sparse_x.to_dense()
                    #[[0., 1., 0., 2.],
                    # [0., 0., 3., 0.],
                    # [4., 5., 0., 0.]]
        """

        return _C_ops.sparse_to_dense(self)

    @framework.dygraph_only
    def to_sparse_coo(self, sparse_dim):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**
        Convert the current DenseTensor to SparseTensor in COO format.

        Returns:
            Tensor: A SparseCooTensor

        Examples:
            .. code-block:: python

                import paddle
                from paddle.fluid.framework import _test_eager_guard
                with _test_eager_guard():
                    dense_x = [[0, 1, 0, 2], [0, 0, 3, 4]]
                    dense_x = paddle.to_tensor(dense_x, dtype='float32')
                    sparse_x = dense_x.to_sparse_coo(sparse_dim=2)
                    #indices=[[0, 0, 1, 1],
                    #         [1, 3, 2, 3]],
                    #values=[1., 2., 3., 4.]
        """

        return _C_ops.sparse_to_sparse_coo(self, sparse_dim)

    if framework._in_eager_mode_ and not hasattr(core, "eager"):
        return

    for method_name, method in (("__bool__", __bool__), ("__nonzero__",
                                                         __nonzero__),
                                ("_to_static_var",
                                 _to_static_var), ("set_value", set_value),
                                ("block", block), ("backward", backward),
                                ("clear_grad", clear_grad), ("inplace_version",
                                                             inplace_version),
                                ("gradient", gradient), ("register_hook",
                                                         register_hook),
                                ("__str__", __str__), ("__repr__", __str__),
                                ("__deepcopy__", __deepcopy__), ("__module__",
                                                                 "paddle"),
                                ("__array__",
                                 __array__), ("__getitem__",
                                              __getitem__), ("item", item),
                                ("__setitem__",
                                 __setitem__), ("_to", _to), ("values", values),
                                ("to_dense", to_dense), ("to_sparse_coo",
                                                         to_sparse_coo)):
        if framework._in_eager_mode_:
            setattr(core.eager.Tensor, method_name, method)
        else:
            setattr(core.VarBase, method_name, method)

    if framework._in_eager_mode_:
        setattr(core.eager.Tensor, "_grad_ivar", _grad_ivar)
        setattr(core.eager.Tensor, "_set_grad_ivar", _set_grad_ivar)
        setattr(core.eager.Tensor, "value", value)
        setattr(core.eager.Tensor, "cpu", cpu)
        setattr(core.eager.Tensor, "cuda", cuda)
        setattr(core.eager.Tensor, "pin_memory", pin_memory)
        setattr(core.eager.Tensor, "_slice", _slice)
        setattr(core.eager.Tensor, "_numel", _numel)
        setattr(core.eager.Tensor, "_uva", _uva)
        setattr(core.eager.Tensor, "_clear_data", _clear_data)
    else:
        setattr(core.VarBase, "__name__", "Tensor")
        setattr(core.VarBase, "grad", grad)

    global _already_patch_repr
    if not _already_patch_repr:
        # NOTE(zhiqiu): pybind11 will set a default __str__ method of enum class.
        # So, we need to overwrite it to a more readable one.
        # See details in https://github.com/pybind/pybind11/issues/2537.
        origin = getattr(core.VarDesc.VarType, "__repr__")

        def dtype_str(dtype):
            if dtype in _PADDLE_DTYPE_2_NUMPY_DTYPE:
                numpy_dtype = _PADDLE_DTYPE_2_NUMPY_DTYPE[dtype]
                if numpy_dtype == 'uint16':
                    numpy_dtype = 'bfloat16'
                prefix = 'paddle.'
                return prefix + numpy_dtype
            else:
                # for example, paddle.fluid.core.VarDesc.VarType.LOD_TENSOR
                return origin(dtype)

        setattr(core.VarDesc.VarType, "__repr__", dtype_str)
        _already_patch_repr = True

    # patch math methods for varbase
    monkey_patch_math_varbase()
