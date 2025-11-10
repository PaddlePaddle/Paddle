# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import warnings
from collections import namedtuple
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    overload,
)

import numpy as np
from typing_extensions import Self

import paddle
from paddle import Tensor, nn
from paddle.base import framework

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from paddle._typing import PlaceLike
    from paddle.nn.parameter import Parameter

from paddle import dtype
from paddle.nn.layer.layers import (
    _convert_camel_to_snake,
    _scope_dist2single,
)


class _IncompatibleKeys(
    # pyrefly: ignore [invalid-inheritance]
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"]),
):
    __slots__ = ()

    def __repr__(self) -> str:
        # pyrefly: ignore [missing-attribute]
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__


class Module(nn.Layer):
    """
    Base class for all neural network modules.

    Example:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> class MyModel(paddle.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.conv1 = paddle.nn.Conv2D(1, 20, 5)
            ...         self.conv2 = paddle.nn.Conv2D(20, 20, 5)
            ...
            ...     def forward(self, x):
            ...         x = F.relu(self.conv1(x))
            ...         return F.relu(self.conv2(x))

            >>> model = MyModel()
            >>> x = paddle.randn([1, 1, 28, 28])
            >>> out = model(x)
            >>> print(out.shape)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        name_scope = kwargs.pop('name_scope', None)
        dtype = kwargs.pop('dtype', 'float32')

        if name_scope is None:
            name_scope = _convert_camel_to_snake(self.__class__.__name__)
            name_scope = _scope_dist2single(name_scope)

        super().__init__(name_scope, dtype)

    @property
    def _modules(self):
        return self._sub_layers

    @_modules.setter
    def _modules(self, value):
        if not isinstance(value, dict):
            raise TypeError(f"_modules must be dict-like, got {type(value)}")
        self._sub_layers.clear()
        self._sub_layers.update(value)

    @property
    def _non_persistent_buffers_set(self):
        return self._non_persistable_buffer_names_set

    @_non_persistent_buffers_set.setter
    def _non_persistent_buffers_set(self, value):
        if not isinstance(value, set):
            raise TypeError(
                f"_non_persistent_buffers_set must be a set, got {type(value)}"
            )
        self._non_persistable_buffer_names_set.clear()
        self._non_persistable_buffer_names_set.update(value)

    def register_buffer(
        self, name: str, tensor: Tensor | None, persistent: bool = True
    ) -> None:
        """
        Registers a tensor as buffer into the module.

        `buffer` is a non-trainable tensor and will not be updated by optimizer,
        but is necessary for evaluation and inference. For example, the mean and variance in BatchNorm layers.
        The registered buffer is persistent by default, and will be saved into
        `state_dict` alongside parameters. If set persistent=False, it registers
        a non-persistent buffer, so that it will not be a part of `state_dict` .

        Buffers can be accessed as attributes using given names.

        Parameters:
            name (string): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Optional[Tensor]): the tensor to be registered as buffer.
            persistent (bool): whether the buffer is part of this module's
                state_dict.

        Returns:
            None
        """
        super().register_buffer(name, tensor, persistable=persistent)

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        """
        Adds a Parameter instance. Added parameter can be accessed by self.name

        Parameters:
            name(str): name of this submodule.
            parameter(Optional[Parameter]): an instance of Parameter.
        Returns:
            None
        """
        super().add_parameter(name, param)

    def add_module(self, name: str, module: Module | None) -> None:
        """
        Adds a sub module instance. Added module can be accessed by self.name

        Parameters:
            name(str): name of this submodule.
            module(Module): an instance of Module.
        Returns:
            None
        """
        super().add_sublayer(name, module)

    def register_module(self, name: str, module: Module | None) -> None:
        """Alias for :func:`add_module`."""
        self.add_module(name, module)

    def get_submodule(self, target: str) -> Module:
        """
        Return the submodule given by ``target`` if it exists, otherwise throw an error.

        Parameters:
            target(str): The fully-qualified string name of the submodule to look for.

        Returns:
            Module: The submodule referenced by ``target``.
        """
        if target == "":
            return self

        atoms: list[str] = target.split(".")
        # mod: paddle.nn.Module = self
        mod: paddle.nn.Layer = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )

            mod = getattr(mod, item)

            # if not isinstance(mod, paddle.nn.Module):
            if not isinstance(mod, (paddle.nn.Module, paddle.nn.Layer)):
                raise AttributeError("`" + item + "` is not an nn.Module")

        return mod

    def set_submodule(
        self, target: str, module: Module, strict: bool = False
    ) -> None:
        """
        Set the submodule given by ``target`` if it exists, otherwise throw an error.

        Parameters:
            target(str): The fully-qualified string name of the submodule to look for.
            module(Module): The module to set the submodule to.
            strict(bool): If ``False``, the method will replace an existing submodule
                or create a new submodule if the parent module exists. If ``True``,
                the method will only attempt to replace an existing submodule and throw an error
                if the submodule doesn't already exist.
        """
        if target == "":
            raise ValueError("Cannot set the submodule without a target name!")

        atoms: list[str] = target.split(".")
        if not isinstance(module, (paddle.nn.Module, paddle.nn.Layer)):
            raise ValueError(
                "`" + "module" + f"` is not an nn.Module, found {type(module)}"
            )
        if len(atoms) == 1:
            parent: paddle.nn.Module = self
        else:
            parent_key = ".".join(atoms[:-1])
            parent = self.get_submodule(parent_key)

        if strict and not hasattr(parent, atoms[-1]):
            raise AttributeError(
                parent._get_name() + " has no attribute `" + atoms[-1] + "`"
            )
        if hasattr(parent, atoms[-1]):
            mod = getattr(parent, atoms[-1])
            if not isinstance(mod, (paddle.nn.Module, paddle.nn.Layer)):
                raise AttributeError("`" + atoms[-1] + "` is not an nn.Module")
        setattr(parent, atoms[-1], module)

    def get_parameter(self, target: str) -> Parameter:
        """
        Return the parameter given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Parameters:
            target(str): The fully-qualified string name of the Parameter to look for.

        Returns:
            Parameter: The Parameter referenced by ``target``.
        """
        module_path, _, param_name = target.rpartition(".")

        mod: paddle.nn.Module = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + param_name + "`"
            )

        param: paddle.nn.Parameter = getattr(mod, param_name)

        if not isinstance(param, (paddle.nn.Parameter, paddle.Tensor)):
            raise AttributeError("`" + param_name + "` is not an nn.Parameter")

        return param

    def get_buffer(self, target: str) -> Tensor:
        """
        Return the buffer given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Parameters:
            target(str): The fully-qualified string name of the buffer to look for.

        Returns:
            Tensor: The buffer referenced by ``target``.
        """
        module_path, _, buffer_name = target.rpartition(".")

        mod = self.get_submodule(module_path)

        if not hasattr(mod, buffer_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + buffer_name + "`"
            )

        buffer = getattr(mod, buffer_name)

        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        return buffer

    def get_extra_state(self) -> Any:
        raise RuntimeError(
            "Reached a code path in Module.get_extra_state() that should never be called. "
        )

    def cuda(self, device: int | PlaceLike | None = None) -> Self:
        """
        Move all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing the optimizer if the module will
        live on GPU while being optimized.

        Parameters:
            device(int, optional): if specified, all parameters will be copied to that device.

        Returns:
            Module: self
        """
        if device is None:
            device = paddle.CUDAPlace(paddle.cuda.current_device())
        elif isinstance(device, int):
            device = paddle.CUDAPlace(device)
        elif isinstance(device, paddle.CUDAPlace):
            pass
        else:
            raise TypeError(
                f"device must be int, paddle.CUDAPlace or None, got {type(device)}"
            )

        return self._to_impl(device=device)

    def xpu(self, device: int | PlaceLike | None = None) -> Self:
        """
        Move all model parameters and buffers to the XPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on XPU while being optimized.

        Parameters:
            device(int, optional): if specified, all parameters will be copied to that device.

        Returns:
            Module: self
        """
        if device is None:
            device = paddle.XPUPlace(0)
        elif isinstance(device, int):
            device = paddle.XPUPlace(device)
        elif isinstance(device, paddle.XPUPlace):
            pass
        else:
            raise TypeError(
                f"device must be int, paddle.XPUPlace or None, got {type(device)}"
            )

        return self._to_impl(device=device)

    def cpu(self) -> Self:
        """
        Move all model parameters and buffers to the CPU.

        Returns:
            Module: self
        """
        return self._to_impl(device=paddle.CPUPlace())

    def type(self, dst_type: dtype | str) -> Self:
        """
        Casts all parameters and buffers to :attr:`dst_type`.

        Parameters:
            dtype(str|paddle.dtype): target data type of layer.
                If set str, it can be "bool", "bfloat16", "float16", "float32", "float64",
                "int8", "int16", "int32", "int64", "uint8", "complex64", "complex128".
                Default: None

        Returns:
            Module: self
        """
        valid_dtypes = [
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "complex64",
            "complex128",
            "bool",
        ]
        if (
            isinstance(dst_type, (paddle.dtype, np.dtype))
            or type(dst_type) is str
            and dst_type in valid_dtypes
        ):
            if isinstance(dst_type, (str, np.dtype)):
                dst_type = framework.convert_np_dtype_to_dtype_(dst_type)

            def layer_trans(layer):
                layer._to_impl(
                    dtype=dst_type, floating_only=False, include_sublayers=True
                )

            return self.apply(layer_trans)
        else:
            raise ValueError(
                "dtype value error, must be 'bfloat16', 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'complex64', 'complex128', 'bool', or paddle.dtype, numpy.dtype, but receive "
                + str(dtype)
            )

    def double(self) -> Self:
        """
        Casts all floating point parameters and buffers to ``double`` datatype.

        Returns:
            Module: self
        """
        return self.type(paddle.float64)

    def half(self) -> Self:
        """
        Casts all floating point parameters and buffers to ``half`` datatype.

        Returns:
            Module: self
        """
        return self.type(paddle.float16)

    def bfloat16(self) -> Self:
        """
        Casts all floating point parameters and buffers to ``bfloat16`` datatype.

        Returns:
            Module: self
        """
        return self.type(paddle.bfloat16)

    T_destination = TypeVar("T_destination", bound=dict[str, Any])

    @overload
    def state_dict(
        self,
        *,
        destination: T_destination,
        prefix: str = ...,
        keep_vars: bool = ...,
    ) -> T_destination: ...

    @overload
    def state_dict(
        self,
        *,
        prefix: str = ...,
        keep_vars: bool = ...,
    ) -> dict[str, Any]: ...

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        '''
        Get all parameters and persistable buffers of current module and its sub-module. And set them into a dict

        Parameters:
            destination(dict, optional) : If provide, all the parameters and persistable buffers will be set to this dict . Default: None.
            prefix (str, optional): a prefix added to parameter and buffer names to compose the keys in state_dict. Default: ``''``.
            keep_vars(bool, optional) : If false, the returned tensors in the state dict are detached from autograd. Default: True.

        Returns:
            dict: a dict contains all the parameters and persistable buffers.

        '''
        if len(args) > 0:
            warnings.warn(
                "Positional args are deprecated, use kwargs instead.",
                FutureWarning,
                stacklevel=2,
            )
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == "":
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]

        return super().state_dict(
            destination=destination,
            include_sublayers=True,
            structured_name_prefix=prefix,
            use_hook=True,
            keep_vars=keep_vars,
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        """
        Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

        If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.


        Parameters:
            state_dict (dict): a dict containing parameters and persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
            assign (bool, optional): When set to ``False``, the properties of the tensors
                in the current module are preserved whereas setting it to ``True`` preserves
                properties of the Tensors in the state dict. The only
                exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`
                for which the value from the module is preserved. Default: ``False``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * ``missing_keys`` is a list of str containing any keys that are expected
                    by this module but missing from the provided ``state_dict``.
                * ``unexpected_keys`` is a list of str containing the keys that are not
                    expected by this module but present in the provided ``state_dict``.
        """
        error_msgs: list[str] = []

        missing_keys, unexpected_keys = super().set_state_dict(
            state_dict, use_structured_name=True
        )

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in missing_keys)
                    ),
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """

        Returns a list of all Parameters from current module and its sub-modules.
        Parameters:
            recurse (bool, optional): Whether to return the parameters of the submodule.
                If True, the returned list contains the parameters of the submodule.
                Default: True.

        Returns:
            list, list of Tensor, a list of Parameters.
        """
        return super().parameters(include_sublayers=recurse)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
        **kwargs,
    ) -> Iterator[tuple[str, Parameter]]:
        """
        Returns an iterator over all parameters in the module, yielding tuple of name and parameter.

        Parameters:
            prefix(str, optional): Prefix to prepend to all parameter names. Default: ''.
            recurse(bool, optional): Whether include the parameters of submodules.
                If True, also include the named parameters from submodules. Default: True.
            remove_duplicate(bool, optional): Whether to remove duplicated parameters in the result.
                Default: True.

        Yields:
            (string, Parameter): Tuple of name and Parameter
        """
        include_sublayers = kwargs.pop("include_sublayers", recurse)
        return super().named_parameters(
            prefix=prefix,
            include_sublayers=include_sublayers,
            remove_duplicate=remove_duplicate,
        )

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """
        Returns a list of all buffers from current module and its sub-modules.

        Parameters:
            recurse(bool, optional): Whether include the buffers of submodules. If True, also include the buffers from submodules. Default: True.

        Returns:
            list of Tensor, a list of buffers.
        """
        return super().buffers(include_sublayers=recurse)

    def named_buffers(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Tensor]]:
        """
        Returns an iterator over all buffers in the module, yielding tuple of name and Tensor.

        Parameters:
            prefix(str, optional): Prefix to prepend to all buffer names. Default: ''.
            recurse(bool, optional): Whether include the buffers of submodules.
                If True, also include the named buffers from submodules. Default: True.
            remove_duplicate(bool, optional): Whether to remove duplicated buffers in the result.
                Default: True.

        Yields:
            (string, Tensor): Tuple of name and tensor
        """
        return super().named_buffers(
            prefix=prefix,
            include_sublayers=recurse,
            remove_duplicate=remove_duplicate,
        )

    def modules(self) -> Iterator[Module]:
        """
        Return an iterator over all modules in the network.

        Yields:
            Module: a module in the network.

        """
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: set[Module] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        """
        Returns an iterator over all submodules in the Module, yielding tuple of name and submodule.
        The duplicate submodule will only be yielded once.

        Parameters:
            memo(set, optional): The set to record duplicate submodules. Default: None.
            prefix(str, optional): Prefix to prepend to all parameter names. Default: ''.
            remove_duplicate(bool, optional): Whether to remove duplicated submodules in the result.
                Default: True.

        Yields:
            (string, Module): Tuple of name and Module
        """
        include_self = True
        layers_set = memo
        return super().named_sublayers(
            prefix=prefix,
            include_self=include_self,
            layers_set=layers_set,
            remove_duplicate=remove_duplicate,
        )

    def train(self, mode: bool = True) -> Self:
        """
        Sets this Module and all its submodules to training mode.
        This only effects certain modules like `Dropout` and `BatchNorm`.

        Returns:
            Module: self
        """
        return super().train(mode=mode)

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """
        Change if autograd should record operations on parameters in this module.

        Parameters:
            requires_grad (bool): whether autograd should record operations on
                                  parameters in this module. Default: ``True``.

        Returns:
            Module: self
        """
        for p in self.parameters():
            p.stop_gradient = not requires_grad
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Reset gradients of all model parameters.

        Parameters:
            set_to_none (bool): instead of setting to zero, set the grads to None. Currently, set_to_none=True
            is not fully supported.
        """
        if getattr(self, "_is_replica", False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead.",
                stacklevel=2,
            )
        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.clear_gradient(set_to_zero=False)
                else:
                    p.clear_gradient(set_to_zero=True)

    def _get_name(self):
        return self.__class__.__name__
