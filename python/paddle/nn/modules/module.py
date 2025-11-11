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

# Note:
# This section primarily addresses compatibility issues involving differing paths, API aliases, or parameter aliases.
# Avoid adding unnecessary code here. Only introduce new module-related aliases and paths.
# New class methods should be added to `paddle.nn.Layer`.

from __future__ import annotations

import warnings
from collections import namedtuple
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    overload,
)

from typing_extensions import Self

import paddle
from paddle import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from paddle.nn.parameter import Parameter

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
        return super().get_sublayer(target=target)

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

        super().set_sublayer(target, module, strict)

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
        Returns a iterator of all buffers from current module and its sub-modules.

        Parameters:
            recurse(bool, optional): Whether include the buffers of submodules. If True, also include the buffers from submodules. Default: True.

        Returns:
            list of Tensor, a list of buffers.
        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

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
