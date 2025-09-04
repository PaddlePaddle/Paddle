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

import functools
import inspect
import warnings
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from collections.abc import Iterable

_InputT = ParamSpec("_InputT")
_RetT = TypeVar("_RetT")


class DecoratorBase:
    """Decorative base class, providing a universal decorative framework.

    Subclass only needs to implement the 'process' method to define the core logic.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(
        self, func: Callable[_InputT, _RetT]
    ) -> Callable[_InputT, _RetT]:
        """As an entry point for decorative applications"""

        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            # Pretreatment parameters
            processed_args, processed_kwargs = self.process(args, kwargs)
            return func(*processed_args, **processed_kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return cast("Callable[_InputT, _RetT]", wrapper)

    def process(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """To be implemented by subclass"""
        raise NotImplementedError("Subclasses must implement this method")


# Example implementation: Parameter alias decorator
class ParamAliasDecorator(DecoratorBase):
    """Implementation of Decorator for Parameter Alias Processing"""

    def __init__(self, alias_mapping: dict[str, Iterable[str]]) -> None:
        super().__init__()
        # Check alias_mapping types
        if not isinstance(alias_mapping, dict):
            raise TypeError("alias_mapping must be a dictionary")
        for k, v in alias_mapping.items():
            if not isinstance(v, (list, tuple, set)):
                raise TypeError(f"Aliases for '{k}' must be iterable")

        # Build a reverse alias map for faster lookup
        self.alias_mapping = {}
        for original, aliases in alias_mapping.items():
            for alias in aliases:
                self.alias_mapping[alias] = original

    def process(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Process parameters to handle alias mapping"""
        if not kwargs:
            return args, kwargs

        processed_kwargs = kwargs
        alias_mapping = self.alias_mapping

        # Directly modify kwargs based on alias mapping (only modify if necessary)
        for alias, original in alias_mapping.items():
            if alias in processed_kwargs:
                if original not in processed_kwargs:
                    # Only modify the dictionary if necessary
                    processed_kwargs[original] = processed_kwargs.pop(alias)
                else:
                    raise ValueError(
                        f"Cannot specify both '{original}' and its alias '{alias}'"
                    )

        return args, processed_kwargs


class SetDefaultParaAliasDecorator(DecoratorBase):
    """Support default parameter settings, implementation of parameter alias processing decorator"""

    def __init__(
        self,
        alias_mapping: dict[str, Iterable[str]],
        default_params: dict[str, Any],
    ) -> None:
        super().__init__()
        # Check alias_mapping types
        if not isinstance(alias_mapping, dict):
            raise TypeError("alias_mapping must be a dictionary")
        for k, v in alias_mapping.items():
            if not isinstance(v, (list, tuple, set)):
                raise TypeError(f"Aliases for '{k}' must be iterable")

        # Build a reverse alias map for faster lookup
        self.alias_mapping = {}
        for original, aliases in alias_mapping.items():
            for alias in aliases:
                self.alias_mapping[alias] = original

        self.default_params = default_params
        warnings.simplefilter("always", category=Warning)

    def process(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Process parameters to handle alias mapping"""
        if not kwargs:
            return args, kwargs

        is_torch_call = False

        # Directly modify kwargs based on alias mapping (only modify if necessary)
        for alias, original in self.alias_mapping.items():
            if alias in kwargs:
                if original not in kwargs:
                    kwargs[original] = kwargs.pop(alias)
                    is_torch_call = True
                else:
                    raise ValueError(
                        f"Cannot specify both '{original}' and its alias '{alias}'"
                    )

        if is_torch_call:
            warnings.warn(
                "Set default parameters " + str(self.default_params),
                category=Warning,
            )
            for key, value in self.default_params.items():
                if key not in kwargs:
                    kwargs[key] = value

        return args, kwargs


def softmax_param_alias(
    func: Callable[_InputT, _RetT],
) -> Callable[_InputT, _RetT]:
    @functools.wraps(func)
    def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        # Process parameters to handle alias mapping
        if "input" in kwargs:
            kwargs["x"] = kwargs.pop("input")
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")
        return func(*args, **kwargs)

    wrapper.__signature__ = inspect.signature(func)
    return cast("Callable[_InputT, _RetT]", wrapper)


def param_one_alias(alias_list):
    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            if not kwargs:
                return func(*args, **kwargs)
            if (alias_list[0] not in kwargs) and (alias_list[1] in kwargs):
                kwargs[alias_list[0]] = kwargs.pop(alias_list[1])
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def param_two_alias(alias_list1, alias_list2):
    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            if not kwargs:
                return func(*args, **kwargs)
            if (alias_list1[0] not in kwargs) and (alias_list1[1] in kwargs):
                kwargs[alias_list1[0]] = kwargs.pop(alias_list1[1])
            if (alias_list2[0] not in kwargs) and (alias_list2[1] in kwargs):
                kwargs[alias_list2[0]] = kwargs.pop(alias_list2[1])
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def param_two_alias_one_default(alias_list1, alias_list2, default_param):
    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            if not kwargs:
                return func(*args, **kwargs)

            is_torch_call = False

            if (alias_list1[0] not in kwargs) and (alias_list1[1] in kwargs):
                kwargs[alias_list1[0]] = kwargs.pop(alias_list1[1])
                is_torch_call = True
            if (alias_list2[0] not in kwargs) and (alias_list2[1] in kwargs):
                kwargs[alias_list2[0]] = kwargs.pop(alias_list2[1])
                is_torch_call = True

            if is_torch_call:
                warnings.warn(
                    "Set default parameters " + str(default_param),
                    category=Warning,
                )
                if default_param[0] not in kwargs:
                    kwargs[default_param[0]] = default_param[1]
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


# *size => shape decorator
class SizeArgsDecorator(DecoratorBase):
    """
    Usage Example:

    paddle.ones(1, dtype=paddle.float32)
    paddle.ones(1, 2, 3, dtype=paddle.float32)
    paddle.ones([1, 2, 3], dtype=paddle.float32)
    paddle.ones(size=[1, 2, 3], dtype=paddle.float32)

    paddle.ones([1, 2, 3], paddle.float32)
    paddle.ones(shape=[1, 2, 3], dtype=paddle.float32)
    """

    def process(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if 'size' in kwargs:
            kwargs['shape'] = kwargs.pop('size')
        elif len(args) >= 1 and isinstance(args[0], int):
            kwargs['shape'] = list(args)
            args = ()

        return args, kwargs


def size_args_decorator(func: Callable) -> Callable:
    """
    A decorator that normalizes the 'size' argument to 'shape'.

    Usage Example:

    paddle.ones(1, dtype=paddle.float32)
    paddle.ones(1, 2, 3, dtype=paddle.float32)
    paddle.ones([1, 2, 3], dtype=paddle.float32)
    paddle.ones(size=[1, 2, 3], dtype=paddle.float32)
    paddle.ones([1, 2, 3], paddle.float32)
    paddle.ones(shape=[1, 2, 3], dtype=paddle.float32)
    """

    @functools.wraps(func)
    def wrapped_func(*args: Any, **kwargs: Any) -> Any:
        if 'size' in kwargs:
            kwargs['shape'] = kwargs.pop('size')
        elif len(args) >= 1 and isinstance(args[0], int):
            kwargs['shape'] = list(args)
            args = ()

        return func(*args, **kwargs)

    wrapped_func.__signature__ = inspect.signature(func)

    return wrapped_func


def size_args_decorator_patch(method: Callable) -> Callable:
    """
    A decorator that allow *size for patching method to Tensor.
    e.g. Tensor.method(*size, *, ...).

    Usage Example:

    paddle.randn([]).new_ones(1, dtype=paddle.float32)
    paddle.randn([]).new_ones(1, 2, 3, dtype=paddle.float32)
    paddle.randn([]).new_ones([1, 2, 3], dtype=paddle.float32)
    paddle.randn([]).new_ones(size=[1, 2, 3], dtype=paddle.float32)
    paddle.randn([]).new_ones([1, 2, 3], paddle.float32)
    """

    @functools.wraps(method)
    def wrapped_func(*args: Any, **kwargs: Any) -> Any:
        if len(args) >= 2 and isinstance(args[1], int):
            # args[0]: Tensor
            # args[1:]: *size
            kwargs['size'] = list(args[1:])
            args = (args[0],)

        return method(*args, **kwargs)

    wrapped_func.__signature__ = inspect.signature(method)

    return wrapped_func


class VariableArgsDecorator(DecoratorBase):
    def __init__(self, var: str) -> None:
        super().__init__()
        if not isinstance(var, str):
            raise TypeError("var must be a string")
        self.var = var

    def process(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if len(args) >= 2 and isinstance(args[1], int):
            kwargs[self.var] = list(args[1:])
            args = args[:1]
        return args, kwargs


def view_decorator():
    """
    Usage Example:
    paddle.view(x=tensor_x, shape_or_dtype=[-1, 1, 3], name=None)
    tensor_x.view(paddle.float32) -> paddle.view(tensor_x, paddle.float32)
    tensor_x.view(dtype=paddle.float32) -> paddle.view(tensor_x, dtype=paddle.float32)
    tensor_x.view([-1, 1, 3]) -> paddle.view(tensor_x, [-1, 1, 3])
    tensor_x.view(-1, 1, 3) -> paddle.view(tensor_x, -1, 1, 3)
    tensor_x.view(size=[-1, 1, 3]) -> paddle.view(tensor_x, size=[-1, 1, 3])
    """

    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            if ("dtype" in kwargs) and ("shape_or_dtype" not in kwargs):
                kwargs["shape_or_dtype"] = kwargs.pop("dtype")
            elif ("size" in kwargs) and ("shape_or_dtype" not in kwargs):
                kwargs["shape_or_dtype"] = kwargs.pop("size")
            elif len(args) >= 2 and type(args[1]) is int:
                if all(type(arg) is int for arg in args[1:]):
                    kwargs["x"] = args[0]
                    kwargs['shape_or_dtype'] = list(args[1:])
                    args = ()
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


class ForbidKeywordsDecorator(DecoratorBase):
    """A decorator that hints users to use the correct `compat` functions, when erroneous keyword arguments are detected"""

    def __init__(
        self,
        illegal_keys: set[str],
        func_name: str,
        correct_name: str,
        url_suffix: str = "",
    ) -> None:
        """
        Args:
            illegal_keys (set[str]): the keywords to reject
            func_name (str): the name of the function being decorated (should incorporate module name, like paddle.nn.Unfold)
            correct_name (str): the user hint that points to the correct function
            url_suffix (str, optional): Only specified in non paddle.compat functions. If specified, the function being decorated
                will emit a warning upon the first call, warning the users about the API difference and points to Docs.
                Please correctly specifying the `url_suffix`, this should be the suffix of the api-difference doc. For example:

                (prefix omitted)/docs/zh/develop/guides/model_convert/convert_from_pytorch/api_difference/**torch/torch.nn.Unfold**.html

                In this example, the correct `url_suffix` should be 'torch/torch.nn.Unfold'. Defaults to an empty str.
        """
        super().__init__()
        self.illegal_keys = illegal_keys
        self.func_name = func_name
        self.correct_name = correct_name
        self.warn_msg = None
        if url_suffix:
            self.warn_msg = f"\nNon compatible API. Please refer to https://www.paddlepaddle.org.cn/documentation/docs/en/develop/guides/model_convert/convert_from_pytorch/api_difference/{url_suffix}.html first."

    def process(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        found_keys = [key for key in self.illegal_keys if key in kwargs]

        if found_keys:
            found_keys.sort()
            keys_str = ", ".join(f"'{key}'" for key in found_keys)
            plural = "s" if len(found_keys) > 1 else ""

            raise TypeError(
                f"{self.func_name}() received unexpected keyword argument{plural} {keys_str}. "
                f"\nDid you mean to use {self.correct_name}() instead?"
            )
        if self.warn_msg is not None:
            warnings.warn(
                self.warn_msg,
                category=Warning,
            )
            self.warn_msg = None
        return args, kwargs


class ForbidKeywordsIgnoreOneParamDecorator(ForbidKeywordsDecorator):
    """A decorator that hints users to use the correct `compat` functions, when erroneous keyword arguments are detected and one argument is ignored"""

    def __init__(
        self,
        illegal_keys: set[str],
        ignore_param: tuple[str, int, type[Any]],
        func_name: str,
        correct_name: str,
        url_suffix: str = "",
    ) -> None:
        """
        Args:
            illegal_keys (set[str]): the keywords to reject
            ignore_param: (tuple[str, int, type[Any]]): A tuple of (parameter_name, index, type) to ignore by name, position and type
            func_name (str): the name of the function being decorated (should incorporate module name, like paddle.nn.Unfold)
            correct_name (str): the user hint that points to the correct function
            url_suffix (str, optional): Only specified in non paddle.compat functions. If specified, the function being decorated
                will emit a warning upon the first call, warning the users about the API difference and points to Docs.
                Please correctly specifying the `url_suffix`, this should be the suffix of the api-difference doc. For example:

                (prefix omitted)/docs/zh/develop/guides/model_convert/convert_from_pytorch/api_difference/**torch/torch.nn.Unfold**.html

                In this example, the correct `url_suffix` should be 'torch/torch.nn.Unfold'. Defaults to an empty str.
        """
        super().__init__(illegal_keys, func_name, correct_name, url_suffix)
        self.ignore_param = ignore_param

    def process(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args, kwargs = super().process(args, kwargs)

        if self.ignore_param:
            name, index, typ = self.ignore_param
            if index < len(args) and isinstance(args[index], typ):
                args = args[:index] + args[index + 1 :]
            else:
                kwargs.pop(name, None)

        return args, kwargs


def reshape_decorator():
    """
    Usage Example:
    paddle.reshape(x=tensor_x, shape=[-1, 1, 3], name=None)
    paddle.reshape(input=tensor_x, shape=[-1, 1, 3], name=None)
    tensor_x.reshape([-1, 1, 3]) -> paddle.reshape(tensor_x, [-1, 1, 3])
    tensor_x.reshape(-1, 1, 3) -> paddle.reshape(tensor_x, -1, 1, 3])
    """

    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            if ("input" in kwargs) and ("x" not in kwargs):
                kwargs["x"] = kwargs.pop("input")
            elif len(args) >= 2 and type(args[1]) is int:
                if all(type(arg) is int for arg in args[1:]):
                    kwargs["x"] = args[0]
                    kwargs['shape'] = list(args[1:])
                    args = ()
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def transpose_decorator():
    """
    Usage Example:
    PyTorch:
        torch.transpose(x, dim0=0, dim1=1)
    Paddle:
        paddle.transpose(x, perm=[1, 0, 2])
    """

    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            if ("input" in kwargs) and ("x" not in kwargs):
                kwargs["x"] = kwargs.pop("input")

            dim0 = kwargs.pop("dim0", kwargs.pop("axis0", None))
            dim1 = kwargs.pop("dim1", kwargs.pop("axis1", None))

            if dim0 is None and len(args) > 1 and isinstance(args[1], int):
                dim0 = args[1]
            if dim1 is None and len(args) > 2 and isinstance(args[2], int):
                dim1 = args[2]

            if dim0 is not None and dim1 is not None:
                ndim = kwargs["x"].ndim if "x" in kwargs else args[0].ndim
                perm = list(range(ndim))
                perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
                kwargs["perm"] = perm
                if len(args) > 1:
                    args = (args[0],)

            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def expand_decorator():
    """
    Usage Example:
    paddle.expand(x=tensor_x, shape=[3, 4], name=None)
    tensor_x.expand([3, 4]) -> paddle.expand(tensor_x, [3, 4])
    tensor_x.expand(3, 4) -> paddle.expand(tensor_x, 3, 4)
    tensor_x.expand(size=[3, 4]) -> paddle.expand(tensor_x, size=[3, 4])
    """

    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            if ("input" in kwargs) and ("x" not in kwargs):
                kwargs["x"] = kwargs.pop("input")
            if ("size" in kwargs) and ("shape" not in kwargs):
                kwargs["shape"] = kwargs.pop("size")
            elif len(args) >= 2 and type(args[1]) is int:
                if all(type(arg) is int for arg in args[1:]):
                    kwargs["x"] = args[0]
                    kwargs['shape'] = list(args[1:])
                    args = ()
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def index_select_decorator():
    """
    Usage Example:
    PyTorch: index_select(input, dim, index)
        torch.index_select(input=input_tensor, dim=1, index=indices)
        torch.index_select(input_tensor, 1, indices)
    Paddle: index_select(x, index, axis=0)
        paddle.index_select(x=input_tensor, index=indices, axis=1)
        paddle.index_select(input_tensor, indices, axis=1)
    """

    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            if "input" in kwargs and "x" not in kwargs:
                kwargs["x"] = kwargs.pop("input")
            if "dim" in kwargs and "axis" not in kwargs:
                kwargs["axis"] = kwargs.pop("dim")
            if len(args) >= 2 and isinstance(args[1], int):
                if len(args) < 3 and "index" not in kwargs:
                    raise TypeError(
                        "index_select() missing 1 required argument: 'index'"
                    )
                input_tensor = args[0]
                dim_or_axis = args[1]
                if "x" not in kwargs:
                    kwargs["x"] = input_tensor
                if "axis" not in kwargs:
                    kwargs["axis"] = dim_or_axis
                if len(args) > 2 and "index" not in kwargs:
                    kwargs["index"] = args[2]
                    args = args[3:]
                else:
                    args = args[2:]
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def sum_decorator():
    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            if ("input" in kwargs) and ("x" not in kwargs):
                kwargs["x"] = kwargs.pop("input")
            if ("dim" in kwargs) and ("axis" not in kwargs):
                kwargs["axis"] = kwargs.pop("dim")
            if len(args) == 3:
                kwargs["x"] = args[0]
                kwargs["axis"] = args[1]
                if isinstance(args[2], bool):
                    kwargs["keepdim"] = args[2]
                else:
                    kwargs["dtype"] = args[2]
                args = ()
            elif len(args) == 4:
                kwargs["x"] = args[0]
                kwargs["axis"] = args[1]
                if isinstance(args[2], bool):
                    kwargs["keepdim"] = args[2]
                    kwargs["dtype"] = args[3]
                else:
                    kwargs["dtype"] = args[2]
                    kwargs["keepdim"] = args[3]
                args = ()

            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator
