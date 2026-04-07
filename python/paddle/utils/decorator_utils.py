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
from collections.abc import Iterable
from typing import Any, Callable, TypeVar, cast

from typing_extensions import ParamSpec, get_overloads

import paddle

_InputT = ParamSpec("_InputT")
_RetT = TypeVar("_RetT")
_SENTINEL = object()


def _is_int_or_scalar_tensor(x):
    if isinstance(x, int):
        return True
    if isinstance(x, (paddle.Tensor, paddle.pir.Value)):
        return x.ndim == 0
    return False


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


def param_one_alias(
    alias_list,
) -> Callable[[Callable[_InputT, _RetT]], Callable[_InputT, _RetT]]:
    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            if not kwargs:
                return func(*args, **kwargs)
            if alias_list[1] in kwargs:
                if alias_list[0] not in kwargs:
                    kwargs[alias_list[0]] = kwargs.pop(alias_list[1])
                else:
                    raise ValueError(
                        f"Cannot specify both '{alias_list[0]}' and its alias '{alias_list[1]}'"
                    )
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def param_two_alias(
    alias_list1: list[str], alias_list2: list[str]
) -> Callable[[Callable[_InputT, _RetT]], Callable[_InputT, _RetT]]:
    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            if not kwargs:
                return func(*args, **kwargs)
            if alias_list1[1] in kwargs:
                if alias_list1[0] not in kwargs:
                    kwargs[alias_list1[0]] = kwargs.pop(alias_list1[1])
                else:
                    raise ValueError(
                        f"Cannot specify both '{alias_list1[0]}' and its alias '{alias_list1[1]}'"
                    )
            if alias_list2[1] in kwargs:
                if alias_list2[0] not in kwargs:
                    kwargs[alias_list2[0]] = kwargs.pop(alias_list2[1])
                else:
                    raise ValueError(
                        f"Cannot specify both '{alias_list2[0]}' and its alias '{alias_list2[1]}'"
                    )
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def lp_pool_layer_decorator(
    func: Callable[_InputT, _RetT],
) -> Callable[_InputT, _RetT]:
    @functools.wraps(func)
    def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        if len(args) == 5 and isinstance(args[4], bool):
            warnings.warn(
                "The 5th positional argument in '__init__' method is a boolean value, which is being interpreted as 'ceil_mode'.",
                category=Warning,
                stacklevel=2,
            )
            kwargs["ceil_mode"] = args[4]
            args = args[:4]
        return func(*args, **kwargs)

    wrapper.__signature__ = inspect.signature(func)
    return wrapper


def lp_pool_function_decorator(
    func: Callable[_InputT, _RetT],
) -> Callable[_InputT, _RetT]:
    @functools.wraps(func)
    def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        if "input" in kwargs:
            kwargs["x"] = kwargs.pop("input")
        if len(args) == 5 and isinstance(args[4], bool):
            warnings.warn(
                "The 5th positional argument is a boolean value, which is being interpreted as 'ceil_mode'.",
                category=Warning,
                stacklevel=2,
            )
            kwargs["ceil_mode"] = args[4]
            args = args[:4]
        return func(*args, **kwargs)

    wrapper.__signature__ = inspect.signature(func)
    return wrapper


def param_two_alias_one_default(
    alias_list1: list[str], alias_list2: list[str], default_param: list[str]
) -> Callable[[Callable[_InputT, _RetT]], Callable[_InputT, _RetT]]:
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


def size_args_decorator(
    func: Callable[_InputT, _RetT],
) -> Callable[_InputT, _RetT]:
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

        if 'shape' in kwargs and isinstance(kwargs['shape'], int):
            kwargs['shape'] = [kwargs['shape']]

        return func(*args, **kwargs)

    wrapped_func.__signature__ = inspect.signature(func)

    return wrapped_func


def size_args_decorator_patch(
    method: Callable[_InputT, _RetT],
) -> Callable[_InputT, _RetT]:
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


def view_decorator() -> Callable[
    [Callable[_InputT, _RetT]], Callable[_InputT, _RetT]
]:
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
            elif len(args) >= 2 and _is_int_or_scalar_tensor(args[1]):
                if all(_is_int_or_scalar_tensor(arg) for arg in args[1:]):
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

                (prefix omitted)/docs/zh/develop/guides/model_convert/convert_from_pytorch/api_difference/invok_only_diff/**torch.nn.Unfold**.html

                In this example, the correct `url_suffix` should be 'torch/torch.nn.Unfold'. Defaults to an empty str.
        """
        super().__init__()
        self.illegal_keys = illegal_keys
        self.func_name = func_name
        self.correct_name = correct_name
        self.warn_msg = None
        if url_suffix:
            self.warn_msg = (
                f"The API '{func_name}' may behave differently from its PyTorch counterpart. "
                "Refer to the compatibility guide for details:\n"
                "https://www.paddlepaddle.org.cn/documentation/docs/en/develop/guides/model_convert/"
                f"convert_from_pytorch/api_difference/invok_only_diff/{url_suffix}.html"
            )

    def process(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        found_keys = [key for key in self.illegal_keys if key in kwargs]

        if found_keys:
            found_keys.sort()
            keys_str = ", ".join(f"'{key}'" for key in found_keys)
            plural = "s" if len(found_keys) > 1 else ""

            if (
                self.warn_msg is not None
            ):  # warn the users only when the API is mis-used
                warnings.warn(
                    self.warn_msg,
                    category=UserWarning,
                    stacklevel=3,
                )
                self.warn_msg = None
            raise TypeError(
                f"{self.func_name}() received unexpected keyword argument{plural} {keys_str}. "
                f"\nDid you mean to use {self.correct_name}() instead?"
            )
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

                (prefix omitted)/docs/zh/develop/guides/model_convert/convert_from_pytorch/api_difference/invok_only_diff/**torch.nn.Unfold**.html

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


def reshape_decorator() -> Callable[
    [Callable[_InputT, _RetT]], Callable[_InputT, _RetT]
]:
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
            elif len(args) >= 2 and _is_int_or_scalar_tensor(args[1]):
                if all(_is_int_or_scalar_tensor(arg) for arg in args[1:]):
                    kwargs["x"] = args[0]
                    kwargs['shape'] = list(args[1:])
                    args = ()
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def transpose_decorator() -> Callable[
    [Callable[_InputT, _RetT]], Callable[_InputT, _RetT]
]:
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


def expand_decorator() -> Callable[
    [Callable[_InputT, _RetT]], Callable[_InputT, _RetT]
]:
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
            elif len(args) >= 2 and _is_int_or_scalar_tensor(args[1]):
                if all(_is_int_or_scalar_tensor(arg) for arg in args[1:]):
                    kwargs["x"] = args[0]
                    kwargs['shape'] = list(args[1:])
                    args = ()
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def tile_decorator() -> Callable[
    [Callable[_InputT, _RetT]], Callable[_InputT, _RetT]
]:
    """
    Usage Example:
    paddle.tile(x=tensor_x, repeat_times=[2, 3], name=None)
    paddle.tile(input=tensor_x, dims=[2, 3])
    tensor_x.tile([2, 3]) -> paddle.tile(tensor_x, [2, 3])
    tensor_x.tile(2, 3) -> paddle.tile(tensor_x, 2, 3)
    """

    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            if "input" in kwargs:
                if "x" in kwargs:
                    raise ValueError(
                        "Cannot specify both 'x' and its alias 'input'"
                    )
                kwargs["x"] = kwargs.pop("input")

            if "dims" in kwargs:
                if "repeat_times" in kwargs:
                    raise ValueError(
                        "Cannot specify both 'repeat_times' and its alias 'dims'"
                    )
                kwargs["repeat_times"] = kwargs.pop("dims")

            if len(args) >= 2 and isinstance(args[1], int):
                kwargs["x"] = args[0]
                kwargs["repeat_times"] = list(args[1:])
                args = ()
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def index_select_decorator() -> Callable[
    [Callable[_InputT, _RetT]], Callable[_InputT, _RetT]
]:
    """
    Usage Example:
    PyTorch: torch.index_select(input, dim, index)
    Paddle: paddle.index_select(x, index, axis)
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


_SA0_RD1 = {'size_average': 0, 'reduce': 1}
_SA1_RD2 = {'size_average': 1, 'reduce': 2}
_SA1_RD3 = {'size_average': 1, 'reduce': 3}
_SA3_RD4 = {'size_average': 3, 'reduce': 4}
_SA4_RD5 = {'size_average': 4, 'reduce': 5}
_SA2_RD4 = {'size_average': 2, 'reduce': 4}

LEGACY_POS: dict[str, dict[str, int]] = {
    **dict.fromkeys(
        (
            'L1Loss',
            'MSELoss',
            'KLDivLoss',
            'SmoothL1Loss',
            'SoftMarginLoss',
            'MultiLabelMarginLoss',
        ),
        _SA0_RD1,
    ),
    **dict.fromkeys(
        (
            'BCELoss',
            'BCEWithLogitsLoss',
            'MultiLabelSoftMarginLoss',
            'HingeEmbeddingLoss',
            'CosineEmbeddingLoss',
            'MarginRankingLoss',
        ),
        _SA1_RD2,
    ),
    'CrossEntropyLoss': _SA1_RD3,
    'NLLLoss': _SA1_RD3,
    'PoissonNLLLoss': _SA2_RD4,
    'MultiMarginLoss': _SA3_RD4,
    'TripletMarginLoss': _SA4_RD5,
}


def compute_legacy_reduction(reduce_val, size_average_val):
    if reduce_val is False:
        return 'none'
    if reduce_val is True:
        return 'sum' if size_average_val is False else 'mean'
    return 'sum' if size_average_val is False else 'mean'


def get_legacy_reduce_and_size_average(cls_name, args, kwargs):
    reduce_val = ''
    size_avg_val = ''
    pos = LEGACY_POS.get(cls_name)
    idx = pos.get('size_average')
    if 'size_average' in kwargs:
        size_avg_val = kwargs.pop('size_average')
    elif len(args) > idx:
        v = args[idx]
        if type(v) is bool:
            size_avg_val = v
    idx = pos.get('reduce')
    if 'reduce' in kwargs:
        reduce_val = kwargs.pop('reduce')
    elif len(args) > idx:
        v = args[idx]
        if type(v) is bool:
            reduce_val = v
    return reduce_val, size_avg_val


def raise_deprecated_error(cls_name, reduce_val, size_avg_val):
    suggested = compute_legacy_reduction(reduce_val, size_avg_val)
    reduce_val = None if reduce_val == '' else reduce_val
    size_avg_val = None if size_avg_val == '' else size_avg_val
    raise ValueError(
        f"[Deprecated] '{cls_name}' no longer supports 'reduce' or 'size_average'."
        f"\nDetected: reduce={reduce_val}, size_average={size_avg_val}"
        f"\nPlease use: reduction='{suggested}' instead."
    )


def legacy_reduction_decorator(
    init_func: Callable[_InputT, _RetT],
) -> Callable[_InputT, _RetT]:
    """
    Function decorator for __init__: intercept deprecated 'reduce' and 'size_average'.
    """

    @functools.wraps(init_func)
    def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        # avoid subclass calling parent class init, causing cls_name to be inaccurate
        cls_name = init_func.__qualname__.split(".")[0]
        reduce_val, size_avg_val = get_legacy_reduce_and_size_average(
            cls_name, args[1:], kwargs
        )
        if reduce_val != '' or size_avg_val != '':
            raise_deprecated_error(cls_name, reduce_val, size_avg_val)

        return init_func(*args, **kwargs)

    wrapper.__signature__ = inspect.signature(init_func)
    return wrapper


def legacy_reduction_special_decorator(
    init_func: Callable[_InputT, _RetT],
) -> Callable[_InputT, _RetT]:
    """
    Specialized decorator: add CrossEntropyLoss / KLDivLoss special case judgment
    based on the general legacy_reduction_decorator logic.
    """

    @functools.wraps(init_func)
    def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        cls_name = init_func.__qualname__.split(".")[0]
        use_args = args[1:]
        reduce_val, size_avg_val = get_legacy_reduce_and_size_average(
            cls_name, use_args, kwargs
        )
        if reduce_val != '' or size_avg_val != '':
            if not (
                (
                    cls_name == 'CrossEntropyLoss'
                    and len(use_args) > 2
                    and use_args[2] in {'mean', 'sum', 'none'}
                )
                or (
                    cls_name == 'KLDivLoss'
                    and len(use_args) > 0
                    and use_args[0] in {'mean', 'sum', 'none', 'batchmean'}
                )
            ):
                raise_deprecated_error(cls_name, reduce_val, size_avg_val)
        return init_func(*args, **kwargs)

    wrapper.__signature__ = inspect.signature(init_func)
    return wrapper


def index_add_decorator() -> Callable[
    [Callable[_InputT, _RetT]], Callable[_InputT, _RetT]
]:
    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> _RetT:
            if "input" in kwargs:
                kwargs["x"] = kwargs.pop("input")
            if "dim" in kwargs:
                kwargs["axis"] = kwargs.pop("dim")
            if "source" in kwargs:
                kwargs["value"] = kwargs.pop("source")

            if len(args) >= 2 and isinstance(args[1], int):
                kwargs["x"] = args[0]
                kwargs["axis"] = args[1]
                if len(args) > 2:
                    kwargs["index"] = args[2]
                if len(args) > 3:
                    kwargs["value"] = args[3]
                args = ()

            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def maxpool_decorator() -> Callable[
    [Callable[_InputT, _RetT]], Callable[_InputT, _RetT]
]:
    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> _RetT:
            if "input" in kwargs:
                kwargs["x"] = kwargs.pop("input")
            if "return_indices" in kwargs:
                kwargs["return_mask"] = kwargs.pop("return_indices")

            if len(args) >= 5 and not isinstance(args[4], bool):
                kwargs["x"] = args[0]
                kwargs["kernel_size"] = args[1]
                kwargs["stride"] = args[2]
                kwargs["padding"] = args[3]
                kwargs["dilation"] = args[4]
                # The order of `ceil_mode` and `return_indices` is different from nn.MaxPool in PyTorch
                if len(args) > 5:
                    kwargs["ceil_mode"] = args[5]
                if len(args) > 6:
                    kwargs["return_mask"] = args[6]
                args = ()

            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def maxpool_layer_decorator() -> Callable[
    [Callable[_InputT, _RetT]], Callable[_InputT, _RetT]
]:
    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> _RetT:
            if "return_indices" in kwargs:
                kwargs["return_mask"] = kwargs.pop("return_indices")

            if len(args) >= 5 and not isinstance(args[4], bool):
                kwargs["kernel_size"] = args[1]
                kwargs["stride"] = args[2]
                kwargs["padding"] = args[3]
                kwargs["dilation"] = args[4]
                # The order of `ceil_mode` and `return_indices` is different from F.max_pool in PyTorch
                if len(args) > 5:
                    kwargs["return_mask"] = args[5]
                if len(args) > 6:
                    kwargs["ceil_mode"] = args[6]
                args = (args[0],)

            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def use_first_signature(
    func: Callable[_InputT, _RetT],
) -> Callable[_InputT, _RetT]:
    overloads = get_overloads(func)
    if not overloads:
        return func
    first_overload = overloads[0]
    sig = inspect.signature(first_overload)
    func.__signature__ = sig
    return func


def variadic_tensor_decorator(
    param_name: str,
) -> Callable[[Callable[_InputT, _RetT]], Callable[_InputT, _RetT]]:
    """
    Decorator to handle variadic tensor arguments.

    Usage Example:
    PyTorch: torch.block_diag(x, y, z)
    Paddle: paddle.block_diag([x, y, z])

    Args:
        param_name: The parameter name to use for the list (e.g., 'inputs', 'input', 'x')
    """

    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            # PyTorch usage: variadic tensor arguments
            if len(args) >= 1 and isinstance(
                args[0], (paddle.Tensor, paddle.pir.Value)
            ):
                kwargs[param_name] = list(args)
                args = ()
            # Paddle usage: list/tuple argument
            elif len(args) >= 1 and isinstance(args[0], (list, tuple)):
                kwargs[param_name] = args[0]
                args = ()
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def grad_scaler_decorator() -> Callable[
    [Callable[_InputT, _RetT]], Callable[_InputT, _RetT]
]:
    """Decorator for GradScaler.__init__ to support three calling conventions:

    GradScaler(enable, init_loss_scaling, incr_ratio, decr_ratio, incr_every_n_steps, decr_every_n_nan_or_inf, use_dynamic_loss_scaling)
    GradScaler(device, init_scale, growth_factor, backoff_factor, growth_interval, enabled)
    GradScaler(init_scale, growth_factor, backoff_factor, growth_interval, enabled)
    """

    _ALIAS_MAP = {
        'enabled': 'enable',
        'init_scale': 'init_loss_scaling',
        'growth_factor': 'incr_ratio',
        'backoff_factor': 'decr_ratio',
        'growth_interval': 'incr_every_n_steps',
    }
    # PyTorch positional order (device already stripped): init_scale, growth_factor, backoff_factor, growth_interval, enabled
    _TORCH_POS_NAMES = [
        'init_loss_scaling',
        'incr_ratio',
        'decr_ratio',
        'incr_every_n_steps',
        'enable',
    ]

    def _remap_kwargs(kwargs: dict[str, Any]) -> None:
        for torch_key, paddle_key in _ALIAS_MAP.items():
            if torch_key in kwargs:
                if paddle_key not in kwargs:
                    kwargs[paddle_key] = kwargs.pop(torch_key)
                else:
                    raise ValueError(
                        f"Cannot specify both '{paddle_key}' and its alias '{torch_key}'"
                    )

    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> _RetT:
            # args[0] is always `self` for a bound __init__ call
            real_args = args[1:]

            # Drop PyTorch-only 'device' kwarg (no Paddle equivalent)
            kwargs.pop('device', None)

            # Remap PyTorch keyword aliases to Paddle names unconditionally
            _remap_kwargs(kwargs)

            if real_args and isinstance(real_args[0], str):
                # PyTorch with device prefix: GradScaler('cuda', init_scale=1024, ...)
                # Strip device; remaining positional follow torch order
                torch_pos = real_args[1:]
                args = args[:1]  # keep only self
                for i, val in enumerate(torch_pos):
                    if i < len(_TORCH_POS_NAMES):
                        name = _TORCH_POS_NAMES[i]
                        if name not in kwargs:
                            kwargs[name] = val
            elif real_args and not isinstance(real_args[0], bool):
                # PyTorch positional without device: GradScaler(1024, 2.0, 0.5, ...)
                # int/float but not bool: first arg is init_scale (PyTorch), not enable (Paddle)
                args = args[:1]  # keep only self
                for i, val in enumerate(real_args):
                    if i < len(_TORCH_POS_NAMES):
                        name = _TORCH_POS_NAMES[i]
                        if name not in kwargs:
                            kwargs[name] = val
            # else: Paddle call — pass through unchanged

            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def index_fill_decorator() -> Callable[
    [Callable[_InputT, _RetT]], Callable[_InputT, _RetT]
]:
    """
    Decorator for index_fill API to handle parameter name and order differences.

    Usage Example:
    PyTorch: torch.index_fill(input, dim, index, value)
    Paddle: paddle.index_fill(x, index, axis, value)
    """

    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            # Handle keyword argument aliases
            if "input" in kwargs and "x" not in kwargs:
                kwargs["x"] = kwargs.pop("input")
            if "dim" in kwargs and "axis" not in kwargs:
                kwargs["axis"] = kwargs.pop("dim")

            # Handle PyTorch positional argument order: (input, dim, index, value)
            # Paddle order: (x, index, axis, value)
            if len(args) >= 2 and isinstance(args[1], int):
                # PyTorch order detected
                kwargs["x"] = args[0]
                kwargs["axis"] = args[1]
                if len(args) > 2:
                    kwargs["index"] = args[2]
                if len(args) > 3:
                    kwargs["value"] = args[3]
                args = args[4:] if len(args) > 4 else ()

            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def tensor_cuda_decorator() -> Callable[
    [Callable[_InputT, _RetT]], Callable[_InputT, _RetT]
]:
    """
    Usage Example:
    PyTorch: Tensor.cuda(device: DeviceLike, non_blocking: bool = False)
    Paddle: Tensor.cuda(device_id: DeviceLike, blocking: bool = True)
    """

    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            if "device" in kwargs:
                if "device_id" not in kwargs:
                    kwargs["device_id"] = kwargs.pop("device")
                else:
                    raise ValueError(
                        "Cannot specify both 'device' and its alias 'device_id'."
                    )

            if "non_blocking" in kwargs:
                if "blocking" not in kwargs:
                    kwargs["blocking"] = not (kwargs.pop("non_blocking"))
                else:
                    raise ValueError(
                        "Cannot specify both 'blocking' and 'non_blocking'."
                    )

            if len(args) >= 3 and isinstance(args[1], str):
                # using pytorch signature
                # args[0] is self
                if "device_id" not in kwargs:
                    kwargs["device_id"] = args[1]
                if "blocking" not in kwargs:
                    kwargs["blocking"] = not args[2]
                if len(args) > 3:
                    raise ValueError("cuda() received too many arguments")
                args = args[:1]
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


def batch_sampler_decorator() -> Callable[
    [Callable[_InputT, _RetT]], Callable[_InputT, _RetT]
]:
    """
    Usage Example:
    PyTorch: torch.utils.data.BatchSampler(sampler, batch_size, drop_last)
    Paddle: paddle.utils.data.BatchSampler(dataset, sampler, shuffle, batch_size, drop_last)
    """

    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            # args[0] is self
            # args[1] is Sampler / Iterable, use torch signature
            if len(args) >= 2 and isinstance(
                args[1], (paddle.io.Sampler, Iterable)
            ):
                kwargs["sampler"] = args[1]
                if len(args) >= 3:
                    kwargs["batch_size"] = args[2]
                if len(args) == 4:
                    kwargs["drop_last"] = args[3]
                if len(args) > 4:
                    raise TypeError(
                        "BatchSampler() received too many arguments"
                    )
                args = (args[0],)
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator
