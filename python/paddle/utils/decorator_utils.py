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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    TypeVar,
    cast,
)

from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from collections.abc import Iterable


_P = ParamSpec("_P")
_R = TypeVar("_R")
_DecoratedFunc = Callable[_P, _R]


class DecoratorBase(Generic[_P, _R]):
    """装饰器基类，提供通用装饰器框架

    子类只需实现 `process` 方法定义核心逻辑
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """初始化装饰器参数"""
        self.args = args
        self.kwargs = kwargs

    def __call__(self, func: _DecoratedFunc[_P, _R]) -> _DecoratedFunc[_P, _R]:
        """作为装饰器应用的入口点"""

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            # 预处理参数
            processed_args, processed_kwargs = self.process(args, kwargs)
            # 调用原函数
            return func(*processed_args, **processed_kwargs)

        # 保留原始签名
        wrapper.__signature__ = inspect.signature(func)
        return cast("_DecoratedFunc[_P, _R]", wrapper)

    def process(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """子类必须实现的核心处理方法

        Args:
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            处理后的 (args, kwargs) 元组
        """
        raise NotImplementedError("Subclasses must implement this method")


# 示例实现：参数别名装饰器
class ParamAliasDecorator(DecoratorBase[_P, _R]):
    """参数别名处理的装饰器实现"""

    def __init__(self, alias_mapping: dict[str, Iterable[str]]) -> None:
        super().__init__()
        if not isinstance(alias_mapping, dict):
            raise TypeError("alias_mapping must be a dictionary")
        for k, v in alias_mapping.items():
            if not isinstance(v, (list, tuple, set)):
                raise TypeError(f"Aliases for '{k}' must be iterable")
        self.alias_mapping = alias_mapping

    def process(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not kwargs:
            return args, kwargs
        processed_kwargs = kwargs.copy()
        for original, aliases in self.alias_mapping.items():
            for alias in aliases:
                if alias in processed_kwargs:
                    if original not in processed_kwargs:
                        processed_kwargs[original] = processed_kwargs.pop(alias)
                    else:
                        raise ValueError(
                            f"Cannot specify both '{original}' and its alias '{alias}'"
                        )
        return args, processed_kwargs
