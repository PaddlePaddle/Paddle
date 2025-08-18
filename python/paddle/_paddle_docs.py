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

import ast
import inspect
from typing import Any

import paddle

from .base.dygraph.generated_tensor_methods_patch import methods_map


def _parse_function_signature(
    func_str: str,
) -> tuple[inspect.Signature, str, dict]:
    """
    Return the inspect.Signaturn for Python function and string signature
    such as "(x,axis=None)" for builtin_function
    """
    func_str = func_str.strip()

    if not func_str.startswith('def '):
        func_str = 'def ' + func_str

    # Create a complete function
    full_def = func_str + ":\n    pass"

    try:
        # Parse AST
        module = ast.parse(full_def)
        func_def = next(
            node for node in module.body if isinstance(node, ast.FunctionDef)
        )
    except Exception as e:
        raise ValueError(f"Failed to parse function definition: {e}") from e

    builtin_annotations_dict = {}

    # Get return annotation
    return_annotation = inspect.Signature.empty
    if func_def.returns:
        return_annotation = _ast_unparse(func_def.returns)
    if return_annotation is not inspect.Signature.empty:
        builtin_annotations_dict.update({"return": str(return_annotation)})

    builtin_sig_str = "("
    # Create parameters
    parameters = []
    count = 0

    # Process the POSITIONAL_OR_KEYWORD parameters
    for param in func_def.args.posonlyargs + func_def.args.args:
        param_name = param.arg
        builtin_param_str = param_name

        annotation = inspect.Parameter.empty
        if param.annotation:
            annotation = _ast_unparse(param.annotation)
            builtin_annotations_dict.update({param_name: str(annotation)})
        # Get Default value
        default = inspect.Parameter.empty

        if func_def.args.defaults and len(func_def.args.defaults) > (
            len(func_def.args.args) - len(func_def.args.defaults)
        ):

            idx = count - (
                len(func_def.args.args) - len(func_def.args.defaults)
            )
            if idx >= 0:
                default_node = func_def.args.defaults[idx]
                default = _ast_literal_eval(default_node)
                builtin_param_str += " = " + str(default)

        # Create inspect.Parameter
        param_obj = inspect.Parameter(
            name=param_name,
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default,
            annotation=annotation,
        )
        builtin_sig_str += f"{builtin_param_str},"

        count += 1
        parameters.append(param_obj)

    # Process the key word only params such as out
    count = 0
    if len(func_def.args.kwonlyargs) > 0:
        builtin_sig_str += "*,"
    for param in func_def.args.kwonlyargs:
        para_name = param.arg
        builtin_param_str = param_name
        annotation = (
            _ast_unparse(param.annotation)
            if param.annotation
            else inspect.Parameter.empty
        )
        if param.annotation:
            builtin_annotations_dict.update({param_name: str(annotation)})
        idx = count
        default = inspect.Parameter.empty
        if idx >= 0 and idx < len(func_def.args.kw_defaults):
            default_node = func_def.args.kw_defaults[idx]
            default = _ast_literal_eval(default_node)
            builtin_param_str += " = " + str(default)
        parameters.append(
            inspect.Parameter(
                name=para_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )
        builtin_sig_str += f"{builtin_param_str}"
        count += 1

    builtin_sig_str += ")"
    # Create inspect.Signature and return builtin_sig_str
    return (
        inspect.Signature(
            parameters=parameters, return_annotation=return_annotation
        ),
        builtin_sig_str,
        builtin_annotations_dict,
    )


def _ast_unparse(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Subscript):
        value = _ast_unparse(node.value)
        slice_str = _ast_unparse(node.slice)
        return f"{value}[{slice_str}]"
    elif isinstance(node, ast.Index):
        return _ast_unparse(node.value)
    elif isinstance(node, ast.Constant):
        # process string
        if isinstance(node.value, str):
            return f"'{node.value}'"
        return str(node.value)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _ast_unparse(node.left)
        right = _ast_unparse(node.right)
        return f"{left} | {right}"
    elif isinstance(node, ast.Attribute):
        return f"{_ast_unparse(node.value)}.{node.attr}"
    elif isinstance(node, ast.Tuple):
        return ", ".join(_ast_unparse(el) for el in node.elts)
    else:
        return ast.dump(node)


def _ast_literal_eval(node: ast.AST) -> Any:
    """Eval and transpose AST node to Python literal"""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.NameConstant):
        return node.value
    elif isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.Str):
        return node.s
    elif isinstance(node, ast.Name) and node.id == "None":
        return None
    elif isinstance(node, ast.Name) and node.id == "True":
        return True
    elif isinstance(node, ast.Name) and node.id == "False":
        return False
    else:
        raise ValueError(f"Unsupported default value: {ast.dump(node)}")


# Add docstr for some C++ functions in paddle
_add_docstr = paddle.base.core.eager._add_docstr


def add_doc_and_signature(method: str, docstr: str, signature: str) -> None:
    """
    Add docstr for function (paddle.*) and method (paddle.Tensor.*) if method exists
    """
    # builtin_sig = "(a,b=1,c=0)"
    python_api_sig, builtin_sig, builtin_ann = _parse_function_signature(
        signature
    )
    for module in [paddle, paddle.Tensor]:
        if hasattr(module, method):
            func = getattr(module, method)
            if inspect.isfunction(func):
                func.__doc__ = docstr
            elif inspect.ismethod(func):
                func.__self__.__doc__ = docstr
            elif inspect.isbuiltin(func):
                _add_docstr(func, docstr, builtin_sig, builtin_ann)
    methods_dict = dict(methods_map)
    if method in methods_dict.keys():
        tensor_func = methods_dict[method]
        tensor_func.__signature__ = python_api_sig


__all__ = ['add_doc_and_signature']
add_doc_and_signature(
    "amin",
    r"""
    Computes the minimum of tensor elements over the given axis

    Note:
        The difference between min and amin is: If there are multiple minimum elements,
        amin evenly distributes gradient between these equal values,
        while min propagates gradient to all of them.

    Args:
        x (Tensor): A tensor, the data type is float32, float64, int32, int64,
            the dimension is no more than 4.
        axis (int|list|tuple|None, optional): The axis along which the minimum is computed.
            If :attr:`None`, compute the minimum over all elements of
            `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim, x.ndim)`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of minimum on the specified axis of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python
            >>> # type: ignore
            >>> import paddle
            >>> # data_x is a Tensor with shape [2, 4] with multiple minimum elements
            >>> # the axis is a int element

            >>> x = paddle.to_tensor([[0.2, 0.1, 0.1, 0.1],
            ...                         [0.1, 0.1, 0.6, 0.7]],
            ...                         dtype='float64', stop_gradient=False)
            >>> # There are 5 minimum elements:
            >>> # 1) amin evenly distributes gradient between these equal values,
            >>> #    thus the corresponding gradients are 1/5=0.2;
            >>> # 2) while min propagates gradient to all of them,
            >>> #    thus the corresponding gradient are 1.
            >>> result1 = paddle.amin(x)
            >>> result1.backward()
            >>> result1
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            0.10000000)
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.20000000, 0.20000000, 0.20000000],
             [0.20000000, 0.20000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result1_min = paddle.min(x)
            >>> result1_min.backward()
            >>> result1_min
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            0.10000000)


            >>> x.clear_grad()
            >>> result2 = paddle.amin(x, axis=0)
            >>> result2.backward()
            >>> result2
            Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.10000000, 0.10000000, 0.10000000, 0.10000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.50000000, 1.        , 1.        ],
             [1.        , 0.50000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result3 = paddle.amin(x, axis=-1)
            >>> result3.backward()
            >>> result3
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.10000000, 0.10000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.33333333, 0.33333333, 0.33333333],
             [0.50000000, 0.50000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result4 = paddle.amin(x, axis=1, keepdim=True)
            >>> result4.backward()
            >>> result4
            Tensor(shape=[2, 1], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.10000000],
             [0.10000000]])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.33333333, 0.33333333, 0.33333333],
             [0.50000000, 0.50000000, 0.        , 0.        ]])

            >>> # data_y is a Tensor with shape [2, 2, 2]
            >>> # the axis is list
            >>> y = paddle.to_tensor([[[0.2, 0.1], [0.1, 0.1]],
            ...                       [[0.1, 0.1], [0.6, 0.7]]],
            ...                       dtype='float64', stop_gradient=False)
            >>> result5 = paddle.amin(y, axis=[1, 2])
            >>> result5.backward()
            >>> result5
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.10000000, 0.10000000])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[0.        , 0.33333333],
              [0.33333333, 0.33333333]],
             [[0.50000000, 0.50000000],
              [0.        , 0.        ]]])

            >>> y.clear_grad()
            >>> result6 = paddle.amin(y, axis=[0, 1])
            >>> result6.backward()
            >>> result6
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.10000000, 0.10000000])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[0.        , 0.33333333],
              [0.50000000, 0.33333333]],
             [[0.50000000, 0.33333333],
              [0.        , 0.        ]]])
""",
    """
def amin(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    keepdim: bool = False,
    name: str | None = None,
) -> Tensor
""",
)

add_doc_and_signature(
    "amax",
    """
    Computes the maximum of tensor elements over the given axis.

    Note:
        The difference between max and amax is: If there are multiple maximum elements,
        amax evenly distributes gradient between these equal values,
        while max propagates gradient to all of them.

    Args:
        x (Tensor): A tensor, the data type is float32, float64, int32, int64,
            the dimension is no more than 4.
        axis (int|list|tuple|None, optional): The axis along which the maximum is computed.
            If :attr:`None`, compute the maximum over all elements of
            `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim(x), x.ndim(x))`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of maximum on the specified axis of input tensor,
        it's data type is the same as `x`.

    Examples:
        .. code-block:: python
            >>> # type: ignore
            >>> import paddle
            >>> # data_x is a Tensor with shape [2, 4] with multiple maximum elements
            >>> # the axis is a int element

            >>> x = paddle.to_tensor([[0.1, 0.9, 0.9, 0.9],
            ...                         [0.9, 0.9, 0.6, 0.7]],
            ...                         dtype='float64', stop_gradient=False)
            >>> # There are 5 maximum elements:
            >>> # 1) amax evenly distributes gradient between these equal values,
            >>> #    thus the corresponding gradients are 1/5=0.2;
            >>> # 2) while max propagates gradient to all of them,
            >>> #    thus the corresponding gradient are 1.
            >>> result1 = paddle.amax(x)
            >>> result1.backward()
            >>> result1
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            0.90000000)
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.20000000, 0.20000000, 0.20000000],
             [0.20000000, 0.20000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result1_max = paddle.max(x)
            >>> result1_max.backward()
            >>> result1_max
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            0.90000000)


            >>> x.clear_grad()
            >>> result2 = paddle.amax(x, axis=0)
            >>> result2.backward()
            >>> result2
            Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.90000000, 0.90000000, 0.90000000, 0.90000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.50000000, 1.        , 1.        ],
             [1.        , 0.50000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result3 = paddle.amax(x, axis=-1)
            >>> result3.backward()
            >>> result3
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.90000000, 0.90000000])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.33333333, 0.33333333, 0.33333333],
             [0.50000000, 0.50000000, 0.        , 0.        ]])

            >>> x.clear_grad()
            >>> result4 = paddle.amax(x, axis=1, keepdim=True)
            >>> result4.backward()
            >>> result4
            Tensor(shape=[2, 1], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.90000000],
             [0.90000000]])
            >>> x.grad
            Tensor(shape=[2, 4], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[0.        , 0.33333333, 0.33333333, 0.33333333],
             [0.50000000, 0.50000000, 0.        , 0.        ]])

            >>> # data_y is a Tensor with shape [2, 2, 2]
            >>> # the axis is list
            >>> y = paddle.to_tensor([[[0.1, 0.9], [0.9, 0.9]],
            ...                         [[0.9, 0.9], [0.6, 0.7]]],
            ...                         dtype='float64', stop_gradient=False)
            >>> result5 = paddle.amax(y, axis=[1, 2])
            >>> result5.backward()
            >>> result5
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.90000000, 0.90000000])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[0.        , 0.33333333],
              [0.33333333, 0.33333333]],
             [[0.50000000, 0.50000000],
              [0.        , 0.        ]]])

            >>> y.clear_grad()
            >>> result6 = paddle.amax(y, axis=[0, 1])
            >>> result6.backward()
            >>> result6
            Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [0.90000000, 0.90000000])
            >>> y.grad
            Tensor(shape=[2, 2, 2], dtype=float64, place=Place(cpu), stop_gradient=False,
            [[[0.        , 0.33333333],
              [0.50000000, 0.33333333]],
             [[0.50000000, 0.33333333],
              [0.        , 0.        ]]])
    """,
    """
def amax(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    keepdim: bool = False,
    name: str | None = None,
) -> Tensor
""",
)
