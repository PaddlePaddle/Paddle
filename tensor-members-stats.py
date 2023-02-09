# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from dataclasses import dataclass

import paddle

monkey_patch_tensor_methods_1: list[str] = paddle.tensor.tensor_method_func
# python/paddle/fluid/dygraph/math_op_patch.py - eager_methods
monkey_patch_tensor_methods_2: list[str] = [
    '__neg__',
    '__float__',
    '__long__',
    '__int__',
    '__len__',
    '__index__',
    'astype',
    'dim',
    'ndimension',
    'ndim',
    'size',
    'T',
    '__array_ufunc__',
]
# python/paddle/fluid/dygraph/varbase_patch_methods.py
monkey_patch_tensor_methods_3: list[str] = [
    "__bool__",
    "__nonzero__",
    "_to_static_var",
    "set_value",
    "block",
    "backward",
    "clear_grad",
    "inplace_version",
    "gradient",
    "register_hook",
    "__str__",
    "__repr__",
    "__deepcopy__",
    "__module__",
    "__array__",
    "__getitem__",
    "item",
    "__setitem__",
    "_to",
    "values",
    "to_dense",
    "to_sparse_coo",
    # --
    "_set_grad_ivar",
    "value",
    "cpu",
    "cuda",
    "pin_memory",
    "_slice",
    "_numel",
    "_uva",
    "_clear_data",
    "__hash__",
    "_use_gpudnn",
]
# python/paddle/tensor/manipulation.py - __METHODS
monkey_patch_tensor_methods_4: list[str] = [
    'fill_',
    'zero_',
    'fill_diagonal_',
    'fill_diagonal_tensor_',
    'fill_diagonal_tensor',
    'tolist',
]
monkey_patch_tensor_methods = (
    monkey_patch_tensor_methods_1
    + monkey_patch_tensor_methods_2
    + monkey_patch_tensor_methods_3
    + monkey_patch_tensor_methods_4
)
# In Docs
# JSON.stringify([...document.querySelectorAll(".method span.sig-name.descname > span"), ...document.querySelectorAll(".property span.sig-name.descname > span")].map(node => node.innerText))
members_in_docs: list[str] = [
    "abs",
    "acos",
    "acosh",
    "add",
    "add_",
    "add_n",
    "addmm",
    "all",
    "allclose",
    "amax",
    "amin",
    "angle",
    "any",
    "argmax",
    "argmin",
    "argsort",
    "as_complex",
    "as_real",
    "asin",
    "asinh",
    "astype",
    "atan",
    "atanh",
    "backward",
    "bincount",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "bmm",
    "broadcast_shape",
    "broadcast_tensors",
    "broadcast_to",
    "bucketize",
    "cast",
    "ceil",
    "ceil_",
    "cholesky",
    "cholesky_solve",
    "chunk",
    "clear_grad",
    "clip",
    "clip_",
    "concat",
    "cond",
    "conj",
    "corrcoef",
    "cos",
    "cosh",
    "count_nonzero",
    "cov",
    "create_parameter",
    "create_tensor",
    "cross",
    "cumprod",
    "cumsum",
    "deg2rad",
    "diagonal",
    "diff",
    "digamma",
    "dist",
    "divide",
    "dot",
    "eig",
    "eigvals",
    "eigvalsh",
    "equal",
    "equal_all",
    "erf",
    "erfinv",
    "erfinv_",
    "exp",
    "exp_",
    "expand",
    "expand_as",
    "exponential_",
    "fill_",
    "fill_diagonal_",
    "fill_diagonal_tensor",
    "fill_diagonal_tensor_",
    "flatten",
    "flatten_",
    "flip",
    "floor",
    "floor_",
    "floor_divide",
    "floor_mod",
    "fmax",
    "fmin",
    "frac",
    "frexp",
    "gather",
    "gather_nd",
    "gcd",
    "gradient",
    "greater_equal",
    "greater_than",
    "heaviside",
    "histogram",
    "imag",
    "increment",
    "index_add",
    "index_add_",
    "index_sample",
    "index_select",
    "inner",
    "inverse",
    "is_complex",
    "is_empty",
    "is_floating_point",
    "is_integer",
    "is_tensor",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "item",
    "kron",
    "kthvalue",
    "lcm",
    "lerp",
    "lerp_",
    "less_equal",
    "less_than",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "logcumsumexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logit",
    "logsumexp",
    "lstsq",
    "lu",
    "lu_unpack",
    "masked_select",
    "matmul",
    "matrix_power",
    "max",
    "maximum",
    "mean",
    "median",
    "min",
    "minimum",
    "mm",
    "mod",
    "mode",
    "moveaxis",
    "multi_dot",
    "multiplex",
    "multiply",
    "mv",
    "nan_to_num",
    "nanmean",
    "nanmedian",
    "nanquantile",
    "nansum",
    "neg",
    "nonzero",
    "norm",
    "not_equal",
    "numel",
    "outer",
    "pow",
    "prod",
    "put_along_axis",
    "put_along_axis_",
    "qr",
    "quantile",
    "rad2deg",
    "rank",
    "real",
    "reciprocal",
    "reciprocal_",
    "register_hook",
    "remainder",
    "remainder_",
    "repeat_interleave",
    "reshape",
    "reshape_",
    "reverse",
    "roll",
    "rot90",
    "round",
    "round_",
    "rsqrt",
    "rsqrt_",
    "scale",
    "scale_",
    "scatter",
    "scatter_",
    "scatter_nd",
    "scatter_nd_add",
    "set_value",
    "sgn",
    "shard_index",
    "sign",
    "sin",
    "sinh",
    "slice",
    "solve",
    "sort",
    "split",
    "sqrt",
    "sqrt_",
    "square",
    "squeeze",
    "squeeze_",
    "stack",
    "stanh",
    "std",
    "strided_slice",
    "subtract",
    "subtract_",
    "sum",
    "t",
    "take",
    "take_along_axis",
    "tanh",
    "tanh_",
    "tensordot",
    "tile",
    "to_dense",
    "to_sparse_coo",
    "tolist",
    "topk",
    "trace",
    "transpose",
    "trunc",
    "unbind",
    "uniform_",
    "unique",
    "unique_consecutive",
    "unsqueeze",
    "unsqueeze_",
    "unstack",
    "values",
    "var",
    "vsplit",
    "where",
    "zero_",
    "inplace_version",
]


@dataclass
class Member:
    id: int
    name: str
    has_signature: bool
    has_doc: bool
    is_patched: bool
    is_magic_member: bool
    in_doc: bool
    aliases: list[str]

    def add_alias(self, alias: str):
        self.aliases.append(alias)


def is_inherited_member(name: str, cls: type) -> bool:
    """Check if the member is inherited from parent class"""

    if name in cls.__dict__:
        return False

    for base in cls.__bases__:
        if name in base.__dict__:
            return True

    return any(is_inherited_member(name, base) for base in cls.__bases__)


def is_magic_member(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def get_tensor_members():
    tensor_class = paddle.Tensor

    members: dict[int, Member] = {}
    for name, member in inspect.getmembers(tensor_class):
        member_id = id(member)
        member_doc = inspect.getdoc(member)
        member_has_doc = member_doc is not None
        try:
            sig = inspect.signature(member)
            member_has_signature = True
        except (TypeError, ValueError):
            member_has_signature = False

        if is_inherited_member(name, tensor_class):
            continue

        # Filter out private members except magic methods
        if name.startswith("_") and not (
            name.startswith("__") and name.endswith("__")
        ):
            continue

        if member_id in members:
            members[member_id].add_alias(name)
            continue

        members[member_id] = Member(
            member_id,
            name,
            has_signature=member_has_signature,
            has_doc=member_has_doc,
            is_patched=name in monkey_patch_tensor_methods,
            is_magic_member=is_magic_member(name),
            in_doc=name in members_in_docs,
            aliases=[],
        )

    return members


if __name__ == "__main__":
    members = get_tensor_members()
    count_has_signature = 0
    count_has_doc = 0
    count_is_patched = 0
    count_in_doc = 0
    total = len(members)

    for member in members.values():
        if member.has_signature:
            count_has_signature += 1
        if member.has_doc:
            count_has_doc += 1
        if member.is_patched:
            count_is_patched += 1
        if member.in_doc:
            count_in_doc += 1

        # # API 在文档的充要条件：有文档，不是 private，不是 magic method
        # if member.in_doc != (member.has_doc and not member.is_magic_member):
        #     print(member)
        # 即，与文档对齐就是，补全 magic members 的属性，可手动维护

        if not member.has_doc:
            print(member)

    print("Total:", total)
    print("Has signature:", count_has_signature)
    print("Has doc:", count_has_doc)
    print("Is patched:", count_is_patched)
    print("In doc:", count_in_doc)
    print(paddle.__version__)
    print(len(monkey_patch_tensor_methods))
