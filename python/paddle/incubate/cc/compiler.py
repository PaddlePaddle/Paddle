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

import inspect
import os
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import paddle
from paddle.static import InputSpec

from . import typing as pct

__all__ = ['compile']


# Usage:
# import paddle.incubate.cc.typing as pct
# import paddle.incubate.cc as pcc
# import paddle.nn.functional as F
#
# N = pct.DimVar('N', min=2)
# K = pct.DimVar("K", min=2)
# M = pct.DimVar("M", 7168)
# DType = pct.DTypeVar("T", "bfloat16", "float32")
#
# def foo(
#   x: pct.Tensor([N, K], DType),
#   y: pct.Tensor([K, M], DType),
#   b: pct.Tensor([M], DType)
# ):
#   @pcc.force_register_fusion
#   def activate(out):
#     return F.relu(out + b)
#   return activate(x @ y)
#
# fused_foo = pcc.compile(
#   foo
# )
def compile(func, *args, **kwargs):
    annotations = _get_input_annotations(func)
    dtypes2func = {}
    for input_specs in _get_input_spec_lists(annotations):
        dtypes = tuple(input_spec.dtype for input_spec in input_specs)
        dtypes2func[dtypes] = _compile(func, input_specs, *args, **kwargs)
    return OverloadedFunc(FuncOverloadCtx(dtypes2func))


def _compile(
    func,
    input_specs,
    train=False,
    ap_path=None,
    ap_workspace_dir='/tmp/paddle/ap',
    backend_device='cuda',
    target_framework='paddle',
    compile_engine='PCC',
):
    assert ap_path is not None
    assert not train, "only support inference now"
    os.makedirs(ap_workspace_dir, exist_ok=True)
    build_strategy = paddle.static.BuildStrategy()
    assert compile_engine in ('CINN', 'PCC')
    with _ap_envs(ap_path, ap_workspace_dir):
        static_fn = paddle.jit.to_static(
            func,
            input_spec=input_specs,
            build_strategy=build_strategy,
            full_graph=True,
            backend=compile_engine,
        )
        if not train:
            static_fn.eval()
        else:
            static_fn.train()
        concrete_program, partial_program_layer = (
            static_fn.get_concrete_program(
                *input_specs, is_train=static_fn._is_train_mode()
            )
        )
        partial_program_layer.training = static_fn._is_train_mode()
        # Force to generate the program immediately.
        if train:
            _ = partial_program_layer.train_program.forward_program
        else:
            _ = partial_program_layer.infer_program.forward_program
        return partial_program_layer


@dataclass
class FuncOverloadCtx:
    dtypes2func: dict[list[paddle.dtype], Callable]


class OverloadedFunc:
    def __init__(self, func_overload_ctx: FuncOverloadCtx):
        self.func_overload_ctx = func_overload_ctx

    def __call__(self, *args):
        dtypes = tuple(tensor.dtype for tensor in args)
        func = self.func_overload_ctx.dtypes2func.get(dtypes, None)
        assert func is not None, self.mismatched_debug_info(dtypes)
        return func(inputs=[*args])

    def mismatched_debug_info(self, dtypes):
        valid_signatures = "; ".join(
            f"[{idx+1}] {dtypes}"
            for idx, pair in enumerate(
                self.func_overload_ctx.dtypes2func.items()
            )
            for dtypes in [pair[0]]
        )
        return f"input signature {dtypes} mismatched, valid signatures are: {valid_signatures}"


@dataclass
class InputSpecMakeCtx:
    name2dtype_num_candidates: dict[str, int]
    name2dtype_candidate_idx: dict[str, int]


@contextmanager
def _ap_envs(ap_path, ap_workspace_dir):
    old_ap_path = os.environ.get('AP_PATH')
    old_ap_workspace_dir = os.environ.get('AP_WORKSPACE_DIR')
    os.environ['AP_PATH'] = (
        f"{ap_path}:{old_ap_path}" if old_ap_path is not None else ap_path
    )
    os.environ['AP_WORKSPACE_DIR'] = ap_workspace_dir
    old_flags = paddle.get_flags(['FLAGS_enable_ap'])
    flags = dict(old_flags)
    flags['FLAGS_enable_ap'] = True
    paddle.set_flags(flags)
    yield
    if old_ap_path is not None:
        os.environ['AP_PATH'] = old_ap_path
    else:
        del os.environ['AP_PATH']
    if old_ap_workspace_dir is not None:
        os.environ['AP_WORKSPACE_DIR'] = old_ap_workspace_dir
    else:
        del os.environ['AP_WORKSPACE_DIR']
    paddle.set_flags(old_flags)


def _get_input_annotations(func):
    full_arg_spec = inspect.getfullargspec(func)
    return [
        pct_type
        for arg_name in full_arg_spec.args
        for pct_type in [full_arg_spec.annotations[arg_name]]
    ]


def _get_input_spec_lists(annotations):
    ctx = _create_empty_input_spec_make_ctx(annotations)
    assert len(ctx.name2dtype_num_candidates) > 0
    dtype_var_names = [
        pair[0] for pair in ctx.name2dtype_num_candidates.items()
    ]
    dtype_num_candidates = [
        pair[1] for pair in ctx.name2dtype_num_candidates.items()
    ]
    dtype_candidate_idx_compositions = _cartesian_product(
        [range(num_candidates) for num_candidates in dtype_num_candidates]
    )
    for idx_composition in dtype_candidate_idx_compositions:
        for arg_idx, candidate_idx in enumerate(idx_composition):
            ctx.name2dtype_candidate_idx[dtype_var_names[arg_idx]] = (
                candidate_idx
            )
        yield _get_input_specs(annotations, ctx)


def _create_empty_input_spec_make_ctx(annotations):
    ctx = InputSpecMakeCtx(OrderedDict(), OrderedDict())
    _init_empty_input_spec_make_ctx(annotations, ctx)
    return ctx


def _init_empty_input_spec_make_ctx(annotations, mut_ctx: InputSpecMakeCtx):
    for pct_type in annotations:
        _init_input_spec_make_ctx_name2dtype_num_candidates(pct_type, mut_ctx)


def _init_input_spec_make_ctx_name2dtype_num_candidates(
    pct_type, mut_ctx: InputSpecMakeCtx
):
    assert isinstance(
        pct_type.dtype, pct.DTypeVar
    ), f"pct_type.dtype should be a DTypeVar, but {type(pct_type.dtype)} were given."
    name = pct_type.dtype.name
    if name in mut_ctx.name2dtype_num_candidates:
        assert mut_ctx.name2dtype_num_candidates[name] == len(
            pct_type.dtype.candidates
        )
    else:
        mut_ctx.name2dtype_num_candidates[name] = len(pct_type.dtype.candidates)


def _get_input_specs(annotations, ctx: InputSpecMakeCtx):
    return [_get_input_spec(pct_type, ctx) for pct_type in annotations]


def _get_input_spec(pct_type, ctx: InputSpecMakeCtx):
    assert isinstance(pct_type, pct.Tensor)
    return InputSpec(
        shape=_get_input_spec_shape(pct_type, ctx),
        dtype=_get_input_spec_dtype(pct_type, ctx),
    )


def _get_input_spec_shape(pct_type, ctx: InputSpecMakeCtx):
    return [_get_input_spec_shape_dim(dim_var) for dim_var in pct_type.shape]


def _get_input_spec_shape_dim(dim_var: pct.DimVar):
    if isinstance(dim_var, int):
        return dim_var
    assert isinstance(dim_var, pct.DimVar)
    if isinstance(dim_var.name_or_value, int):
        return dim_var.name_or_value
    return None


def _get_input_spec_dtype(pct_type, ctx: InputSpecMakeCtx):
    assert isinstance(pct_type.dtype, pct.DTypeVar)
    name = pct_type.dtype.name
    candidate_idx = ctx.name2dtype_candidate_idx[name]
    return pct_type.dtype.candidates[candidate_idx]


def _cartesian_product(lst_of_lst):
    assert len(lst_of_lst) > 0
    return _cartesian_product_impl([()], lst_of_lst)


def _cartesian_product_impl(collect_lst, lst_of_lst):
    if len(lst_of_lst) == 0:
        return collect_lst
    collect_lst = [(*x, y) for x in collect_lst for y in lst_of_lst[0]]
    return _cartesian_product_impl(collect_lst, lst_of_lst[1:])
