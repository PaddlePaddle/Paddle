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

from __future__ import annotations

import os
from functools import wraps
from typing import Callable, Union

import numpy as np

import paddle
from paddle import base, get_flags, set_flags, static
from paddle.base import core
from paddle.base.framework import _dygraph_guard
from paddle.base.wrapped_decorator import signature_safe_contextmanager
from paddle.pir_utils import DygraphOldIrGuard
from paddle.utils.environments import (
    BooleanEnvironmentVariable,
    EnvironmentVariableGuard,
)

__all__ = ['DyGraphProgramDescTracerTestHelper', 'is_equal_program']


def is_equal_program(prog1, prog2):
    with _dygraph_guard(None):
        return _is_equal_program(prog1, prog2)


def _is_equal_program(prog1, prog2):
    block_num = prog1.num_blocks
    if block_num != prog2.num_blocks:
        return False

    for block_id in range(block_num):
        block1 = prog1.block(block_id)
        block2 = prog2.block(block_id)

        if len(block1.ops) != len(block2.ops):
            return False

        if len(block1.vars) != len(block2.vars):
            return False

        for op1, op2 in zip(block1.ops, block2.ops):
            if op1.input_arg_names != op2.input_arg_names:
                return False

            if op1.output_arg_names != op2.output_arg_names:
                return False

            attr1 = op1.all_attrs()
            attr2 = op2.all_attrs()

            if len(attr1) != len(attr2):
                return False

            for key1, value1 in attr1.items():
                if key1 not in attr2:
                    return False

                if value1 != attr2.get(key1):
                    return False

        for var1 in block1.vars.values():
            if var1.name not in block2.vars:
                return False

            var2 = block2.vars.get(var1.name)
            if var1.name != var2.name:
                return False

            if var1.type != var2.type:
                return False

            if var1.dtype != var2.dtype:
                return False

            if var1.lod_level != var2.lod_level:
                return False

            if var1.persistable != var2.persistable:
                return False

    return True


def load_dygraph_vars_to_scope(model_path, scope, place):
    def load_dict_to_scope(scope, dictionary):
        if scope is None:
            scope = base.global_scope()

        for k, v in dictionary.items():
            dst_t = scope.var(k).get_tensor()
            src_t = v.value().get_tensor()
            dst_t.set(np.array(src_t), place)
            dst_t.set_lod(src_t.lod())

    param_dict = paddle.load(model_path + '.pdparams')
    opti_dict = paddle.load(model_path + '.pdopt')
    if param_dict:
        load_dict_to_scope(scope, param_dict)

    if opti_dict:
        load_dict_to_scope(scope, opti_dict)


class DyGraphProgramDescTracerTestHelper:
    def __init__(self, unittest_obj):
        self.unittest_obj = unittest_obj

    def assertEachVar(self, out_dygraph, out_static_graph, func=None):
        if func is None:
            func = lambda x, y: np.array_equal(x, y)

        if not isinstance(out_dygraph, (list, tuple)):
            out_dygraph = [out_dygraph]

        if not isinstance(out_static_graph, (list, tuple)):
            out_static_graph = [out_static_graph]

        for v1, v2 in zip(out_dygraph, out_static_graph):
            self.unittest_obj.assertTrue(func(v1.numpy(), v2))


@signature_safe_contextmanager
def dygraph_guard():
    in_dygraph_outside = paddle.base.framework.in_dygraph_mode()
    try:
        if not in_dygraph_outside:
            paddle.disable_static()
        yield
    finally:
        if not in_dygraph_outside:
            paddle.enable_static()


@signature_safe_contextmanager
def static_guard():
    in_dygraph_outside = paddle.base.framework.in_dygraph_mode()
    try:
        if in_dygraph_outside:
            paddle.enable_static()
        yield
    finally:
        if in_dygraph_outside:
            paddle.disable_static()


@signature_safe_contextmanager
def pir_executor_guard():
    tmp_env = os.environ.get("FLAGS_enable_pir_in_executor")
    tmp_cpp = get_flags("FLAGS_enable_pir_in_executor")[
        "FLAGS_enable_pir_in_executor"
    ]
    try:
        os.environ["FLAGS_enable_pir_in_executor"] = 'True'
        set_flags({"FLAGS_enable_pir_in_executor": True})
        yield
    finally:
        if tmp_env is None:
            del os.environ["FLAGS_enable_pir_in_executor"]
        else:
            os.environ["FLAGS_enable_pir_in_executor"] = tmp_env
        set_flags({"FLAGS_enable_pir_in_executor": tmp_cpp})


ENV_ENABLE_PIR_WITH_PT = BooleanEnvironmentVariable(
    "FLAGS_enable_pir_in_executor", False
)


def to_pir_pt_test(fn):
    @wraps(fn)
    def impl(*args, **kwargs):
        with DygraphOldIrGuard():
            pt_flag = ENV_ENABLE_PIR_WITH_PT.name
            original_flag_value = get_flags(pt_flag)[pt_flag]
            if os.environ.get('FLAGS_use_stride_kernel', False):
                return
            with static.scope_guard(static.Scope()):
                with static.program_guard(static.Program()):
                    with EnvironmentVariableGuard(ENV_ENABLE_PIR_WITH_PT, True):
                        try:
                            set_flags({pt_flag: True})
                            ir_outs = fn(*args, **kwargs)
                        finally:
                            set_flags({pt_flag: original_flag_value})
        return ir_outs

    return impl


def compare_legacy_with_pt(fn):
    @wraps(fn)
    def impl(*args, **kwargs):
        outs = fn(*args, **kwargs)
        if core._is_bwd_prim_enabled() or core._is_fwd_prim_enabled():
            return outs
        ir_outs = to_pir_pt_test(fn)(*args, **kwargs)
        np.testing.assert_equal(
            outs,
            ir_outs,
            err_msg=f'Dy2St Unittest Check ({fn.__name__}) has diff \n'
            + f'Expect {outs}\n'
            + f'But Got {ir_outs}',
        )
        return outs

    return impl


FuncType = Callable[[], bool]
PlaceType = Union[paddle.CPUPlace, paddle.CUDAPlace, str]


def convert_place(place: PlaceType) -> str:
    if isinstance(place, paddle.CPUPlace):
        return 'cpu'
    if isinstance(place, paddle.CUDAPlace):
        return 'gpu'
    return place


def get_places(
    func: FuncType = lambda: True, isStr: bool = False
) -> list[PlaceType]:
    places: list[PlaceType] = []
    if paddle.is_compiled_with_cuda() and func():
        places.append(paddle.CUDAPlace(0))
    if (
        os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
        in ['1', 'true', 'on']
        or not places
    ):
        places.insert(0, paddle.CPUPlace())
    if isStr:
        places = [convert_place(place) for place in places]
    return places
