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


import cinn
from cinn import ir, lang, to_cinn_llir
from cinn.runtime.data_array import DataArray


def _assert_print(script, expected):
    script_str = str(script).strip()
    expected_str = str(expected).strip()
    min_len = (
        len(script_str)
        if len(script_str) < len(expected_str)
        else len(expected_str)
    )
    sufix_err_msg = ""
    if len(script_str) > min_len:
        sufix_err_msg = f"script has more code:\n{sufix_err_msg[min_len:-1]}"
    elif len(expected_str) > min_len:
        sufix_err_msg = f"script has more code:\n{expected_str[min_len:-1]}"
    for i in range(min_len):
        if script_str[i] != expected_str[i]:
            raise Exception(
                f"\n Different characters exist.\nscript:\n{script_str[0:i+1]}...\nexpected:\n{expected_str[0:i+1]}...\n"
            )


def test_cinn_type():
    obj = cinn.common.Float(32)
    expected = 'cinn.common.type_of("float32")'
    _assert_print(obj, expected)


def test_int_imm():
    a = ir.Expr(42)
    expected = '42'
    _assert_print(a, expected)


def test_uint_imm():
    a = ir.Expr(True)
    expected = "True"
    _assert_print(a, expected)


def test_float_imm():
    a = ir.Expr(42.5)
    expected = "42.5"
    _assert_print(a, expected)


def test_var():
    a = ir.Var("a", cinn.common.Float(32))
    expected = 'a'
    _assert_print(a, expected)


def test_buffer():
    a = ir._Buffer_.make(
        "A", cinn.common.Float(32), [ir.Expr(128), ir.Expr(128)]
    )
    expected = "A"
    _assert_print(a, expected)


def test_cast():
    pass
    # b = ir.Cast(cinn.common.Float(64), ir.Var("a", cinn.common.Float(32)))
    # print(b)


def test_binary_op():
    a = ir.Expr(ir.Var("a", cinn.common.Int(32)))
    b = ir.Expr(ir.Var("b", cinn.common.Int(32)))
    for cinn_op, op_sign in [
        (ir.Add.make, "+"),
        (ir.Sub.make, "-"),
        (ir.Mul.make, "*"),
        (ir.Div.make, "/"),
        (ir.Mod.make, "%"),
        (ir.EQ.make, "=="),
        (ir.NE.make, "!="),
        (ir.LT.make, "<"),
        (ir.LE.make, "<="),
        (ir.GT.make, ">"),
        (ir.GE.make, ">="),
    ]:
        expected = f"(a {op_sign} b)"
        c = cinn_op(a, b)
        _assert_print(c, expected)


def test_logical():
    a = ir.Expr(ir.Var("a", cinn.common.UInt(1)))
    b = ir.Expr(ir.Var("b", cinn.common.UInt(1)))
    c = ir.And.make(a, b)
    expected = "(a and b)"
    _assert_print(c, expected)
    c = ir.Or.make(a, b)
    expected = "(a or b)"
    _assert_print(c, expected)
    c = ir.Not.make(a)
    expected = "not a"
    _assert_print(c, expected)


def test_select():
    select = ir.Select.make(ir.Expr(True), ir.Expr(42), ir.Expr(-42))
    expected = "ir.Select.make(True, 42, -42)"

    _assert_print(select, expected)


def test_tensor_load():
    tensor_shape = [ir.Expr(224), ir.Expr(224), ir.Expr(3)]
    tensor = ir._Tensor_.make(
        "tensor", cinn.common.Float(32), tensor_shape, tensor_shape
    ).Expr()
    load_idx = [ir.Expr(1), ir.Expr(1), ir.Expr(2)]
    load = ir.Load.make(tensor, load_idx)
    expected = "tensor[1, 1, 2]"
    _assert_print(load, expected)


def test_ramp():
    base = ir.Expr(ir.Var("a", cinn.common.Int(32)))
    stride = ir.Expr(2)
    lanes = 32
    ramp = ir.Ramp.make(base, stride, lanes)
    expected = "ir.Ramp(a, 2, 32)"
    _assert_print(ramp, expected)


def test_broadcast():
    broadcast = ir.Broadcast.make(ir.Expr(4), 4)
    expected = "ir.Broadcast.make(4, 4)"
    _assert_print(broadcast, expected)


def test_call():
    call = lang.call_extern("sin", [ir.Expr(1.0)], {})
    expected = 'lang.call_extern("sin", [1], {})'
    _assert_print(call, expected)


def test_call_extern():
    @to_cinn_llir
    def call_sinh(A: DataArray((1, 4, 256, 512)), B: DataArray((1, 4, 256))):
        for i1 in range(1):
            for j1 in range(4):
                for k1 in range(256):
                    with ir.ScheduleBlockContext("init") as init:
                        vi, vj, vk = ir.AxisMap("SSS", [i1, j1, k1])
                        B[vi, vj, vk] = lang.call_extern(
                            "sinh", [A[vi, vi, vj, vk]], {}
                        )

    expected = """
# from cinn import ir, lang, to_cinn_llir
# from cinn.runtime.data_array import DataArray
# from cinn.schedule import IRSchedule as sch
def fn_call_sinh(A: DataArray((1, 4, 256, 512)), B: DataArray((1, 4, 256))):
    for i1 in range(0, 1):
        for j1 in range(0, 4):
            for k1 in range(0, 256):
                with ir.ScheduleBlockContext("init") as init_block:
                    vi, vj, vk = ir.AxisMap("SSS", [i1, j1, k1])
                    B[vi, vj, vk] = lang.call_extern("sinh", [A[vi, vi, vj, vk]], {})
    """
    _assert_print(call_sinh, expected)


def test_stmts():
    # include For, Store, IfThenElse, Block, ScheduleBlock, ScheduleBlockRealize, LowerFunction
    @to_cinn_llir
    def if_then_else(A: DataArray((1, 4, 256, 512)), B: DataArray((1, 4, 256))):
        for i1 in range(1):
            for j1 in range(4):
                for k1 in range(256):
                    if k1 < 128:
                        with ir.ScheduleBlockContext("if") as if_block:
                            vi, vj, vk = ir.AxisMap("SSS", [i1, j1, k1])
                            B[vi, vj, vk] = lang.call_extern(
                                "sinh", [A[vi, vi, vj, vk]], {}
                            )
                    else:
                        with ir.ScheduleBlockContext("else") as else_block:
                            vi, vj, vk = ir.AxisMap("SSS", [i1, j1, k1])
                            B[vi, vj, vk] = 0.0

    expected = """
# from cinn import ir, lang, to_cinn_llir
# from cinn.runtime.data_array import DataArray
# from cinn.schedule import IRSchedule as sch
def fn_if_then_else(A: DataArray((1, 4, 256, 512)), B: DataArray((1, 4, 256))):
    for i1 in range(0, 1):
        for j1 in range(0, 4):
            for k1 in range(0, 256):
                if ((k1 < 128)):
                    with ir.ScheduleBlockContext("if") as if_block:
                        vi, vj, vk = ir.AxisMap("SSS", [i1, j1, k1])
                        B[vi, vj, vk] = lang.call_extern("sinh", [A[vi, vi, vj, vk]], {})
                else:
                    with ir.ScheduleBlockContext("else") as else_block:
                        vi, vj, vk = ir.AxisMap("SSS", [i1, j1, k1])
                        B[vi, vj, vk] = 0
    """
    _assert_print(if_then_else, expected)


if __name__ == "__main__":
    test_cinn_type()
    test_var()
    test_int_imm()
    test_uint_imm()
    test_float_imm()
    test_buffer()
    test_cast()
    test_binary_op()
    test_logical()
    test_select()
    test_tensor_load()
    test_ramp()
    test_broadcast()
    test_call()
    test_call_extern()
    test_stmts()
