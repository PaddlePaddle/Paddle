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
from cinn import ir


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


if __name__ == "__main__":
    test_cinn_type()
    test_var()
    test_int_imm()
    test_uint_imm()
    test_float_imm()
    test_buffer()
