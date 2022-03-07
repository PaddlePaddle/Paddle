# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

nodemakers = {}
jvpmakers = {}
transposemakers = {}


def add_maker(x, y):
    out_var = make_var()
    node = PrimNode(ADD, out_var, x, y)
    node, out_var


def sub_maker(x, y):
    out_var = make_var()
    node = PrimNode(SUB, out_var, x, y)
    return node, out_var


def mul_maker(x, y):
    out_var = make_var()
    node = PrimNode(MUL, out_var, x, y)
    return node, out_var


nodemakers[ADD] = add_maker
nodemakers[SUB] = sub_maker
nodemakers[MUL] = mul_maker


def add_jvpmaker(x, y):
    return lambda tx, ty: ADD(tx, ty)


def sub_jvpmaker(x, y):
    return lambda tx, ty: SUB(tx, ty)


def mul_jvpmaker(x, y):
    return lambda tx, ty: ADD(MUL(x, ty), MUL(tx, y))


jvpmakers[ADD] = add_jvpmaker
jvpmakers[SUB] = sub_jvpmaker
jvpmakers[MUL] = mul_jvpmaker


def add_transposemaker(x, y):
    assert x.is_tangent and y.is_tangent
    return lambda t: t, lambda t: t


def sub_transposemaker(x, y):
    assert x.is_tangent and y.is_tangent
    return lambda t: t, lambda t: NEG(t)


def mul_transposemaker(x, y):
    assert x.is_tangent ^ y.is_tangent
    if x.is_tangent:
        return lambda t: (MUL(t, y), None)
    else:
        return lambda t: (None, MUL(x, t))


transposemakers[ADD] = add_transposemaker
transposemakers[SUB] = sub_transposemaker
transposemakers[MUL] = mul_transposemaker
