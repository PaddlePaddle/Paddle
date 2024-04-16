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

from .primreg import (
    lookup_fn,
    lookup_jvp,
    lookup_orig2prim,
    lookup_prim2orig,
    lookup_transpose,
)


def _orig2prim(op, *args):
    _lowerrule = lookup_orig2prim(op.type)
    return _lowerrule(op, *args)


def _prim2orig(op, *args):
    _lowerrule = lookup_prim2orig(op.type)
    return _lowerrule(op, *args)


def _jvp(op, *args):
    _jvprule = lookup_jvp(op.type)
    return _jvprule(op, *args)


def _transpose(op, dot_checker, *args):
    _transposerule = lookup_transpose(op.type)
    return _transposerule(op, dot_checker, *args)


def linear_jvp(op, *args, **kwargs):
    fn = lookup_fn(op.type)
    out_dot = fn(*args, **kwargs)
    return out_dot


# Register orig2prim lower rules
"""
These original ops are fully supported:

elementwise_add
elementwise_sub
elementwise_mul
tanh
fill_zeros_like
fill_any_like
sum
index_select
scale
assign
sqrt
log
select
equal
elementwise_pow
dropout
uniform_random

These original ops are partially supported:

matmul_v2
reshape2
concat
slice
p_norm
"""
