// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <iostream>
#include <set>
#include <string>

namespace paddle {
static std::set<std::string> primitive_set = {
    /* primitives*/
    "pd_op.add",
    "pd_op.subtract",
    "pd_op.multiply",
    "pd_op.divide",
    "pd_op.less_equal",
    "pd_op.less_than",
    "pd_op.equal",
    "pd_op.not_equal",
    "pd_op.greater_equal",
    "pd_op.greater_than",
    "pd_op.bitwise_and",
    "pd_op.bitwise_not",
    "pd_op.bitwise_or",
    "pd_op.bitwise_xor",
    "pd_op.exp",
    "pd_op.scale",
    "pd_op.matmul",
    "pd_op.expand",
    "pd_op.sum",
    "pd_op.abs",
    "pd_op.assign",
    "pd_op.concat",
    "pd_op.elementwise_pow",
    "pd_op.floor",
    "pd_op.gather",
    "pd_op.gather_nd",
    "pd_op.log",
    "pd_op.max",
    "pd_op.min",
    "pd_op.maximum",
    "pd_op.minimum",
    "pd_op.prod",
    "pd_op.roll",
    "pd_op.scatter",
    "pd_op.scatter_nd_add",
    "pd_op.tile",
    "pd_op.transpose",
    "pd_op.pad",
    "pd_op.cumsum",
    "pd_op.put_along_axis",
    "pd_op.equal",
    "pd_op.greater_than",
    "pd_op.less_equal",
    "pd_op.sin",
    "pd_op.cos",
    "pd_op.where",
    "pd_op.split",
    "pd_op.reshape",
    "pd_op.erf",
    "pd_op.tanh",
    "pd_op.cast",
    "pd_op.sign",
    "pd_op.slice",
    "pd_op.uniform",
    "pd_op.shape",
    "pd_op.full",
    "pd_op.full_int_array",
    "pd_op.if",
    "pd_op.while",
    /* basic ops by PIR*/
    "pd_op.data",
    "pd_op.combine",
    /* skip some special ops */
    "pd_op.squeeze",
    "pd_op.unsqueeze",
    "pd_op.top_p_sampling",
    "pd_op.tril",
};
}  // namespace paddle
