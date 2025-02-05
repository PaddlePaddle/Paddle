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
const std::set<std::string>& GetPrimitiveOpNames() {
  static std::set<std::string> primitive_set = {
      /* primitives*/
      "pd_op.arange",
      "pd_op.add",
      "pd_op.subtract",
      "pd_op.multiply",
      "pd_op.divide",
      "pd_op.less_than",
      "pd_op.less_equal",
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
      "pd_op.assign_value",
      "pd_op.concat",
      "pd_op.elementwise_pow",
      "pd_op.floor",
      "pd_op.gather",
      "pd_op.gather_nd",
      "pd_op.log",
      "pd_op.logical_and",
      "pd_op.logical_or",
      "pd_op.logical_xor",
      "pd_op.logical_not",
      "pd_op.max",
      "pd_op.min",
      "pd_op.maximum",
      "pd_op.minimum",
      "pd_op.argmax",
      "pd_op.argmin",
      "pd_op.pow",
      "pd_op.prod",
      "pd_op.roll",
      "pd_op.scatter",
      "pd_op.scatter_nd_add",
      "pd_op.tile",
      "pd_op.transpose",
      "pd_op.pad",
      "pd_op.cumsum",
      "pd_op.put_along_axis",
      "pd_op.sin",
      "pd_op.cos",
      "pd_op.where",
      "pd_op.split",
      "pd_op.split_with_num",
      "pd_op.reshape",
      "pd_op.erf",
      "pd_op.tanh",
      "pd_op.cast",
      "pd_op.sign",
      "pd_op.slice",
      "pd_op.uniform",
      "pd_op.shape",
      "pd_op.shape64",
      "pd_op.full",
      "pd_op.full_int_array",
      "pd_op.full_with_tensor",
      "pd_op.if",
      "pd_op.while",
      /* Considering better performance, such ops are set as primitive ops
         temporarily*/
      "pd_op.rsqrt",
      "pd_op.sqrt",
      /* basic ops by PIR*/
      "builtin.combine",
      "builtin.slice",
      "builtin.split",
      "pd_op.feed",
      "pd_op.fetch",
      "builtin.set_parameter",
      "builtin.parameter",
      "builtin.constant",
      "pd_op.data",
      "builtin.shadow_output",
      "pd_op.sigmoid",
      "pd_op.reduce_as",
      /* skip some special ops */
      "pd_op.conv2d",
      "pd_op.pad3d",
      "pd_op.nearest_interp",
      "pd_op.squeeze",
      "pd_op.unsqueeze",
      "pd_op.select_input",
      "pd_op.top_p_sampling",
      "pd_op.tril",
      "pd_op.triu",
      "cf.yield",
      "pd_op.increment_",
  };
  return primitive_set;
}
}  // namespace paddle
