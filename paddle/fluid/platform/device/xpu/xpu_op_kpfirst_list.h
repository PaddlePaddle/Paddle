/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#ifdef PADDLE_WITH_XPU_KP
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/framework/op_kernel_type.h"

namespace paddle {
namespace platform {

using vartype = paddle::framework::proto::VarType;
using pOpKernelType = paddle::framework::OpKernelType;
using XPUKernelSet =
    std::unordered_set<pOpKernelType, paddle::framework::OpKernelType::Hash>;
using XPUOpMap = std::unordered_map<std::string, XPUKernelSet>;

XPUOpMap& get_kp_ops() {
  static XPUOpMap s_xpu_kp_kernels{
      {"elementwise_add",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_div",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_sub",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_max",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_min",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_mul",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_floordiv",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace())})},
      // activation op
      {"exp", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"hard_swish", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"leaky_relu", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"softplus", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reciprocal", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"log", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sigmoid", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"relu", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elu", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"celu", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sqrt", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"square", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"silu", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"logsigmoid", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"softshrink", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"ceil", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"floor", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"log1p", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"brelu", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"soft_relu", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"softsign", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"relu6", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"hard_shrink", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"hard_sigmoid",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"swish", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"thresholded_relu",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      // bitwise logical & compare
      {"bitwise_and",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace())})},
      {"bitwise_or",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace())})},
      {"bitwise_not",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace())})},
      {"bitwise_xor",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace())})},

      {"logical_and",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace())})},
      {"logical_or", XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace())})},
      {"logical_not",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace())})},
      {"logical_xor",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace())})},

      {"less_than", XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace())})},
      {"less_equal", XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace())})},
      {"greater_than",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace())})},
      {"greater_equal",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace())})},
      {"equal", XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace())})},
      {"not_equal", XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace())})},
      // reduce op
      // {"reduce_mean", XPUKernelSet({pOpKernelType(vartype::FP32,
      // XPUPlace())})},
      // {"reduce_max", XPUKernelSet({pOpKernelType(vartype::FP32,
      // XPUPlace())})},
      // {"reduce_min", XPUKernelSet({pOpKernelType(vartype::FP32,
      // XPUPlace())})},
      // {"reduce_sum", XPUKernelSet({pOpKernelType(vartype::FP32,
      // XPUPlace())})},
      // {"reduce_prod", XPUKernelSet({pOpKernelType(vartype::FP32,
      // XPUPlace())})},
      // {"reduce_all", XPUKernelSet({pOpKernelType(vartype::BOOL,
      // XPUPlace())})},
      // {"reduce_any", XPUKernelSet({pOpKernelType(vartype::BOOL,
      // XPUPlace())})},
      // {"reduce_amax", XPUKernelSet({pOpKernelType(vartype::FP32,
      // XPUPlace())})},
      // {"reduce_amin", XPUKernelSet({pOpKernelType(vartype::FP32,
      // XPUPlace())})},
      {"pull_box_sparse",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"push_box_sparse",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"c_sync_calc_stream",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"c_sync_comm_stream",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"c_allreduce_sum",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
  };

  return s_xpu_kp_kernels;
}

}  // namespace platform
}  // namespace paddle
#endif
