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
#include "paddle/phi/common/data_type.h"

namespace phi {
namespace backends {
namespace xpu {

using XPUKernelSet = std::unordered_set<phi::DataType>;
using XPUOpMap = std::unordered_map<std::string, XPUKernelSet>;

XPUOpMap& get_kp_ops() {
  static XPUOpMap s_xpu_kp_kernels{
      {"elementwise_add", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_div", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_sub", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_max", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_min", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_mul", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elementwise_floordiv", XPUKernelSet({phi::DataType::INT32})},
      // activation op
      {"exp", XPUKernelSet({phi::DataType::FLOAT32})},
      {"hard_swish", XPUKernelSet({phi::DataType::FLOAT32})},
      {"leaky_relu", XPUKernelSet({phi::DataType::FLOAT32})},
      {"softplus", XPUKernelSet({phi::DataType::FLOAT32})},
      {"reciprocal", XPUKernelSet({phi::DataType::FLOAT32})},
      {"log", XPUKernelSet({phi::DataType::FLOAT32})},
      {"sigmoid", XPUKernelSet({phi::DataType::FLOAT32})},
      {"relu", XPUKernelSet({phi::DataType::FLOAT32})},
      {"elu", XPUKernelSet({phi::DataType::FLOAT32})},
      {"celu", XPUKernelSet({phi::DataType::FLOAT32})},
      {"sqrt", XPUKernelSet({phi::DataType::FLOAT32})},
      {"square", XPUKernelSet({phi::DataType::FLOAT32})},
      {"silu", XPUKernelSet({phi::DataType::FLOAT32})},
      {"logsigmoid", XPUKernelSet({phi::DataType::FLOAT32})},
      {"softshrink", XPUKernelSet({phi::DataType::FLOAT32})},
      {"ceil", XPUKernelSet({phi::DataType::FLOAT32})},
      {"floor", XPUKernelSet({phi::DataType::FLOAT32})},
      {"log1p", XPUKernelSet({phi::DataType::FLOAT32})},
      {"brelu", XPUKernelSet({phi::DataType::FLOAT32})},
      {"soft_relu", XPUKernelSet({phi::DataType::FLOAT32})},
      {"softsign", XPUKernelSet({phi::DataType::FLOAT32})},
      {"relu6", XPUKernelSet({phi::DataType::FLOAT32})},
      {"hard_shrink", XPUKernelSet({phi::DataType::FLOAT32})},
      {"hard_sigmoid", XPUKernelSet({phi::DataType::FLOAT32})},
      {"swish", XPUKernelSet({phi::DataType::FLOAT32})},
      {"thresholded_relu", XPUKernelSet({phi::DataType::FLOAT32})},
      // bitwise logical & compare
      {"bitwise_and",
       XPUKernelSet({phi::DataType::INT32, phi::DataType::BOOL})},
      {"bitwise_or", XPUKernelSet({phi::DataType::INT32, phi::DataType::BOOL})},
      {"bitwise_not",
       XPUKernelSet({phi::DataType::INT32, phi::DataType::BOOL})},
      {"bitwise_xor",
       XPUKernelSet({phi::DataType::INT32, phi::DataType::BOOL})},

      {"logical_and", XPUKernelSet({phi::DataType::INT32})},
      {"logical_or", XPUKernelSet({phi::DataType::INT32})},
      {"logical_not", XPUKernelSet({phi::DataType::INT32})},
      {"logical_xor", XPUKernelSet({phi::DataType::INT32})},

      {"less_than", XPUKernelSet({phi::DataType::INT32})},
      {"less_equal", XPUKernelSet({phi::DataType::INT32})},
      {"greater_than", XPUKernelSet({phi::DataType::INT32})},
      {"greater_equal", XPUKernelSet({phi::DataType::INT32})},
      {"equal", XPUKernelSet({phi::DataType::INT32})},
      {"not_equal", XPUKernelSet({phi::DataType::INT32})},
      {"pull_box_sparse", XPUKernelSet({phi::DataType::FLOAT32})},
      {"push_box_sparse", XPUKernelSet({phi::DataType::FLOAT32})},
      {"c_sync_calc_stream", XPUKernelSet({phi::DataType::FLOAT32})},
      {"c_sync_comm_stream", XPUKernelSet({phi::DataType::FLOAT32})},
      {"c_allreduce_sum", XPUKernelSet({phi::DataType::FLOAT32})},
  };

  return s_xpu_kp_kernels;
}

}  // namespace xpu
}  // namespace backends
}  // namespace phi
#endif
