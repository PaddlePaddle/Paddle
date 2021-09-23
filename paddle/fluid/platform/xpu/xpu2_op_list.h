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

#ifdef PADDLE_WITH_XPU
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

XPUOpMap& get_kl2_ops() {
  // KL1支持的op，通过op_name, data_type, place来索引
  static XPUOpMap s_xpu2_kernels{
      {"label_smooth",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"mul", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                            pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_sub",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_sub_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_add",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_add_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_div",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_div_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_pow",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_floordiv",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_mul",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_mul_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_max",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_max_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_min",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"elementwise_min_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"momentum", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"batch_norm", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"batch_norm_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"layer_norm", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                   pOpKernelType(vartype::FP16, XPUPlace())})},
      {"layer_norm_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"mean", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                             pOpKernelType(vartype::FP16, XPUPlace())})},
      {"mean_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                  pOpKernelType(vartype::FP16, XPUPlace())})},
      {"adam", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"adamw", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"softmax_with_cross_entropy",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"softmax_with_cross_entropy_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"sum", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                            pOpKernelType(vartype::FP16, XPUPlace())})},
      {"transpose", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                  pOpKernelType(vartype::FP16, XPUPlace())})},
      {"transpose_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"transpose2", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                   pOpKernelType(vartype::FP16, XPUPlace())})},
      {"transpose2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace())})},
      {"iou_similarity",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"arg_max", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})}
      // AddMore
  };

  return s_xpu2_kernels;
}

}  // namespace platform
}  // namespace paddle
#endif
