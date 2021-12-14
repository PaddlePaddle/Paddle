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

#include "paddle/pten/kernels/hybird/cuda/elementwise/elementwise_broadcast.cu.h"
#include "paddle/pten/kernels/hybird/cuda/elementwise/elementwise_no_broadcast.cu.h"

namespace pten {

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchElementwiseCudaKernel(
    const paddle::platform::CUDADeviceContext &cuda_ctx,
    const std::vector<const DenseTensor *> &ins,
    std::vector<DenseTensor *> *outs,
    int axis,
    Functor func) {
  std::vector<int> dims_size;
  bool no_broadcast_flag = true;
  for (auto *in : ins) {
    no_broadcast_flag = ins[0]->dims() == in->dims();
    dims_size.emplace_back(in->dims().size());
  }
  if (no_broadcast_flag) {
    LaunchSameDimsElementwiseCudaKernel<ET, InT, OutT>(
        cuda_ctx, ins, outs, func);
  } else {
    axis = axis == -1
               ? *std::max_element(dims_size.begin(), dims_size.end()) -
                     *std::min_element(dims_size.begin(), dims_size.end())
               : axis;
    LaunchBroadcastElementwiseCudaKernel<ET, InT, OutT>(
        cuda_ctx, ins, outs, axis, func);
  }
}

}  // namespace pten
