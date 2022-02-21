// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

// CUDA and HIP use same api
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>
#include "paddle/pten/kernels/gpu/elementwise.h"
namespace pten {
template <typename InT, typename Functor>
void ReduceGrad(const GPUContext& dev_ctx,
                DenseTensor* d_out,
                DenseTensor* d_x,
                DataType out_dtype,
                Functor functor) {
  std::vector<const DenseTensor*> inputs = {d_out};
  std::vector<DenseTensor*> outputs = {d_x};
  PD_VISIT_ALL_TYPES(
      out_dtype, "LaunchBroadcastElementwiseCudaKernel", ([&] {
        LaunchBroadcastElementwiseCudaKernel<pten::ElementwiseType::kUnary,
                                             InT,
                                             data_t>(
            dev_ctx, inputs, &outputs, 0, functor);
      }));
}
}  // namespace pten
#endif
