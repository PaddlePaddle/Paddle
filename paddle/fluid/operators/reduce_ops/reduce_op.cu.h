// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#include "paddle/fluid/framework/tensor.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
namespace paddle {
namespace operators {

template <typename Tx, typename Ty, template <typename> class ReduceOp,
          typename TransformOp>
void TensorReduceImpl(const platform::CUDADeviceContext& dev_ctx,
                      const framework::Tensor& x, framework::Tensor* y,
                      const TransformOp& transform,
                      const std::vector<int>& origin_reduce_dims,
                      gpuStream_t stream, bool is_mean = false) {
  y->mutable_data<Ty>(x.place());

  phi::funcs::ReduceKernel<Tx, Ty, ReduceOp, TransformOp>(
      static_cast<const phi::GPUContext&>(dev_ctx), x, y, transform,
      origin_reduce_dims, is_mean);
}

}  // namespace operators
}  // namespace paddle
