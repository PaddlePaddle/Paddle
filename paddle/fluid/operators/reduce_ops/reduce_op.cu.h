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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif

#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/framework/tensor.h"

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/hybird/cuda/reduce/reduce_cuda_impl.h"

namespace paddle {
namespace operators {

template <typename Tx, typename Ty, template <typename> class ReduceOp,
          typename TransformOp>
void TensorReduceFunctorImpl(const framework::Tensor& x, framework::Tensor* y,
                             const TransformOp& transform,
                             const std::vector<int>& origin_reduce_dims,
                             gpuStream_t stream) {
  y->mutable_data<Ty>(x.place());

  auto pt_x = paddle::experimental::MakePtenDenseTensor(x);
  auto pt_y = paddle::experimental::MakePtenDenseTensor(*y);

  pten::kernels::TensorReduceFunctorImpl<Tx, Ty, ReduceOp, TransformOp>(
      *pt_x.get(), pt_y.get(), transform, origin_reduce_dims, stream);
}

}  // namespace operators
}  // namespace paddle
