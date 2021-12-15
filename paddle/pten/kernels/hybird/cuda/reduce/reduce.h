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

// CUDA and HIP use same api
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include "paddle/pten/common/scalar.h"
#include "paddle/pten/core/dense_tensor.h"

#include "paddle/fluid/platform/device_context.h"
#include "paddle/pten/kernels/hybird/cuda/reduce/reduce_cuda_impl.h"

namespace pten {

using CUDAContext = paddle::platform::CUDADeviceContext;

static inline std::vector<int64_t> GetReduceDim(
    const std::vector<int64_t>& dims, int dim_size, bool reduce_all) {
  std::vector<int64_t> reduce_dims;
  if (reduce_all) {
    reduce_dims.resize(dim_size);
    int reduce_size = reduce_dims.size();
    for (int i = 0; i < reduce_size; ++i) {
      reduce_dims[i] = i;
    }
  } else {
    for (auto e : dims) {
      PADDLE_ENFORCE_LT(e,
                        dim_size,
                        paddle::platform::errors::InvalidArgument(
                            "ReduceOp: invalid axis, when x_dims is %d, "
                            "axis[i] should less than x_dims, but got %d.",
                            dim_size,
                            e));
      reduce_dims.push_back(e >= 0 ? e : e + dim_size);
    }
  }
  return reduce_dims;
}

template <typename T, template <typename, typename> class ReduceFunctor>
void Reduce(const CUDAContext& dev_ctx,
            const DenseTensor& x,
            bool reduce_all,
            const std::vector<int64_t>& dims,
            bool keep_dim,
            DataType out_dtype,
            DenseTensor* out) {
  std::vector<int64_t> reduce_dims =
      GetReduceDim(dims, x.dims().size(), reduce_all);

  gpuStream_t stream = dev_ctx.stream();

  if (out_dtype != pten::DataType::UNDEFINED) {
    PD_DISPATCH_FLOATING_AND_INTEGRAL_AND_COMPLEX_TYPES(
        out_dtype, "TensorReduceFunctorImpl", ([&] {
          pten::detail::TensorReduceFunctorImpl<T, data_t, ReduceFunctor>(
              x, out, reduce_dims, stream);
        }));
  } else {
    pten::detail::TensorReduceFunctorImpl<T, T, ReduceFunctor>(
        x, out, reduce_dims, stream);
  }
}

}  // namespace pten

#endif
