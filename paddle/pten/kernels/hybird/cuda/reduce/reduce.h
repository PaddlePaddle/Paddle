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

#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/hybird/cuda/reduce/reduce_cuda_impl.h"
namespace pten {

template <typename T,
          template <typename> class ReduceOp,
          template <typename, typename> class TransformOp>
void Reduce(const GPUContext& dev_ctx,
            const DenseTensor& x,
            bool reduce_all,
            const std::vector<int64_t>& dims,
            bool keep_dim,
            DataType out_dtype,
            DenseTensor* out) {
  std::vector<int> reduce_dims =
      pten::kernels::details::GetReduceDim(dims, x.dims().size(), reduce_all);

  int reduce_num = 1;
  for (auto i : reduce_dims) {
    reduce_num *= (x.dims())[i];
  }

  gpuStream_t stream = dev_ctx.stream();

  if (out_dtype != pten::DataType::UNDEFINED && out_dtype != x.dtype()) {
    PD_DISPATCH_FLOATING_AND_COMPLEX_AND_2_TYPES(
        pten::DataType::INT32,
        pten::DataType::INT64,
        out_dtype,
        "TensorReduceFunctorImpl",
        ([&] {
          using MPType = typename kps::details::MPTypeTrait<data_t>::Type;
          pten::kernels::TensorReduceFunctorImpl<T,
                                                 data_t,
                                                 ReduceOp,
                                                 TransformOp<T, MPType>>(
              x, out, TransformOp<T, MPType>(reduce_num), reduce_dims, stream);
        }));
  } else {
    using MPType = typename kps::details::MPTypeTrait<T>::Type;
    pten::kernels::
        TensorReduceFunctorImpl<T, T, ReduceOp, TransformOp<T, MPType>>(
            x, out, TransformOp<T, MPType>(reduce_num), reduce_dims, stream);
  }
}

}  // namespace pten

#endif
