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

#include "paddle/phi/kernels/funcs/reduce_function.h"

namespace phi {

template <typename T,
          template <typename> class ReduceOp,
          template <typename, typename> class TransformOp>
void Reduce(const KPDevice& dev_ctx,
            const DenseTensor& x,
            bool reduce_all,
            const std::vector<int64_t>& dims,
            bool keep_dim,
            DataType out_dtype,
            DenseTensor* out,
            bool is_mean = false) {
  std::vector<int> reduce_dims =
      phi::funcs::details::GetReduceDim(dims, x.dims().size(), reduce_all);

  int reduce_num = 1;
  for (auto i : reduce_dims) {
    reduce_num *= (x.dims())[i];
  }

  if (out_dtype != phi::DataType::UNDEFINED && out_dtype != x.dtype()) {
    auto tmp_tensor = phi::Cast<T>(dev_ctx, x, out_dtype);
    PD_VISIT_BOOL_AND_FLOATING_AND_COMPLEX_AND_3_TYPES(
        phi::DataType::INT32,
        phi::DataType::INT64,
        phi::DataType::FLOAT16,
        out_dtype,
        "ReduceKernel",
        ([&] {
          using MPType = typename kps::details::MPTypeTrait<data_t>::Type;
          phi::funcs::ReduceKernel<data_t,
                                   data_t,
                                   ReduceOp,
                                   TransformOp<data_t, MPType>>(
              dev_ctx,
              tmp_tensor,
              out,
              TransformOp<data_t, MPType>(reduce_num),
              reduce_dims,
              is_mean);
        }));
  } else {
    using MPType = typename kps::details::MPTypeTrait<T>::Type;
    phi::funcs::ReduceKernel<T, T, ReduceOp, TransformOp<T, MPType>>(
        dev_ctx,
        x,
        out,
        TransformOp<T, MPType>(reduce_num),
        reduce_dims,
        is_mean);
  }
}
}  // namespace phi

#endif
