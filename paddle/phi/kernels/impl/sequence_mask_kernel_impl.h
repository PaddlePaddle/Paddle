/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#if defined(__NVCC__) || defined(__HIPCC__)
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#else
#include <algorithm>
#endif

#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/sequence_mask.h"

namespace phi {

template <typename T, typename Context>
void SequenceMaskKernel(const Context& ctx,
                        const DenseTensor& x,
                        const paddle::optional<DenseTensor>& max_len_tensor,
                        int maxlen,
                        int out_dtype,
                        DenseTensor* y) {
  if (max_len_tensor) {
    bool is_gpu_place = ctx.GetPlace().GetType() == phi::AllocationType::GPU;
    if (is_gpu_place) {
      phi::DenseTensor temp;
      phi::Copy(ctx, *max_len_tensor.get_ptr(), phi::CPUPlace(), false, &temp);
      maxlen = *temp.data<int32_t>();
    } else {
      maxlen = *max_len_tensor.get_ptr()->data<int32_t>();
    }

    auto y_dim = phi::vectorize<int>(x.dims());
    y_dim.push_back(maxlen);
    y->Resize(phi::make_ddim(y_dim));

    PADDLE_ENFORCE_GT(
        maxlen,
        0,
        phi::errors::InvalidArgument(
            "Input(MaxLenTensor) value should be greater than 0. But "
            "received Input(MaxLenTensor) value = %d.",
            maxlen));
  }

  auto* x_data = x.data<T>();
  auto x_numel = x.numel();

  if (maxlen < 0) {
    if (x_numel == 0) {
      maxlen = 0;
    } else {
#if defined(__NVCC__) || defined(__HIPCC__)
      VLOG(10)
          << "SequenceMaskOp on GPU may be slow when maxlen is not provided.";
      maxlen = static_cast<int>(
          thrust::reduce(thrust::device_pointer_cast(x_data),
                         thrust::device_pointer_cast(x_data) + x_numel,
                         static_cast<T>(0),
                         thrust::maximum<T>()));
#else
      maxlen = static_cast<int>(*std::max_element(x_data, x_data + x_numel));
#endif
    }
    auto y_dim = phi::vectorize<int>(x.dims());
    y_dim.push_back(maxlen);
    y->Resize(phi::make_ddim(y_dim));
  }

  phi::VisitDataType(phi::TransToPhiDataType(out_dtype),
                     phi::funcs::SequenceMaskFunctor<Context, T>(
                         ctx, x_data, y, x_numel * maxlen, maxlen));
}
}  // namespace phi
