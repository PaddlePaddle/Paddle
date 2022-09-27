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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DataType out_dtype,
                DenseTensor* out);

template <typename T, typename Context>
DenseTensor Cast(const Context& dev_ctx,
                 const DenseTensor& x,
                 DataType out_dtype) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  CastInferMeta(x, out_dtype, &meta_out);
  CastKernel<T, Context>(dev_ctx, x, out_dtype, &dense_out);
  return dense_out;
}

}  // namespace phi
