// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/unary.h"

namespace phi {
template <typename T, typename Context>
void SumRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const IntArray& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DataType out_dtype,
                  DenseTensor* out);

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const IntArray& dims,
               DataType out_dtype,
               bool keep_dim,
               DenseTensor* out);

template <typename T, typename Context>
DenseTensor Sum(const Context& dev_ctx,
                const DenseTensor& x,
                const IntArray& axis,
                DataType dtype,
                bool keep_dim) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  SumInferMeta(x, axis, dtype, keep_dim, &meta_out);
  SumKernel<T, Context>(dev_ctx, x, axis, dtype, keep_dim, &dense_out);
  return dense_out;
}

}  // namespace phi
