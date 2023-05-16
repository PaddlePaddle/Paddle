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

#include <vector>

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const IntArray& shape,
                const Scalar& val,
                DataType dtype,
                DenseTensor* out);

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const Scalar& val,
                    DataType dtype,
                    DenseTensor* out);

// In order to be compatible with fill_constant_batch_size_like op
// that are still used in the 2.x APIs
template <typename T, typename Context>
void FullBatchSizeLikeKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const std::vector<int>& shape,
                             const Scalar& val,
                             DataType dtype,
                             int x_batch_size_dim,
                             int out_batch_size_dim,
                             DenseTensor* out);

template <typename T, typename Context>
void Full(const Context& dev_ctx,
          const IntArray& shape,
          const Scalar& val,
          DenseTensor* out) {
  FullKernel<T, Context>(
      dev_ctx, shape, val, phi::CppTypeToDataType<T>::Type(), out);
}

template <typename T, typename Context>
DenseTensor Full(const Context& dev_ctx,
                 const IntArray& shape,
                 const Scalar& val) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  DataType dtype = phi::CppTypeToDataType<T>::Type();
  CreateInferMeta(shape, dtype, &meta_out);
  FullKernel<T, Context>(dev_ctx, shape, val, dtype, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor FullLike(const Context& dev_ctx,
                     const DenseTensor& x,
                     const Scalar& val) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  DataType dtype = phi::CppTypeToDataType<T>::Type();
  CreateLikeInferMeta(x, dtype, &meta_out);
  FullLikeKernel<T, Context>(dev_ctx, x, val, dtype, &dense_out);
  return dense_out;
}

}  // namespace phi
