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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void MatmulKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  DenseTensor* out);

// In order to be compatible with `mul` op in fluid,
// it is no longer used in 2.x API
template <typename T, typename Context>
void MatmulWithFlattenKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             int x_num_col_dims,
                             int y_num_col_dims,
                             DenseTensor* out);

template <typename T, typename Context>
DenseTensor Matmul(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   bool transpose_x = false,
                   bool transpose_y = false) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  MatmulInferMeta(x, y, transpose_x, transpose_y, &meta_out);
  MatmulKernel<T, Context>(dev_ctx, x, y, transpose_x, transpose_y, &dense_out);
  return dense_out;
}

}  // namespace phi
