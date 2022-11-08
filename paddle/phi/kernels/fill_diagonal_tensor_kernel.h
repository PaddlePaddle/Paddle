// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

namespace phi {

template <typename T, typename Context>
void FillDiagonalTensorKernel(const Context& ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              int64_t offset,
                              int dim1,
                              int dim2,
                              DenseTensor* out);

template <typename T, typename Context>
DenseTensor FillDiagonalTensor(const Context& ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               int64_t offset,
                               int dim1,
                               int dim2) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  FillDiagonalTensorInferMeta(x, y, offset, dim1, dim2, &meta_out);
  FillDiagonalTensorKernel<T, Context>(
      ctx, x, y, offset, dim1, dim2, &dense_out);
  return dense_out;
}

void CalMatDims(phi::DDim out_dims,
                int dim1,
                int dim2,
                int64_t* offset,
                int64_t* new_dims,
                int64_t* strides,
                int64_t* matoffset);

}  // namespace phi
