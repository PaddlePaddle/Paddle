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
void TriangularSolveKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           bool upper,
                           bool transpose,
                           bool unitriangular,
                           DenseTensor* out);

template <typename T, typename Context>
DenseTensor TriangularSolve(const Context& ctx,
                            const DenseTensor& x,
                            const DenseTensor& y,
                            bool upper,
                            bool transpose,
                            bool unitriangular) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  TriangularSolveInferMeta(x, y, upper, transpose, unitriangular, &meta_out);
  TriangularSolveKernel<T, Context>(
      ctx, x, y, upper, transpose, unitriangular, &dense_out);
  return dense_out;
}

}  // namespace phi
