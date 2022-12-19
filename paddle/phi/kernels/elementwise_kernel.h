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
void FMaxRawKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   int axis,
                   DenseTensor* out);

template <typename T, typename Context>
void FMaxKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                DenseTensor* out);

template <typename T, typename Context>
void FMinRawKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   int axis,
                   DenseTensor* out);

template <typename T, typename Context>
void FMinKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                DenseTensor* out);

template <typename T, typename Context>
void MaximumRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      int axis,
                      DenseTensor* out);

template <typename T, typename Context>
void MaximumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out);

template <typename T, typename Context>
void MinimumRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      int axis,
                      DenseTensor* out);

template <typename T, typename Context>
void MinimumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out);

template <typename T, typename Context>
void RemainderRawKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        int axis,
                        DenseTensor* out);

template <typename T, typename Context>
void RemainderKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out);

template <typename T, typename Context>
void FloorDivideRawKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          int axis,
                          DenseTensor* out);

template <typename T, typename Context>
void FloorDivideKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out);

template <typename T, typename Context>
void ElementwisePowRawKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             int axis,
                             DenseTensor* out);

template <typename T, typename Context>
void ElementwisePowKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out);

template <typename T, typename Context>
void ElementwiseHeavisideRawKernel(const Context& dev_ctx,
                                   const DenseTensor& x,
                                   const DenseTensor& y,
                                   int axis,
                                   DenseTensor* out);

template <typename T, typename Context>
void ElementwiseHeavisideKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                const DenseTensor& y,
                                DenseTensor* out);

template <typename T, typename Context>
DenseTensor Maximum(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  ElementwiseInferMeta(x, y, &meta_out);
  MaximumKernel<T, Context>(dev_ctx, x, y, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor Minimum(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  ElementwiseInferMeta(x, y, &meta_out);
  MinimumKernel<T, Context>(dev_ctx, x, y, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor Remainder(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  ElementwiseInferMeta(x, y, &meta_out);
  RemainderKernel<T, Context>(dev_ctx, x, y, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor FloorDivide(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  ElementwiseInferMeta(x, y, &meta_out);
  FloorDivideKernel<T, Context>(dev_ctx, x, y, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor ElementwiseHeaviside(const Context& dev_ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  ElementwiseInferMeta(x, y, &meta_out);
  ElementwiseHeavisideKernel<T, Context>(dev_ctx, x, y, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor ElementwisePow(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  ElementwiseInferMeta(x, y, &meta_out);
  ElementwisePowKernel<T, Context>(dev_ctx, x, y, &dense_out);
  return dense_out;
}

}  // namespace phi
