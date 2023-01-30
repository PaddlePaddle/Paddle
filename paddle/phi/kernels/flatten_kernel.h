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
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
<<<<<<< HEAD
void FlattenInferKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        int start_axis,
                        int stop_axis,
                        DenseTensor* out);

template <typename T, typename Context>
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
void FlattenKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   int start_axis,
                   int stop_axis,
<<<<<<< HEAD
                   DenseTensor* out,
                   DenseTensor* xshape);
=======
                   DenseTensor* out);

template <typename T, typename Context>
void FlattenWithXShape(const Context& dev_ctx,
                       const DenseTensor& x,
                       int start_axis,
                       int stop_axis,
                       DenseTensor* out,
                       DenseTensor* xshape);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

template <typename T, typename Context>
DenseTensor Flatten(const Context& dev_ctx,
                    const DenseTensor& x,
                    int start_axis,
                    int stop_axis) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  FlattenInferMeta(x, start_axis, stop_axis, &meta_out);
<<<<<<< HEAD
  FlattenInferKernel<T, Context>(dev_ctx, x, start_axis, stop_axis, &dense_out);
=======
  FlattenKernel<T, Context>(dev_ctx, x, start_axis, stop_axis, &dense_out);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  return dense_out;
}

}  // namespace phi
